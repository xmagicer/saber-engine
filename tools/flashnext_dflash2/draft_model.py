"""BF16 implementation of Inco's Flash-Next DFlash2 checkpoint. No target forward is present."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def checkpoint_tensor(root, index, name):
    filename = index[name]
    with safe_open(str(root / filename), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


class DraftModel:
    def __init__(
        self,
        weights,
        checkpoint,
        device,
        vocab=None,
        graphs=True,
        mat_dtype=torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.dtype = mat_dtype
        self.graphs = graphs
        weights = Path(weights)
        checkpoint = Path(checkpoint)
        cfg = json.loads((weights / "config.json").read_text())
        dc = cfg["dflash_config"]
        assert dc == {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248077,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 15, 25, 35, 45],
        }
        assert (
            cfg["hidden_size"],
            cfg["num_hidden_layers"],
            cfg["head_dim"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["sliding_window"],
            cfg["vocab_size"],
            cfg["intermediate_size"],
        ) == (2560, 5, 128, 32, 8, 2048, 248320, 7680)
        self.h = 2560
        self.nq = 32
        self.nkv = 8
        self.dim = 128
        self.block = 8
        self.window = 2048
        self.eps = cfg["rms_norm_eps"]
        self.mask_id = dc["mask_token_id"]
        self.vsize = cfg["vocab_size"]
        self.pos_inv = cfg["rope_parameters"]["rope_theta"] ** (
            -torch.arange(0, 128, 2, device=self.device, dtype=torch.float32) / 128.0
        )
        with safe_open(
            str(weights / "model.safetensors"), framework="pt", device="cpu"
        ) as f:
            self.w = {
                k: f.get_tensor(k).to(device=self.device, dtype=self.dtype)
                for k in f.keys()
            }
        assert self.w["fc.weight"].shape == (2560, 12800)
        for layer in range(5):
            p = f"layers.{layer}."

            def concat(key, names):
                self.w[p + key] = torch.cat(
                    [self.w.pop(p + n) for n in names], dim=0
                ).contiguous()

            concat(
                "qkv",
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
            )
            concat("gate_up", ["mlp.gate_proj.weight", "mlp.up_proj.weight"])
        files = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.embed = checkpoint_tensor(
            checkpoint, files, "model.language_model.embed_tokens.weight"
        )
        borrowed_head = checkpoint_tensor(checkpoint, files, "lm_head.weight")
        assert (
            self.embed.shape == borrowed_head.shape == (248320, 2560)
            and self.embed.dtype == borrowed_head.dtype == torch.bfloat16
        )
        if vocab is None:
            selected = torch.arange(self.vsize, dtype=torch.int64)
        else:
            v = Path(vocab)
            assert v.stat().st_size % 4 == 0
            selected = (
                torch.from_file(
                    str(v), shared=False, size=v.stat().st_size // 4, dtype=torch.int32
                )
                .long()
                .sort()
                .values
            )
            assert (
                selected.numel() >= 16
                and 0 <= int(selected[0])
                and int(selected[-1]) < self.vsize
                and torch.all(selected[1:] > selected[:-1])
            )
        self.vocab = selected.to(self.device)
        self.head = torch.empty(
            (len(selected), self.h), dtype=self.dtype, device=self.device
        )
        for i in range(0, len(selected), 4096):
            self.head[i : i + 4096].copy_(
                torch.index_select(borrowed_head, 0, selected[i : i + 4096])
            )
        self.keys = [
            torch.zeros(self.window, 8, 128, device=self.device, dtype=self.dtype)
            for _ in range(5)
        ]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.tag = torch.full(
            (self.window,), -(1 << 40), dtype=torch.int64, device=self.device
        )
        self.mask_emb = self.embed[self.mask_id].to(
            device=self.device, dtype=self.dtype
        )
        self.query = self.mask_emb.expand(8, -1).clone()
        self.qpos = torch.empty(1, device=self.device, dtype=torch.int64)
        self.qid = torch.empty(1, device=self.device, dtype=torch.int64)
        self.eight = torch.arange(8, device=self.device, dtype=torch.int64)
        self.graph_cache = {}
        self.append_cache = {}
        self.pool = None
        self.next_pos = 0
        # Probe once before capturing; ROCm releases vary in support for BF16 input -> FP32 output.
        self.fp32_mm = False
        if self.dtype == torch.bfloat16 and self.device.type == "cuda":
            try:
                torch.mm(self.query[:1], self.head[:32].T, out_dtype=torch.float32)
                self.fp32_mm = True
            except (TypeError, RuntimeError):
                pass

    def norm(self, x, weight):
        f = x.float()
        return (
            f
            * torch.rsqrt(f.square().mean(-1, keepdim=True) + self.eps)
            * weight.float()
        ).to(self.dtype)

    def angle(self, positions):
        f = positions.float()[:, None] * self.pos_inv[None, :]
        f = torch.cat((f, f), dim=-1)[:, None, :]
        return f.cos(), f.sin()

    def rope(self, x, c, s):
        a, b = x.float().chunk(2, dim=-1)
        return (x.float() * c + torch.cat((-b, a), dim=-1) * s).to(self.dtype)

    def commit(self, x, pos):
        base = self.norm(F.linear(x, self.w["fc.weight"]), self.w["hidden_norm.weight"])
        c, s = self.angle(pos)
        idx = pos.remainder(self.window)
        for l in range(5):
            p = f"layers.{l}."
            kv = F.linear(base, self.w[p + "qkv"][4096:]).view(-1, 2, 8, 128)
            key = self.rope(
                self.norm(kv[:, 0], self.w[p + "self_attn.k_norm.weight"]), c, s
            )
            self.keys[l].index_copy_(0, idx, key)
            self.values[l].index_copy_(0, idx, kv[:, 1])
        self.tag.index_copy_(0, idx, pos)

    def capture(self, call):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, stream=stream, pool=self.pool):
            result = call()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        return graph, result

    def reset(self):
        self.tag.fill_(-(1 << 40))
        self.next_pos = 0

    def append(self, position, features):
        assert position == self.next_pos and 1 <= len(features) <= 2048
        assert (
            features.shape == (len(features), 12800)
            and features.dtype == torch.bfloat16
        )
        n = len(features)
        if self.graphs and n <= 5:
            if n not in self.append_cache:
                x = features.to(device=self.device, dtype=self.dtype).clone()
                pp = torch.arange(position, position + n, device=self.device)
                g, _ = self.capture(lambda: self.commit(x, pp))
                self.append_cache[n] = (x, pp, g)
            x, pp, g = self.append_cache[n]
            x.copy_(features)
            torch.arange(position, position + n, out=pp)
            g.replay()
        else:
            for i in range(0, n, 256):
                x = features[i : i + 256].to(device=self.device, dtype=self.dtype)
                pp = torch.arange(
                    position + i, position + i + len(x), device=self.device
                )
                self.commit(x, pp)
        self.next_pos += n

    def convolve(self, x, coeff, base):
        # The first row has no predecessor in the proposed block.
        x32 = x.float()
        shift = F.pad(x32[:-1], (0, 0, 1, 0))
        c = coeff.float().repeat_interleave(16, dim=-1)
        base = base.float()
        return ((base[0] + c[:, 0]) * x32 + (base[1] + c[:, 1]) * shift).to(self.dtype)

    def prepare_conv(self, x, p):
        c = F.linear(x, self.w[p + "kernel_projection.weight"]).reshape(8, 2, 2, 160)
        return self.convolve(x, c[:, 0], self.w[p + "base_kernel"][0]), c[:, 1]

    def body(self, cap, width):
        w = self.w
        h = self.query
        pos = self.qpos + self.eight
        c, s = self.angle(pos)
        kpos = torch.cat((self.tag[:cap], pos))
        visible = (pos[:, None] - kpos[None, :]).abs() < self.window
        for l in range(5):
            p = f"layers.{l}."
            pa = p + "self_attn."
            u = self.norm(h, w[p + "input_layernorm.weight"])
            u, c1 = self.prepare_conv(u, p + "attention_conv.")
            q, k, v = F.linear(u, w[p + "qkv"]).split((4096, 1024, 1024), dim=-1)
            q = self.rope(self.norm(q.view(8, 32, 128), w[pa + "q_norm.weight"]), c, s)
            k = self.rope(self.norm(k.view(8, 8, 128), w[pa + "k_norm.weight"]), c, s)
            K = torch.cat((self.keys[l][:cap], k), dim=0).transpose(0, 1).contiguous()
            V = (
                torch.cat((self.values[l][:cap], v.view(8, 8, 128)), dim=0)
                .transpose(0, 1)
                .contiguous()
            )
            # GQA sharing without materializing 4 physical copies of every K/V row.
            Q = q.reshape(8, 8, 4, 128).permute(1, 2, 0, 3).reshape(8, 32, 128)
            score = (
                torch.bmm(Q, K.transpose(1, 2))
                .to(torch.float32)
                .mul_(128**-0.5)
                .view(8, 4, 8, -1)
            )
            prob = (
                score.masked_fill(~visible[None, None, :, :], -torch.inf)
                .softmax(-1)
                .to(self.dtype)
                .reshape(8, 32, -1)
            )
            out = (
                torch.bmm(prob, V)
                .view(8, 4, 8, 128)
                .permute(2, 0, 1, 3)
                .reshape(8, 4096)
            )
            out = F.linear(out, w[pa + "o_proj.weight"])
            h = h + self.convolve(out, c1, w[p + "attention_conv.base_kernel"][1])
            u = self.norm(h, w[p + "post_attention_layernorm.weight"])
            u, c2 = self.prepare_conv(u, p + "mlp_conv.")
            gate, up = F.linear(u, w[p + "gate_up"]).chunk(2, dim=-1)
            out = F.linear(
                (F.silu(gate.float()) * up.float()).to(self.dtype),
                w[p + "mlp.down_proj.weight"],
            )
            h = h + self.convolve(out, c2, w[p + "mlp_conv.base_kernel"][1])
        out = self.norm(h, w["norm.weight"])[1 : width + 1]
        return self.select_tokens(out)

    def select_tokens(self, out):
        """Select the greedy predecessor-conditioned path from the next block rows."""
        w = self.w
        logits = (
            torch.mm(out, self.head.T, out_dtype=torch.float32)
            if self.fp32_mm
            else F.linear(out, self.head).float()
        )
        # Stable tie resolution is specified on ascending target vocab IDs.
        positions = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :16]
        cand, ordr = self.vocab[positions].sort(dim=-1)
        unary = logits.gather(1, positions).gather(1, ordr)
        hp = F.linear(out, w["candidate_selector.hidden_projection.weight"]).float()
        pred_ids = torch.cat((self.qid.expand(1, 16), cand[:-1]), dim=0)
        A = (
            w["candidate_selector.predecessor_codebook"][pred_ids].float()
            * hp[:, None, :]
        )
        B = w["candidate_selector.successor_codebook"][cand].float()
        cond = unary[:, None, :] + torch.bmm(A, B.transpose(1, 2))
        index = cond.argmax(-1)
        chosen = []
        lane = torch.zeros((1,), device=self.device, dtype=torch.int64)
        for i in range(out.shape[0]):
            lane = index[i].gather(0, lane)
            chosen.append(cand[i].gather(0, lane))
        return torch.cat(chosen)

    def propose(self, position, anchor, width):
        assert (
            position == self.next_pos
            and position > 0
            and 0 <= anchor < self.vsize
            and 1 <= width <= 4
        )
        self.query[:1].copy_(self.embed[anchor : anchor + 1])
        self.qpos.fill_(position)
        self.qid.fill_(anchor)
        cap = next(n for n in (128, 512, 2048) if n >= min(self.window, position))
        if self.graphs:
            key = (cap, width)
            if key not in self.graph_cache:
                self.graph_cache[key] = self.capture(lambda: self.body(cap, width))
            g, res = self.graph_cache[key]
            g.replay()
            return res.cpu().tolist()
        return self.body(cap, width).cpu().tolist()
