"""Check coarse score ties and the public INT2 head in both block and dense modes.

Run on a gfx1201 device with `python tools/dspine/gpu_head_gate.py`. An optional
`--draft-pack` uses an existing two-shard INT2 drafter instead of the small fixture.
For the selector alone, set TRITON_INTERPRET=1 and pass `--selector-only --device cpu`.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import canonical_coarse
import native_head


def check_selector(device):
    rng = np.random.default_rng(807)
    n = 15520
    scores = rng.integers(-8, 9, size=(15,n)).astype(np.float32)
    ids = np.stack([rng.permutation(n).astype(np.int32) for _ in scores])
    scores[0] = 0
    scores[1,:5] = [np.inf,-np.inf,np.nan,-0.,0.]
    scores[2] = np.arange(n,dtype=np.float32)
    scores[2,-80] = np.nextafter(scores[2,-81],np.float32(np.inf))
    values = torch.from_numpy(scores).to(device)
    tokens = torch.from_numpy(ids).to(device)
    # NumPy sorts the independent fields directly. Signed zeros tie, NaNs rank first.
    truth = np.lexsort((ids,-np.where(np.isnan(scores),0,scores),~np.isnan(scores)),axis=1)
    for k in (0,16,80,256):
        for _ in range(3):
            positions = canonical_coarse.positions(values,tokens,k)
            assert np.array_equal(positions.cpu().numpy(),truth[:,:k]), k
    assert canonical_coarse.positions(values[:0],tokens[:0],80).shape == (0,80)


def make_fixture(root):
    h,n = 5120,1280
    ranks = []
    for rank in (0,1):
        path = root/f"rank{rank}.pack"
        records = []
        with path.open("wb") as f:
            for name,layout,columns,element in (("wq",9,h//4,b"\0"),("s",10,h//128,b"\x80\x3f"),
                                                  ("zs",11,h//128,b"\0\0")):
                length = n*columns*len(element)
                records.append({"name":"head.int2."+name,"shape":[n,columns],"layout_id":layout,
                                "length":length,"data_off":f.tell()})
                f.write(element*(n*columns))
        ranks.append({"rank":rank,"file":path.name,"file_bytes":path.stat().st_size,"records":records})
    (root/"manifest.json").write_text(json.dumps({"ranks":ranks}))


def check_int2(draft_pack,prebuilt,device):
    reader = native_head.NativeSparseHead(draft_pack,prebuilt,device.index)
    # A zero activation produces exactly tied scores both before and after the rerank,
    # independent of the weight pack; a broadcast scalar avoids a full dense allocation.
    hidden = torch.zeros((15,reader.hidden),dtype=torch.bfloat16,device=device)
    final_head = torch.zeros((),dtype=torch.bfloat16,device=device).expand(reader.vocab,reader.hidden)
    for mode in ("block","dense"):
        pool = []
        for shard in reader.shards:
            assert shard["n"]>=640
            # The INT2 kernel emits eight distinct low-ID tie winners per 64-column tile.
            local = range(80) if mode=="dense" else [64*b+i for b in range(10) for i in range(8)]
            pool.extend(shard["base"]+i for i in local)
        for k in (16,len(pool)):
            expected = torch.tensor(sorted(pool)[:k],dtype=torch.int64,device=device).expand(15,-1)
            for _ in range(3):
                result = reader.candidates(hidden,final_head,k=k,pool_per_shard=80,coarse_mode=mode)
                assert torch.equal(result,expected),(mode,k)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prebuilt",type=Path,default=Path(__file__).resolve().parents[2]/"prebuilt/gfx1201")
    p.add_argument("--draft-pack",type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--selector-only",action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type=="cuda":
        torch.cuda.set_device(device)
        device=torch.device("cuda",torch.cuda.current_device())
    elif not args.selector_only:
        p.error("Native INT2 validation requires --device cuda:N")
    check_selector(device)
    if not args.selector_only:
        with tempfile.TemporaryDirectory() as tmp:
            pack = args.draft_pack
            if pack is None:
                pack=Path(tmp)
                make_fixture(pack)
            check_int2(pack,args.prebuilt,device)
    print(json.dumps({"passed":True,"selector_cases":True,"native_block_and_dense":not args.selector_only}))


if __name__=="__main__":
    main()
