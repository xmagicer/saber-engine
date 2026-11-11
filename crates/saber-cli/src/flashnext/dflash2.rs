//! Offline Flash-Next DFlash2. Only the TP leader talks to the drafter. Both ranks
//! run the target verifier on the same candidates and commit its greedy picks.

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context};
use saber_exec::loader::FunctionTable;
use saber_hip::{DevicePtr, HipThread};
use saber_mem::ple_tier::PleTier;
use saber_model::qwen4_exp::ple_hash::PleHash;
use serde_json::{json, Value};

use super::super::dev::RankDev;
use super::super::exec::{Emitter, Pending, H, VOCAB, W};
use super::super::run::{d2h_in, FnFlags, Prompt};
use super::super::store::PackRecs;
use super::{bf16_at, decode_step, ref_req, verify, Decoded, Graphs, Live, Rank, TOPK};

/// Completed target layers (zero-based), in the drafter's fc input order.
const LAYERS: [u32; 5] = [5, 15, 25, 35, 45];

pub(crate) fn tap_bytes(f: &FnFlags, prompts: &[Prompt]) -> u64 {
    let prompt = prompts
        .iter()
        .map(|p| p.ids.len() as u64)
        .max()
        .unwrap_or(1);
    let chunk = if f.fn_chunk == 0 {
        f.max_chunk()
    } else {
        f.fn_chunk.min(f.max_chunk())
    };
    let rows = prompt
        .min(u64::from(chunk))
        .max(u64::from(f.fn_dflash2_k.unwrap_or(4) + 1));
    rows * W * 2 * LAYERS.len() as u64
}

struct Client {
    stream: BufReader<UnixStream>,
}

impl Client {
    fn connect(path: &Path) -> anyhow::Result<Self> {
        let s = UnixStream::connect(path)
            .with_context(|| format!("connect DFlash2 at {}", path.display()))?;
        s.set_read_timeout(Some(Duration::from_secs(300)))?;
        s.set_write_timeout(Some(Duration::from_secs(300)))?;
        Ok(Self {
            stream: BufReader::new(s),
        })
    }

    fn call(&mut self, request: Value, data: &[u8]) -> anyhow::Result<Value> {
        let out = self.stream.get_mut();
        serde_json::to_writer(&mut *out, &request)?;
        out.write_all(b"\n")?;
        out.write_all(data)?;
        let mut reply = Vec::new();
        Read::by_ref(&mut self.stream)
            .take(8192)
            .read_until(b'\n', &mut reply)?;
        ensure!(
            reply.last() == Some(&b'\n'),
            "DFlash2 closed or returned an oversized response"
        );
        let v: Value = serde_json::from_slice(&reply)?;
        ensure!(
            v["ok"].as_bool() == Some(true),
            "DFlash2 rejected {}: {v}",
            request["op"]
        );
        Ok(v)
    }
}

/// Both ranks reserve the same budget for a mirrored expert layout. The leader
/// materializes all four HC lanes into stable space without touching target scratch.
pub(super) struct Dflash2 {
    client: Option<Client>,
    taps: DevicePtr,
    written: [u32; LAYERS.len()],
    pos: u32,
}

impl Dflash2 {
    pub(super) fn new(hip: &HipThread, rank: u32, path: &Path, bytes: u64) -> anyhow::Result<Self> {
        Ok(Self {
            client: (rank == 0).then(|| Client::connect(path)).transpose()?,
            taps: hip.malloc(bytes)?,
            written: [0; LAYERS.len()],
            pos: 0,
        })
    }

    pub(super) fn reset(&mut self) -> anyhow::Result<()> {
        if let Some(c) = &mut self.client {
            c.call(json!({"op":"reset"}), &[])?;
        }
        self.pos = 0;
        self.written.fill(0);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture(
        &mut self,
        rank: u32,
        dev: &mut RankDev<'_>,
        table: &FunctionTable<'_>,
        em: &mut Emitter<'_>,
        l: u32,
        p: &Pending,
        m: u32,
    ) -> anyhow::Result<()> {
        if rank != 0 {
            return Ok(());
        }
        let Some(i) = LAYERS.iter().position(|&tap| tap == l) else {
            return Ok(());
        };
        let stride = self.taps.len() / LAYERS.len() as u64;
        ensure!(
            u64::from(m) * W * 2 <= stride && self.written[i] == 0,
            "DFlash2 tap {l} invalid row count or unconsumed previous rows"
        );
        ensure!(
            !p.rp && p.pre.is_none(),
            "DFlash2 requires the completed, reduced HC block"
        );
        em.materialize(
            p,
            u64::from(m),
            self.taps.addr().as_u64() + i as u64 * stride,
        )?;
        dev.exec(table, em)?;
        self.written[i] = m;
        Ok(())
    }

    /// The graph has just been recorded. Only a complete set of host-issued tap
    /// writes certifies it. The live marker is set again only after launch.
    pub(super) fn record_graph(&mut self, rows: u32) -> anyhow::Result<u32> {
        if self.client.is_none() {
            return Ok(0);
        }
        tap_transition(&mut self.written, rows, 0)?;
        Ok(rows)
    }

    pub(super) fn replayed_graph(&mut self, recorded: u32, rows: u32) -> anyhow::Result<()> {
        if self.client.is_some() {
            ensure!(
                recorded == rows,
                "DFlash2 replay width differs from its recorded graph"
            );
            tap_transition(&mut self.written, 0, rows)?;
        }
        Ok(())
    }

    pub(super) fn append(
        &mut self,
        dev: &RankDev<'_>,
        position: u32,
        captured: u32,
        committed: u32,
    ) -> anyhow::Result<()> {
        ensure!(
            position == self.pos && committed > 0 && committed <= captured,
            "DFlash2 append position or count"
        );
        if let Some(c) = &mut self.client {
            ensure!(
                self.written.iter().all(|&n| n == captured),
                "DFlash2 target taps incomplete"
            );
            let mut features = vec![0u8; committed as usize * LAYERS.len() * H as usize * 2];
            let stride = self.taps.len() / LAYERS.len() as u64;
            for i in 0..LAYERS.len() {
                let addr = self.taps.addr().as_u64() + i as u64 * stride;
                // Rejected verify rows never cross the draft boundary.
                let raw = d2h_in(
                    dev,
                    &self.taps,
                    addr,
                    (u64::from(committed) * W * 2) as usize,
                )?;
                pack_tap(&mut features, &raw, i, H as usize);
            }
            c.call(
                json!({"op":"append","position":position,"rows":committed}),
                &features,
            )?;
        }
        self.written.fill(0);
        self.pos += committed;
        Ok(())
    }

    fn propose(&mut self, pos: u32, anchor: u32, k: u32) -> anyhow::Result<Vec<u32>> {
        ensure!(pos == self.pos, "DFlash2 proposal before context append");
        let c = self.client.as_mut().context("DFlash2 leader connection")?;
        let v = c.call(
            json!({"op":"propose","position":pos,"anchor":anchor,"width":k}),
            &[],
        )?;
        let ids: Vec<u32> = serde_json::from_value(v["tokens"].clone())?;
        ensure!(
            ids.len() == k as usize && ids.iter().all(|&id| u64::from(id) < VOCAB),
            "invalid DFlash2 proposal"
        );
        Ok(ids)
    }
}

fn tap_transition(written: &mut [u32; LAYERS.len()], from: u32, to: u32) -> anyhow::Result<()> {
    ensure!(
        (from == 0) != (to == 0) && *written == [from; LAYERS.len()],
        "DFlash2 graph tap rows {written:?}, expected all {from} before transition to {to}"
    );
    written.fill(to);
    Ok(())
}

fn pack_tap(out: &mut [u8], raw: &[u8], tap: usize, h: usize) {
    for (r, row) in raw.chunks_exact(h * 4 * 2).enumerate() {
        let start = (r * LAYERS.len() + tap) * h * 2;
        for (i, dst) in out[start..start + h * 2].chunks_exact_mut(2).enumerate() {
            let x = ((bf16_at(row, i) + bf16_at(row, i + h))
                + (bf16_at(row, i + 2 * h) + bf16_at(row, i + 3 * h)))
                * 0.25;
            // Round the FP32 arithmetic mean to BF16, ties to even.
            let bits = x.to_bits();
            let b = if x.is_nan() {
                0x7fc0
            } else {
                (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
            };
            dst.copy_from_slice(&b.to_le_bytes());
        }
    }
}

/// Keep the follower out of the next target collective while the leader's
/// external drafter consumes features. GPU system-scope waits are bounded.
pub(super) fn sync_context(peer: &saber_exec::RankPeer, tag: &str, end: u32) -> anyhow::Result<()> {
    let bytes = end.to_le_bytes();
    ensure!(
        peer.exchange().xchg_bytes(tag, &bytes)? == bytes,
        "DFlash2 rank positions differ after {tag}"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode(
    rk: &mut Rank<'_, '_>,
    em: &mut Emitter<'_>,
    ple: &mut PleTier,
    ph: &PleHash,
    recs: &PackRecs,
    lives: &mut [Live],
    n: u32,
    f: &FnFlags,
    gr: &mut Graphs,
    dp: &mut usize,
    group: &[&Prompt],
) -> anyhow::Result<Decoded> {
    ensure!(
        lives.len() == 1 && group.len() == 1,
        "DFlash2 requires one prompt"
    );
    let mut res = Decoded {
        acc: vec![],
        widths: vec![],
        mtp: vec![vec![]],
        mtp_ms: vec![],
        ids: vec![vec![]],
        tops: vec![vec![]],
        probes: vec![vec![]],
        shas: vec![vec![]],
        steps: vec![],
    };
    let want = (n - 1) as usize;
    while res.ids[0].len() < want {
        let w = (want - res.ids[0].len()).min((f.fn_dflash2_k.unwrap_or(4) + 1) as usize);
        let k = w - 1;
        let pos = lives[0].len;
        let next = lives[0].next;
        let mut draft = Vec::new();
        let t_propose = Instant::now();
        let mut t_broadcast = t_propose;
        if k > 0 {
            if rk.rank == 0 {
                draft = rk
                    .dflash2
                    .as_mut()
                    .context("DFlash2")?
                    .propose(pos, next, k as u32)?;
            }
            t_broadcast = Instant::now();
            let mine: Vec<u8> = draft.iter().flat_map(|x| x.to_le_bytes()).collect();
            let peer = rk.peer.exchange().xchg_bytes("fn.dflash2.draft", &mine)?;
            if rk.rank != 0 {
                ensure!(peer.len() == k * 4, "DFlash2 leader proposal size");
                draft = peer
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
            }
        }
        let t_verify = Instant::now();
        let reqs: Vec<_> = (0..w)
            .map(|t| ref_req(Some(group[0]), 1 + res.ids[0].len() + t))
            .collect();
        let (rows, mut stat) = decode_step(
            rk,
            em,
            ple,
            ph,
            recs,
            lives,
            1,
            *dp,
            gr,
            f,
            &reqs,
            w as u32,
            std::slice::from_ref(&draft),
        )?;
        let t_walk = Instant::now();
        *dp = 1 - *dp;
        let am: Vec<u32> = rows.iter().map(|row| row.top[0].0).collect();
        let a = verify::greedy_walk(&draft, &am) as usize;
        for (t, row) in rows[..=a].iter().enumerate() {
            res.ids[0].push(am[t]);
            res.tops[0].push(row.top[..TOPK].to_vec());
            res.probes[0].push((row.lse, row.probe.clone()));
            res.shas[0].push(row.sha.clone());
        }
        let lv = &mut lives[0];
        for &id in &draft[..a] {
            let _ = ph.ids_step(&mut lv.ctx, &[id]);
        }
        let committed = a as u32 + 1;
        lv.len += committed;
        lv.next = am[a];
        lv.ple_at = *dp;
        lv.naccept = committed;
        lv.drop = w as u32 - committed;
        let t_append = Instant::now();
        stat.committed_at = Some(t_append);
        rk.dflash2
            .as_mut()
            .context("DFlash2")?
            .append(&rk.dev, pos, w as u32, committed)?;
        sync_context(&rk.peer, "fn.dflash2.append", pos + committed)?;
        for (name, start, end) in [
            ("dflash2_propose", t_propose, t_broadcast),
            ("dflash2_rank_exchange", t_broadcast, t_verify),
            ("dflash2_target_verify", t_verify, t_walk),
            ("dflash2_greedy_commit", t_walk, t_append),
            ("dflash2_feature_append", t_append, Instant::now()),
        ] {
            stat.host.insert(name, (end - start).as_secs_f64() * 1000.);
        }
        res.acc.push(vec![a as u32]);
        res.widths.push(w as u32);
        res.mtp[0].push(draft);
        res.steps.push(stat);
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_taps_require_complete_record_and_fresh_replay() {
        let mut taps = [5, 5, 5, 0, 5];
        assert!(tap_transition(&mut taps, 5, 0).is_err());
        assert_eq!(taps, [5, 5, 5, 0, 5]);
        taps[3] = 5;
        tap_transition(&mut taps, 5, 0).unwrap();
        assert!(tap_transition(&mut taps, 0, 0).is_err());
        tap_transition(&mut taps, 0, 5).unwrap();
        assert!(tap_transition(&mut taps, 0, 5).is_err());
        assert!(tap_transition(&mut taps, 4, 0).is_err());
        assert_eq!(taps, [5; 5]);
    }

    #[test]
    fn committed_hc_means_are_row_then_layer_then_channel() {
        let h = 2;
        let mut out = vec![0; 2 * 5 * h * 2];
        for t in 0..5 {
            let raw = (0..2)
                .flat_map(|r| {
                    (0..4).flat_map(move |s| {
                        (0..h).flat_map(move |i| {
                            ((((r * 64 + t * 8 + s * 2 + i) as f32).to_bits() >> 16) as u16)
                                .to_le_bytes()
                        })
                    })
                })
                .collect::<Vec<_>>();
            pack_tap(&mut out, &raw, t, h);
        }
        for r in 0..2 {
            for t in 0..5 {
                for i in 0..h {
                    assert_eq!(
                        bf16_at(&out, (r * 5 + t) * h + i),
                        (r * 64 + t * 8 + 3 + i) as f32
                    );
                }
            }
        }
    }

    #[test]
    fn leader_socket_preserves_frame_boundary_and_rejects_short_reply() {
        use std::io::Read;
        let (a, b) = UnixStream::pair().unwrap();
        let peer = std::thread::spawn(move || {
            let mut b = BufReader::new(b);
            let mut head = String::new();
            b.read_line(&mut head).unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&head).unwrap()["op"],
                "append"
            );
            let mut v = [0u8; 3];
            b.read_exact(&mut v).unwrap();
            assert_eq!(v, [0, 10, 255]);
            b.get_mut().write_all(b"{\"ok\":true}\n").unwrap();
            head.clear();
            b.read_line(&mut head).unwrap();
            b.get_mut().write_all(b"{\"ok\":true}").unwrap();
        });
        let mut c = Client {
            stream: BufReader::new(a),
        };
        c.call(json!({"op":"append"}), &[0, 10, 255]).unwrap();
        assert!(c.call(json!({"op":"reset"}), &[]).is_err());
        peer.join().unwrap();
    }
}
