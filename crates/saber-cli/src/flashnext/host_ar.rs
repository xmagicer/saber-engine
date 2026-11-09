//! Exact two-rank BF16 mailbox over registered host pages, with a process-local GPU epoch.
use super::dev::ArState;
use super::exec::H;
use anyhow::{ensure, Context};
use saber_exec::RankPeer;
use saber_hip::{DevicePtr, HipThread, SharedHost, Stream};
use std::path::Path;

pub(super) const SYMBOL: &str = "fn_ar_host_exact_bf16";
// All calls advance all these epochs, even for tiny payloads. Reusing a per-block
// counter with a variable grid can otherwise reuse a parity slot still being read.
pub(super) const BLOCKS: u64 = 32;

pub(super) struct HostAr {
    mine: SharedHost,
    peer: SharedHost,
    seq: DevicePtr,
    slot: u64,
}

impl HostAr {
    pub(super) fn open(
        hip: &HipThread,
        stream: &Stream,
        peer: &RankPeer,
        slot: u64,
    ) -> anyhow::Result<Self> {
        ensure!(
            slot > 0 && slot % 16 == 0 && slot / 16 <= i32::MAX as u64,
            "invalid host mailbox slot"
        );
        peer.agree("fn.host-ar.slot", slot)?;
        let bytes = 2 * slot + 4 * BLOCKS;
        let mut mine = hip.shared_host_create(bytes)?;
        let seq = hip.malloc(4 * BLOCKS)?;
        hip.memset_shared_zero(&mine, stream)?;
        hip.memset(&seq, 0, stream)?;
        hip.sync_stream(stream)?;
        let path = mine
            .path()
            .and_then(Path::to_str)
            .context("host mailbox needs a UTF-8 path")?;
        let remote_path = peer
            .exchange()
            .xchg_bytes("fn.host-ar.path", path.as_bytes())?;
        let other = hip.shared_host_open(Path::new(std::str::from_utf8(&remote_path)?), bytes)?;
        peer.rendezvous("fn.host-ar.mapped")?;
        mine.unlink()?;
        Ok(Self {
            mine,
            peer: other,
            seq,
            slot,
        })
    }

    pub(super) fn states(&self) -> (ArState, ArState) {
        let state = |m: &SharedHost, seq| {
            let scratch = m.addr().as_u64();
            ArState {
                scratch,
                flags: scratch + 2 * self.slot,
                seq,
                loc: 0,
            }
        };
        (
            state(&self.mine, self.seq.addr().as_u64()),
            state(&self.peer, 0),
        )
    }
}

pub(super) fn slot_bytes(max_chunk: u32) -> u64 {
    // Do not use the normal prefill mailbox law: it prices compressed WHT6 payloads.
    u64::from(max_chunk.max(super::run::MAX_ROWS)) * H * 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flashnext::kern::{Arg, Decls};
    use saber_exec::loader::PluginSet;
    use saber_exec::{LoadCounters, TcpExchange};
    use saber_hip::{StreamCaptureMode, StreamFlags};
    use std::{net::TcpListener, path::PathBuf, process::Command, sync::Arc, time::Duration};

    #[test]
    fn host_mailbox_holds_the_uncompressed_prefill_and_reserves_both_parities() {
        for rows in [1, 256, 2048, 8192] {
            let slot = slot_bytes(rows);
            assert!(slot >= u64::from(rows) * H * 2);
            assert_eq!(slot % 16, 0);
            assert!(slot / 16 <= i32::MAX as u64);
        }
        assert_eq!(slot_bytes(8192), 40 << 20);
    }

    fn run_rank(rank: u32, peer: RankPeer, prebuilt: &Path) -> anyhow::Result<()> {
        let hip = crate::open_hip("/opt/rocm/lib/libamdhip64.so.7")?;
        hip.set_device(rank as i32)?;
        let stream = hip.create_stream(StreamFlags::NonBlocking)?;
        let decls = Decls::load_plugins(prebuilt, ["flashnext_ar_host"])?;
        let plugin =
            PluginSet::load_named(&hip, prebuilt, &["flashnext_ar_host"], &LoadCounters::new())?;
        let table = plugin.functions(&hip)?;
        let (kernel, abi) = table
            .get("flashnext_ar_host", SYMBOL)
            .context("host exact kernel")?;
        let count = |i: usize, r: u32, step: usize| {
            ((i * 17 + r as usize * 19 + step * 13) % 97) as i32 - 48
        };
        let bf16 = |x: i32| (((x as f32).to_bits() >> 16) as u16).to_le_bytes();
        // Several distinct shapes per replay, with an odd number of calls so replay
        // cannot accidentally rely on starting the graph at one particular parity.
        let mut counts = vec![8usize, 2560 * 7, 2560 * 257, 2560, 2560 * 2];
        if std::env::var("SABER_HOST_AR_PROBE_WIDE").as_deref() == Ok("1") {
            counts.extend([8192 * H as usize, 16]);
        }
        let expected_epoch = 7 * counts.len() as u32;
        let slot = (*counts.iter().max().unwrap() * 2) as u64;
        let maps = HostAr::open(&hip, &stream, &peer, slot)?;
        let (my, other) = maps.states();
        let mut work = Vec::new();
        for (step, &n) in counts.iter().enumerate() {
            let x = hip.malloc(n as u64 * 2)?;
            let y = hip.malloc(n as u64 * 2)?;
            let payload: Vec<_> = (0..n).flat_map(|i| bf16(count(i, rank, step))).collect();
            hip.memcpy_h2d_blocking(&x, &payload, &stream)?;
            let mut args = decls.pack(
                SYMBOL,
                &[
                    Arg::P(x.addr().as_u64()),
                    Arg::P(other.scratch),
                    Arg::P(my.scratch),
                    Arg::P(y.addr().as_u64()),
                    Arg::P(other.flags),
                    Arg::P(my.flags),
                    Arg::P(my.seq),
                    Arg::I((n / 8) as i32),
                    Arg::I((slot / 16) as i32),
                    Arg::I(1),
                    Arg::I(1),
                ],
            )?;
            args.resize(abi.kernarg_segment_size as usize, 0);
            work.push((x, y, args));
        }
        hip.begin_capture(&stream, StreamCaptureMode::ThreadLocal)?;
        for (_, _, args) in &mut work {
            hip.launch(kernel, [BLOCKS as u32, 1, 1], [256, 1, 1], 0, &stream, args)?;
        }
        let graph = hip.end_capture(&stream)?;
        let exec = hip.instantiate(&graph)?;
        peer.rendezvous("fn.host-ar.probe-ready")?;
        for _ in 0..7 {
            hip.graph_launch(&exec, &stream)?;
        }
        hip.sync_stream(&stream)?;
        for (step, ((_, y, _), &n)) in work.iter().zip(&counts).enumerate() {
            let mut got = vec![0u8; n * 2];
            hip.memcpy_d2h_blocking(&mut got, y, &stream)?;
            for (i, word) in got.chunks_exact(2).enumerate() {
                ensure!(
                    word == bf16(count(i, 0, step) + count(i, 1, step)),
                    "rank {rank} shape {n} word {i}"
                );
            }
        }
        let mut seq = vec![0u8; BLOCKS as usize * 4];
        hip.memcpy_d2h_blocking(&mut seq, &maps.seq, &stream)?;
        ensure!(
            seq.chunks_exact(4)
                .all(|x| x == expected_epoch.to_le_bytes()),
            "unexpected host reduce epoch"
        );
        peer.rendezvous("fn.host-ar.probe-done")?;
        println!(
            "Flash-Next host AR rank {rank}: {expected_epoch} graph reductions, {} shapes, largest payload {} bytes, BF16 sums and all epochs exact",
            counts.len(), slot
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires two gfx1201 cards and the flashnext_ar_host code object"]
    fn host_ar_two_process_graph() -> anyhow::Result<()> {
        let default = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let prebuilt = std::env::var_os("SABER_HOST_AR_PROBE_PREBUILT")
            .map(PathBuf::from)
            .unwrap_or(default);
        let timeout = Duration::from_secs(30);
        if let Ok(port) = std::env::var("SABER_HOST_AR_PROBE_PORT") {
            return run_rank(
                1,
                RankPeer::new(Arc::new(TcpExchange::connect(port.parse()?, timeout)?)),
                &prebuilt,
            );
        }
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let me = "flashnext::host_ar::tests::host_ar_two_process_graph";
        let mut worker = Command::new(std::env::current_exe()?)
            .args(["--ignored", "--exact", me, "--nocapture"])
            .env(
                "SABER_HOST_AR_PROBE_PORT",
                listener.local_addr()?.port().to_string(),
            )
            .spawn()?;
        let result = TcpExchange::accept(&listener, timeout)
            .map_err(anyhow::Error::from)
            .and_then(|stream| run_rank(0, RankPeer::new(Arc::new(stream)), &prebuilt));
        if result.is_err() {
            let _ = worker.kill();
        }
        let status = worker.wait()?;
        result?;
        ensure!(status.success(), "host reduce worker failed: {status}");
        Ok(())
    }
}
