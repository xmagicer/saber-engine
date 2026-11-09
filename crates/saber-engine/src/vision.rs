//! The vision tower's RESIDENCY (workstream D2): the 333 `model.visual.*` records uploaded at
//! boot through the same `saber_mem::upload::Upload` the target's weights take, and the tower's
//! executor (`saber_exec::vision::VisionExecutor`) built over them.
//!
//! **The residency decision.** The pack keeps the tower REPLICATED (`build_pack.py`: the same
//! 333 bf16 records, 921,460,192 B, at the END of every rank's file, records 1312..=1644). Prod
//! shards it (`mm_encoder_tp_mode = weights`: qkv/fc1 column-parallel, proj/fc2 row-parallel,
//! one all-reduce each, 54 per image). Saber runs it **replicated on every rank that serves an
//! image** -- each rank uploads its own copy and computes the same rows from the same pixels,
//! because:
//!
//! * the text prefill needs the image rows on BOTH ranks (the residual stream is replicated at
//!   TP = 2), so a rank-0-only tower pays a peer copy of `[tokens, 5120]` bf16 per image
//!   (168 MB at 16k tokens) and still leaves rank 1 idle while rank 0 computes;
//! * the KV pool is sized per rank from what the byte ledger has left, and the scheduler hands
//!   out one block id space for both ranks, so a tower on rank 0 alone costs rank 1's KV the same
//!   0.92 GB anyway (the pool is only as large as the smaller rank's);
//! * sharding halves the weight bytes (0.46 GB/rank) at the price of 54 all-reduces per image and
//!   a different accumulation order in proj and fc2 (split-K over two ranks) -- a change of class
//!   against the gate D1 passed, for 0.46 GB. Not taken this round; it is the row to revisit if
//!   KV at long context becomes the binding constraint.
//!
//! The KV cost is therefore `0.92 GB` of weights + the tower arena (1.011 GiB at prod's 65,536-
//! patch ceiling, 0.253 GiB at 16,384) on EACH rank, billed to the ledger before the KV pool is
//! derived. A text-only engine (`vision_max_patches = 0`, the default) uploads none of it and the
//! vision request is refused by name.

use std::collections::BTreeMap;
use std::path::Path;

use saber_exec::vision::{VisionExecutor, VISION_PREFIX};
use saber_hip::{DeviceAddr, DevicePtr, HipThread};
use saber_kernels::vision::{tower_arena, VisionRules};
use saber_mem::dio::OpenMode;
use saber_mem::ledger::{Ledger, Pool};
use saber_mem::pack::PackReader;
use saber_mem::upload::{HipSink, PinnedRing, Upload, UploadConfig};

use crate::forward::ForwardError;

/// Pinned staging for the per-image host inputs (tables + pixel rows). 32 MiB = a 2,048-patch
/// image's pixel rows in one batch; larger images go up in several.
pub const VISION_STAGING_BYTES: usize = 32 << 20;

/// The pack window the tower's records occupy, and its byte span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionWindow {
    /// First record index.
    pub first: usize,
    /// Records in the window (all of them `model.visual.*`).
    pub records: usize,
    /// Bytes of the window, aligned as the upload arena lays it out.
    pub span: u64,
}

/// Find the tower's records in a rank's pack table: one CONTIGUOUS run of `model.visual.*`
/// names, with nothing else inside it (the upload reads a byte window, so an interloper would be
/// uploaded and never bound).
///
/// # Errors
/// No tower in the pack (a text-only pack), a non-contiguous run, or a count off the rules'.
pub fn vision_window(
    reader: &PackReader,
    rules: &VisionRules,
) -> Result<VisionWindow, ForwardError> {
    let idx: Vec<usize> = (0..reader.records().len())
        .filter(|&i| reader.name(i).starts_with(VISION_PREFIX))
        .collect();
    let (Some(&lo), Some(&hi)) = (idx.first(), idx.last()) else {
        return Err(ForwardError::Device(
            "vision: the pack carries no `model.visual.*` record -- a text-only pack cannot serve \
             an image"
                .to_owned(),
        ));
    };
    if hi - lo + 1 != idx.len() {
        let inter = (lo..=hi)
            .find(|&i| !reader.name(i).starts_with(VISION_PREFIX))
            .map(|i| reader.name(i).to_owned())
            .unwrap_or_default();
        return Err(ForwardError::Device(format!(
            "vision: the tower's records are not one contiguous run of the pack (`{inter}` sits \
             inside it); the upload reads a byte window"
        )));
    }
    let want = rules.records().len();
    if idx.len() != want {
        return Err(ForwardError::Device(format!(
            "vision: the pack carries {} `model.visual.*` records, the rules bind {want}",
            idx.len()
        )));
    }
    let recs = reader.records();
    let last = &recs[hi];
    let span = saber_mem::pack::align_up(last.data_off + last.length) - recs[lo].data_off;
    Ok(VisionWindow {
        first: lo,
        records: idx.len(),
        span,
    })
}

/// The bytes a vision tier bills per rank: the weight window and the arena.
///
/// # Errors
/// [`vision_window`]'s, or an arena the layout refuses.
pub fn vision_demand(
    reader: &PackReader,
    rules: &VisionRules,
    max_patches: u32,
) -> Result<(VisionWindow, u64), ForwardError> {
    let w = vision_window(reader, rules)?;
    let a = tower_arena(rules, max_patches)
        .map_err(|e| ForwardError::Device(format!("vision arena: {e}")))?;
    Ok((w, a.bytes))
}

/// A resident tower: its executor, and the upload sink that owns the weights' device memory.
pub struct VisionTier<'h> {
    /// The executor.
    pub exec: VisionExecutor,
    /// Records bound.
    pub records: usize,
    /// Weight bytes resident (the upload arena).
    pub weight_bytes: u64,
    /// Resident vision weight tensor bytes, excluding the upload arena's padding.
    pub weight_payload_bytes: u64,
    /// Seconds the upload took.
    pub upload_s: f64,
    /// The resident form's device memory; `None` for the streamed form (P27-11), which owns none.
    _sink: Option<HipSink<'h>>,
}

impl std::fmt::Debug for VisionTier<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VisionTier({} records, {} B weights, {:?})",
            self.records, self.weight_bytes, self.exec
        )
    }
}

/// Upload the tower's window of `pack`'s `rank` file and build the executor for images of up to
/// `max_patches` patches.
///
/// # Errors
/// The window, the upload, a record the plan reads and the upload did not place, or the arena.
pub fn boot_vision<'h>(
    hip: &'h HipThread,
    pack_file: &Path,
    rank: u32,
    world: u32,
    max_patches: u32,
) -> Result<VisionTier<'h>, ForwardError> {
    let rules =
        VisionRules::prod().map_err(|e| ForwardError::Device(format!("vision rules: {e}")))?;
    boot_vision_with(hip, pack_file, rank, world, max_patches, rules)
}

/// FN55: [`boot_vision`] under `rules` -- Flash-Next's tower (`VisionRules::flashnext()`, the
/// merger at 2,560) from the Flash-Next pack's `vision/rank<r>.pack` member.
///
/// # Errors
/// As [`boot_vision`].
pub fn boot_vision_with<'h>(
    hip: &'h HipThread,
    pack_file: &Path,
    rank: u32,
    world: u32,
    max_patches: u32,
    rules: VisionRules,
) -> Result<VisionTier<'h>, ForwardError> {
    boot_vision_in(hip, pack_file, rank, world, max_patches, rules, None)
}

/// FN55c: the bytes the tower's weight upload places (its upload plan's arena) and its arena at
/// `max_patches` -- what a caller that holds the memory itself ([`boot_vision_in`]) allocates.
/// Host only (the pack's table, no HIP).
///
/// # Errors
/// The window, the upload's plan, the arena layout.
pub fn vision_region_bytes(
    pack_file: &Path,
    rank: u32,
    world: u32,
    max_patches: u32,
    rules: &VisionRules,
) -> Result<(u64, u64), ForwardError> {
    let reader = PackReader::open(pack_file)
        .map_err(|e| ForwardError::Device(format!("vision: opening the pack: {e}")))?;
    let w = vision_window(&reader, rules)?;
    let dir = pack_file.parent().unwrap_or(pack_file);
    let up = Upload::open(
        dir,
        rank,
        UploadConfig {
            start_record: w.first,
            limit_bytes: Some(w.span),
            mode: OpenMode::Direct,
            expect_tp: Some(world),
            ..UploadConfig::default()
        },
    )
    .map_err(|e| ForwardError::Device(format!("vision: opening the upload: {e}")))?;
    let a = tower_arena(rules, max_patches)
        .map_err(|e| ForwardError::Device(format!("vision arena: {e}")))?;
    Ok((up.plan().arena_bytes, a.bytes.max(1)))
}

/// FN55c: [`boot_vision_with`], the weights uploaded into `regions.0` and the tower's arena laid
/// in `regions.1` when given (memory the caller holds -- `--fn-vision lazy`'s region, lent to the
/// expert tier while no tower is resident; sized by [`vision_region_bytes`]), else allocated.
///
/// # Errors
/// As [`boot_vision`], or a region smaller than its plan.
pub fn boot_vision_in<'h>(
    hip: &'h HipThread,
    pack_file: &Path,
    rank: u32,
    world: u32,
    max_patches: u32,
    rules: VisionRules,
    regions: Option<(std::rc::Rc<DevicePtr>, std::rc::Rc<DevicePtr>)>,
) -> Result<VisionTier<'h>, ForwardError> {
    let t0 = std::time::Instant::now();
    let reader = PackReader::open(pack_file)
        .map_err(|e| ForwardError::Device(format!("vision: opening the pack: {e}")))?;
    let w = vision_window(&reader, &rules)?;
    let dir = pack_file.parent().unwrap_or(pack_file);
    let up = Upload::open(
        dir,
        rank,
        UploadConfig {
            start_record: w.first,
            limit_bytes: Some(w.span),
            mode: OpenMode::Direct,
            expect_tp: Some(world),
            ..UploadConfig::default()
        },
    )
    .map_err(|e| ForwardError::Device(format!("vision: opening the upload: {e}")))?;
    let uplan = up.plan();
    // A LEDGER OF ITS OWN, as the drafter's: `Upload::reserve` reserves under fixed names the
    // engine's ledger already holds for the target; the engine bills the tower as
    // `engine.vision.weights` / `engine.vision.arena` itself.
    let mut own = Ledger::new();
    let e = |what: &'static str| {
        move |x: saber_mem::ledger::LedgerError| {
            ForwardError::Device(format!("vision: {what}: {x}"))
        }
    };
    own.declare(Pool::Vram(rank), uplan.arena_bytes.max(1))
        .map_err(e("declaring the arena"))?;
    own.declare(Pool::HostPinned, uplan.staging_bytes.max(1))
        .map_err(e("declaring the staging"))?;
    let reserved = up
        .reserve(&mut own)
        .map_err(|x| ForwardError::Device(format!("vision: reserving the upload: {x}")))?;
    let mut stg = hip
        .host_malloc(reserved.staging_bytes())
        .map_err(|x| ForwardError::Device(format!("vision: pinned staging: {x}")))?;
    let ring = PinnedRing::new(&mut stg, &reserved)
        .map_err(|x| ForwardError::Device(format!("vision: the ring: {x}")))?;
    let (wreg, areg) = regions.map_or((None, None), |(w, a)| (Some(w), Some(a)));
    let mut sink = match wreg {
        Some(w) => HipSink::over(hip, &reserved, w),
        None => HipSink::new(hip, &reserved),
    }
    .map_err(|x| ForwardError::Device(format!("vision: the sink: {x}")))?;
    let mut addr: BTreeMap<String, DeviceAddr> = BTreeMap::new();
    let mut weight_payload_bytes = 0;
    up.run(&mut sink, &ring, |r| {
        weight_payload_bytes += r.len;
        addr.insert(r.name.to_string(), r.addr);
    })
    .map_err(|x| ForwardError::Device(format!("vision: uploading the tower: {x}")))?;
    drop(ring);
    drop(stg);
    let records = addr.len();
    let exec = VisionExecutor::new_in(hip, rules, max_patches, addr, VISION_STAGING_BYTES, areg)
        .map_err(|x| ForwardError::Device(format!("vision: {x}")))?;
    Ok(VisionTier {
        exec,
        records,
        weight_bytes: uplan.arena_bytes,
        weight_payload_bytes,
        upload_s: t0.elapsed().as_secs_f64(),
        _sink: Some(sink),
    })
}

/// P27-11 (`--vision lazy`): the tower's pinned host bytes for a STREAMED boot -- the weight
/// window (the slab) and the out rows' copy at `max_patches` -- billed to `Pool::HostPinned`.
///
/// # Errors
/// [`vision_demand`]'s.
pub fn streamed_host_bytes(
    reader: &PackReader,
    max_patches: u32,
) -> Result<(u64, u64), ForwardError> {
    let rules =
        VisionRules::prod().map_err(|e| ForwardError::Device(format!("vision rules: {e}")))?;
    let w = vision_window(reader, &rules)?;
    let a = tower_arena(&rules, max_patches)
        .map_err(|e| ForwardError::Device(format!("vision arena: {e}")))?;
    let out = a.slot("out").map_or(0, |s| s.bytes);
    Ok((w.span, out + VISION_STAGING_BYTES as u64))
}

/// P27-11 (`--vision lazy`): the STREAMED tower. No VRAM of its own: the 333 records are read
/// from `pack_file` into a pinned host slab once at boot, and every image streams them through a
/// two-slot ring laid over `lent` (the engine's shared prefill arena, idle while the tower runs)
/// with the tower arena behind it, sized for the largest image the lent arena holds at or below
/// `max_patches` (a multiple of 1,024, printed). The KV pool is the text-only boot's.
///
/// # Errors
/// The window, the reads, a lent arena that holds no 1,024-patch image, or the executor.
pub fn boot_vision_streamed<'h>(
    hip: &'h HipThread,
    pack_file: &Path,
    max_patches: u32,
    lent: std::rc::Rc<saber_hip::DevicePtr>,
) -> Result<VisionTier<'h>, ForwardError> {
    use std::os::unix::fs::FileExt;
    let t0 = std::time::Instant::now();
    let rules =
        VisionRules::prod().map_err(|e| ForwardError::Device(format!("vision rules: {e}")))?;
    let reader = PackReader::open(pack_file)
        .map_err(|e| ForwardError::Device(format!("vision: opening the pack: {e}")))?;
    let w = vision_window(&reader, &rules)?;
    let recs = reader.records();
    let first = recs[w.first].data_off;
    let mut slab = hip
        .host_malloc(w.span)
        .map_err(|x| ForwardError::Device(format!("vision: the pinned slab: {x}")))?;
    let f = std::fs::File::open(pack_file)
        .map_err(|e| ForwardError::Device(format!("vision: opening the pack: {e}")))?;
    let mut records = Vec::with_capacity(w.records);
    for (i, r) in recs.iter().enumerate().skip(w.first).take(w.records) {
        let off = usize::try_from(r.data_off - first).unwrap_or(usize::MAX);
        let len = usize::try_from(r.length).unwrap_or(usize::MAX);
        let dst = slab
            .as_mut_slice()
            .get_mut(off..off + len)
            .ok_or_else(|| ForwardError::Device("vision: a record past the slab".to_owned()))?;
        f.read_exact_at(dst, r.data_off).map_err(|e| {
            ForwardError::Device(format!("vision: reading `{}`: {e}", reader.name(i)))
        })?;
        records.push(saber_exec::vision::SlabRecord {
            name: reader.name(i).to_owned(),
            offset: off,
            len,
        });
    }
    let mut mp = max_patches - max_patches % 1024;
    while mp >= 1024 {
        match VisionExecutor::streamed_bytes(&rules, mp, &records) {
            Ok(b) if b <= lent.len() => break,
            Ok(_) => mp -= 1024,
            Err(e) => return Err(ForwardError::Device(format!("vision: {e}"))),
        }
    }
    if mp < 1024 {
        return Err(ForwardError::Device(format!(
            "vision lazy: the lent prefill arena ({} B) holds no 1,024-patch image beside the              weight ring",
            lent.len()
        )));
    }
    let exec =
        VisionExecutor::new_streamed(hip, rules, mp, lent, slab, &records, VISION_STAGING_BYTES)
            .map_err(|x| ForwardError::Device(format!("vision: {x}")))?;
    Ok(VisionTier {
        exec,
        records: records.len(),
        weight_bytes: 0,
        weight_payload_bytes: 0,
        upload_s: t0.elapsed().as_secs_f64(),
        _sink: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_text_only_engine_refuses_an_image_by_name() {
        let e = crate::device::check_vision_enabled(0)
            .unwrap_err()
            .to_string();
        assert!(e.contains("TEXT-ONLY"), "{e}");
        crate::device::check_vision_enabled(65_536).expect("an image-serving engine");
    }

    /// The residency's bytes, read off the real pack's table when this host has it (the pack is
    /// read, never uploaded: no HIP).
    #[test]
    fn the_tower_window_of_the_default_pack() {
        let f = Path::new("/mnt/nvme/saber-packs/qwen38-27b-nvfp4-devfinal-fp8head/rank0.pack");
        if !f.exists() {
            eprintln!("skipped: {} is not on this host", f.display());
            return;
        }
        let rules = VisionRules::prod().unwrap();
        let reader = PackReader::open(f).unwrap();
        let (w, arena) = vision_demand(&reader, &rules, 65_536).unwrap();
        assert_eq!(w.records, 333);
        assert_eq!(w.first, 1312);
        assert!(
            w.span >= 921_460_192 && w.span < 921_460_192 + (333 << 12),
            "{}",
            w.span
        );
        // D5: + 65,536 B -- `cu_seqlens` holds every frame 65,536 patches can carry (16,385 i32,
        // aligned), where D2 sized it for one (8 B) and the card refused a two-image request.
        assert_eq!(arena, 1_085_341_952);
    }
}
