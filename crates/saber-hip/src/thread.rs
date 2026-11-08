//! The one OS thread every HIP call runs on.
//!
//! This is a structural rule, not a convention (ARCHITECTURE.md, "Rust-specific hazards"):
//!
//! > Confine every HIP call to one OS thread, structurally: a `HipThread` owning a channel and a
//! > `!Send` context token, HIP FFI private to that module. Tokio for HTTP, a plain `std::thread`
//! > for the step loop, a bounded channel between; then `hipStreamCaptureModeThreadLocal` only
//! > serializes against that thread. With the example-default `Global` mode, a `/metrics` handler
//! > calling `hipMemGetInfo` on a tokio worker fails intermittently, under load, only while a
//! > bucket is capturing.
//!
//! The mechanism: [`HipThread::spawn`] starts one `std::thread`, and that thread -- and only that
//! thread -- ever holds the [`crate::ffi::Lib`]. Work reaches it as boxed closures over a channel.
//! Each closure receives a `&`[`Ctx`], a zero-sized `!Send` token that exists only on that thread,
//! so "am I allowed to call HIP here?" is answered by the type system rather than by a comment.
//!
//! **Nothing async touches HIP.** The step loop is a plain thread; tokio talks to it through a
//! channel. A work-stealing scheduler adds variance to exactly the path that wants fixed cadence.

use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::error::{Error, Result};
use crate::ffi::{
    DeviceSchedule, EventFlags, IpcMemHandle, Lib, MallocFlags, MemcpyKind, StreamCaptureMode,
    StreamFlags,
};
use crate::handles::RcclComm;
use crate::handles::{
    DeviceAddr, DevicePtr, Event, Function, Graph, GraphExec, ImportedPtr, Module, PinnedBuf,
    PinnedSlice, PinnedSliceMut, RawEvent, RawFunction, RawGraph, RawGraphExec, RawModule,
    RawStream, Stream,
};
use crate::lane::{Background, BatchTicket, Deps, Name};
use crate::rccl::{RcclLib, RcclUniqueId};

struct RcclState {
    path: String,
    lib: Arc<RcclLib>,
    owner: Arc<()>,
    comms: Vec<usize>,
}

/// Proof that the holder is running on the HIP thread.
///
/// Zero-sized and `!Send`: it is constructed once, on the HIP thread, and handed out only as a
/// shared reference inside a job closure. Every method on the raw HIP surface takes one, so calling
/// HIP from anywhere else does not typecheck.
pub struct Ctx {
    _not_send: PhantomData<*mut ()>,
}

impl Ctx {
    fn new() -> Ctx {
        Ctx {
            _not_send: PhantomData,
        }
    }

    /// FN59: the lane's host tests drive [`crate::lane::serve`] with a fake runtime.
    #[cfg(test)]
    pub(crate) fn new_for_tests() -> Ctx {
        Ctx::new()
    }
}

/// A closure the HIP thread will run, erased to `'static` to cross the channel.
///
/// Named `Task` rather than `Job` because [`Job`] is the public, *data* description of one HIP call
/// that [`HipThread::run_batch`] takes: a closure crossing the channel is an implementation detail,
/// a batch of jobs is an interface.
pub(crate) type Task<L = Lib> = Box<dyn FnOnce(&L, &Ctx) + Send + 'static>;

/// The same closure before erasure, while it may still borrow from the caller's frame.
type BorrowedTask<'a> = Box<dyn FnOnce(&Lib, &Ctx) + Send + 'a>;

/// A unit of work for the HIP thread.
///
/// Generic over the runtime only so the loop ([`crate::lane::serve`]) is testable with a fake on a
/// host with no ROCm; everything real is `Msg<Lib>`.
pub(crate) enum Msg<L = Lib> {
    /// Run this closure with the loaded runtime and the thread token. [`Deps`] says what of the
    /// background lane it may overtake (FN59; `Deps::All` = nothing, the pre-FN59 order).
    Call(Task<L>, Deps),
    /// FN59: a non-blocking batch for the background lane ([`HipThread::submit_batch`]).
    Background(Background),
    /// Stop the loop. Sent by [`HipThread`]'s `Drop`.
    Shutdown,
}

/// Versions and identity of the loaded runtime, captured once at spawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInfo {
    /// The soname that was loaded, e.g. `libamdhip64.so.7`.
    pub soname: String,
    /// `hipDriverGetVersion`, in HIP's packed form.
    pub driver_version: i32,
    /// `hipRuntimeGetVersion`, in HIP's packed form.
    pub runtime_version: i32,
}

/// One HIP call, as data, for [`HipThread::run_batch`].
///
/// Every call the engine posts to the HIP thread costs one channel round trip, measured at
/// **3.24 µs** on this box (workstream H, 2026-09-18, 2000 `hipGetDevice` round trips). A staging
/// ring that posts `copy, copy, record-event` per wave pays three of those for ~25 µs of DMA, which
/// is how a bulk transfer ends up host-bound. A batch is one round trip for the whole list.
///
/// The variants borrow their handles, so the batch cannot outlive the stream, event or allocation
/// it names; the values that actually cross the channel are the `Send` integer tokens, exactly as
/// for the single-call methods.
#[derive(Debug)]
#[non_exhaustive]
pub enum Job<'a> {
    /// `hipMemcpyAsync` H2D from a pinned staging view into `dst` at `dst_offset`.
    MemcpyH2d {
        /// Destination allocation.
        dst: &'a DevicePtr,
        /// Byte offset into `dst`. Bounds-checked against its length when the batch is lowered.
        dst_offset: u64,
        /// Pinned source; its length is the copy length.
        src: PinnedSlice<'a>,
        /// Stream the copy is issued on.
        stream: &'a Stream,
    },
    /// `hipMemcpyAsync` D2H from `src` at `src_offset` into a pinned staging view.
    ///
    /// The mirror of [`Job::MemcpyH2d`], and the one a session park needs: a park reads VRAM into
    /// page-locked host memory and must **not** wait for it, so the copy is issued here and the
    /// caller waits on an event of its own (or, as the scheduler does, never waits at all and polls).
    MemcpyD2h {
        /// Source allocation.
        src: &'a DevicePtr,
        /// Byte offset into `src`. Bounds-checked against its length when the batch is lowered.
        src_offset: u64,
        /// Pinned destination; its length is the copy length.
        dst: PinnedSliceMut<'a>,
        /// Stream the copy is issued on.
        stream: &'a Stream,
    },
    /// `hipMemcpyAsync` D2D between two ranges of device memory on the SAME device.
    ///
    /// The one copy the engine had no binding for until V4, and the one a sequence-group fork is
    /// made of: a branch's GDN state slot set is the leader's, and its first private KV page is a
    /// copy of the leader's partial one. Both are ranges of allocations this process already
    /// holds, so nothing crosses the host and nothing is staged.
    ///
    /// `src` and `dst` may be the SAME allocation — the KV pool is one buffer and a page copy is
    /// two offsets into it — and the ranges are then refused if they OVERLAP, because
    /// `hipMemcpyAsync` does not define that and a partly-copied page is a page nothing would
    /// report.
    MemcpyD2d {
        /// Destination allocation.
        dst: &'a DevicePtr,
        /// Byte offset into `dst`. Bounds-checked against its length when the batch is lowered.
        dst_offset: u64,
        /// Source allocation.
        src: &'a DevicePtr,
        /// Byte offset into `src`. Bounds-checked against its length when the batch is lowered.
        src_offset: u64,
        /// Bytes to copy.
        bytes: u64,
        /// Stream the copy is issued on.
        stream: &'a Stream,
    },
    /// `hipMemcpyPeerAsync` between two devices.
    MemcpyPeer {
        /// Destination allocation on `dst_device`.
        dst: &'a DevicePtr,
        /// Destination device ordinal.
        dst_device: i32,
        /// Source allocation on `src_device`.
        src: &'a DevicePtr,
        /// Source device ordinal.
        src_device: i32,
        /// Bytes to copy; bounds-checked against both allocations.
        bytes: u64,
        /// Stream the copy is issued on.
        stream: &'a Stream,
    },
    /// `hipMemcpyAsync` between a range of a device allocation and a range of ONE pinned host
    /// allocation, in the direction `kind` names (`HostToDevice` or `DeviceToHost` only).
    ///
    /// FN8's expert mover: the tier's RAM slab is one `hipHostMalloc` of thousands of 2.6 MB slots
    /// and one step moves a victim OUT of a VRAM slot into the spare RAM slot and a promoted record
    /// IN from another RAM slot of the same slab, in one ordered batch. A [`PinnedSliceMut`] per
    /// destination cannot coexist with a [`PinnedSlice`] of the same buffer, so this variant takes
    /// the buffer by shared reference and both ranges by offset; both are bounds-checked when the
    /// batch is lowered. Ordering against every other reader or writer of either range is the
    /// caller's, by stream and event, exactly as for the other asynchronous copies here.
    MemcpyPinned {
        /// Device allocation.
        dev: &'a DevicePtr,
        /// Byte offset into `dev`.
        dev_offset: u64,
        /// Pinned host allocation.
        host: &'a PinnedBuf,
        /// Byte offset into `host`.
        host_offset: u64,
        /// Bytes to copy.
        bytes: u64,
        /// `HostToDevice` (host range -> device range) or `DeviceToHost`.
        kind: MemcpyKind,
        /// Stream the copy is issued on.
        stream: &'a Stream,
    },
    /// `hipMemsetAsync` over the whole allocation.
    Memset {
        /// Allocation to fill.
        dst: &'a DevicePtr,
        /// Byte value.
        value: u8,
        /// Stream the fill is issued on.
        stream: &'a Stream,
    },
    /// `hipEventRecord`.
    RecordEvent {
        /// Event to record.
        event: &'a Event,
        /// Stream to record it on.
        stream: &'a Stream,
    },
    /// `hipStreamWaitEvent`: `stream` waits for `event`, the host does not.
    StreamWaitEvent {
        /// Stream that waits.
        stream: &'a Stream,
        /// Event waited for.
        event: &'a Event,
    },
    /// `hipStreamWaitValue32(GEQ)`: `stream` waits, in the command processor, until the 32-bit
    /// word at `word` (masked by `mask`) is `>= value` (P27-1d: the split all-reduce's handshake).
    /// The host does not block and no CU spins.
    StreamWaitValueGte {
        /// Stream that waits.
        stream: &'a Stream,
        /// The device word compared (4 bytes).
        word: DeviceAddr,
        /// The threshold.
        value: u32,
        /// Bits of the word compared.
        mask: u32,
    },
    /// `hipStreamSynchronize`.
    SyncStream {
        /// Stream to drain.
        stream: &'a Stream,
    },
    /// `hipEventSynchronize`.
    SyncEvent {
        /// Event to wait for.
        event: &'a Event,
    },
    /// `hipDeviceSynchronize`.
    DeviceSynchronize,
}

/// **WHICH POINTER `hipMemcpyAsync` TAKES AS `dst` AND WHICH AS `src`.**
///
/// `hipMemcpyAsync(dst, src, bytes, kind, stream)` copies `src -> dst` and the KIND only says which
/// address space each is in — it does not reorder them. So a DEVICE-TO-HOST copy takes the HOST
/// pointer first and a HOST-TO-DEVICE copy takes the DEVICE pointer first, and a lowering that
/// puts the device address first for both is a copy that runs BACKWARDS while reporting success.
///
/// The batch path did exactly that for [`Job::MemcpyD2h`] (found ws-h31, 2026-09-22): the copy
/// wrote page-locked host memory INTO the device range it was supposed to read, so the caller's
/// buffer kept whatever it already held and the device range was corrupted with it. Its two
/// callers were the engine's batched logits gather — every `bits` read back as a fresh page's zero,
/// which `tp2::combine` then resolves as a tie to the LOWER GLOBAL ID, i.e. always rank 0's column,
/// which is right for every token whose id is in rank 0's half and wrong for the rest — and the
/// ASYNC session park, which is why a restored session diverged.
///
/// It is a function so that both lowerings read the same law and a host test can hold it.
const fn memcpy_args(kind: MemcpyKind, device: usize, host: usize) -> (usize, usize) {
    match kind {
        // `dst` is the device range, `src` the host one.
        MemcpyKind::HostToDevice => (device, host),
        // …and the other way round. This is the line that was missing.
        MemcpyKind::DeviceToHost => (host, device),
        // Nothing else in this module lowers through here; a device-to-device copy has its own
        // variant and `Default`/`HostToHost` have no caller.
        _ => (device, host),
    }
}

/// A lowered [`Job`]: handles replaced by the `Send` integer tokens, ready to cross the channel.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RawJob {
    Memcpy {
        /// What `hipMemcpyAsync` takes as its FIRST argument, already resolved by
        /// [`memcpy_args`] — not "the device one".
        dst: usize,
        /// …and its second.
        src: usize,
        bytes: u64,
        kind: MemcpyKind,
        stream: RawStream,
    },
    MemcpyD2d {
        dst: DeviceAddr,
        src: DeviceAddr,
        bytes: u64,
        stream: RawStream,
    },
    MemcpyPeer {
        dst: DeviceAddr,
        dst_device: i32,
        src: DeviceAddr,
        src_device: i32,
        bytes: u64,
        stream: RawStream,
    },
    Memset {
        dst: DeviceAddr,
        value: i32,
        bytes: u64,
        stream: RawStream,
    },
    RecordEvent {
        event: RawEvent,
        stream: RawStream,
    },
    StreamWaitEvent {
        stream: RawStream,
        event: RawEvent,
    },
    StreamWaitValueGte {
        stream: RawStream,
        word: DeviceAddr,
        value: u32,
        mask: u32,
    },
    SyncStream {
        stream: RawStream,
    },
    SyncEvent {
        event: RawEvent,
    },
    DeviceSynchronize,
}

impl RawJob {
    /// The streams and events this job names (FN59: what a later call must not overtake).
    pub(crate) fn names(&self) -> (Option<Name>, Option<Name>) {
        let s = |x: RawStream| Some(Name::Stream(x));
        let e = |x: RawEvent| Some(Name::Event(x));
        match *self {
            RawJob::Memcpy { stream, .. }
            | RawJob::MemcpyD2d { stream, .. }
            | RawJob::MemcpyPeer { stream, .. }
            | RawJob::Memset { stream, .. }
            | RawJob::StreamWaitValueGte { stream, .. }
            | RawJob::SyncStream { stream } => (s(stream), None),
            RawJob::RecordEvent { event, stream } | RawJob::StreamWaitEvent { stream, event } => {
                (s(stream), e(event))
            }
            RawJob::SyncEvent { event } => (e(event), None),
            RawJob::DeviceSynchronize => (None, None),
        }
    }

    /// Host-blocking jobs: a batch holding one is `Deps::All`, and the background lane refuses it.
    fn blocks_host(&self) -> bool {
        matches!(
            self,
            RawJob::SyncStream { .. } | RawJob::SyncEvent { .. } | RawJob::DeviceSynchronize
        )
    }

    /// Check the job and strip it to its `Send` tokens. Every bounds check in the batch path is
    /// here, on the caller's thread, before anything is posted.
    fn lower(job: &Job<'_>) -> Result<RawJob> {
        Ok(match *job {
            Job::MemcpyH2d {
                dst,
                dst_offset,
                src,
                stream,
            } => {
                let bytes = src.len() as u64;
                let end = dst_offset.checked_add(bytes).ok_or_else(|| {
                    Error::Invalid(format!(
                        "H2D at offset {dst_offset} of {bytes} bytes overflows a u64"
                    ))
                })?;
                if end > dst.len() {
                    return Err(Error::Invalid(format!(
                        "H2D of {bytes} bytes at offset {dst_offset} runs past a {} byte allocation",
                        dst.len()
                    )));
                }
                let addr = dst.sub_addr(dst_offset).ok_or_else(|| {
                    Error::Invalid(format!(
                        "H2D destination offset {dst_offset} is out of range"
                    ))
                })?;
                let (dst, src) = memcpy_args(
                    MemcpyKind::HostToDevice,
                    addr.as_u64() as usize,
                    src.as_usize(),
                );
                RawJob::Memcpy {
                    dst,
                    src,
                    bytes,
                    kind: MemcpyKind::HostToDevice,
                    stream: stream.raw,
                }
            }
            Job::MemcpyD2h {
                src,
                src_offset,
                ref dst,
                stream,
            } => {
                let bytes = dst.len() as u64;
                let end = src_offset.checked_add(bytes).ok_or_else(|| {
                    Error::Invalid(format!(
                        "D2H at offset {src_offset} of {bytes} bytes overflows a u64"
                    ))
                })?;
                if end > src.len() {
                    return Err(Error::Invalid(format!(
                        "D2H of {bytes} bytes at offset {src_offset} runs past a {} byte allocation",
                        src.len()
                    )));
                }
                let addr = src.sub_addr(src_offset).ok_or_else(|| {
                    Error::Invalid(format!("D2H source offset {src_offset} is out of range"))
                })?;
                let (hip_dst, hip_src) = memcpy_args(
                    MemcpyKind::DeviceToHost,
                    addr.as_u64() as usize,
                    dst.as_usize(),
                );
                RawJob::Memcpy {
                    dst: hip_dst,
                    src: hip_src,
                    bytes,
                    kind: MemcpyKind::DeviceToHost,
                    stream: stream.raw,
                }
            }
            Job::MemcpyD2d {
                dst,
                dst_offset,
                src,
                src_offset,
                bytes,
                stream,
            } => {
                let dst_end = dst_offset.checked_add(bytes).ok_or_else(|| {
                    Error::Invalid(format!(
                        "D2D into offset {dst_offset} of {bytes} bytes overflows a u64"
                    ))
                })?;
                let src_end = src_offset.checked_add(bytes).ok_or_else(|| {
                    Error::Invalid(format!(
                        "D2D out of offset {src_offset} of {bytes} bytes overflows a u64"
                    ))
                })?;
                if dst_end > dst.len() {
                    return Err(Error::Invalid(format!(
                        "D2D of {bytes} bytes at offset {dst_offset} runs past a {} byte \
                         destination",
                        dst.len()
                    )));
                }
                if src_end > src.len() {
                    return Err(Error::Invalid(format!(
                        "D2D of {bytes} bytes from offset {src_offset} runs past a {} byte source",
                        src.len()
                    )));
                }
                // The same allocation on both sides is the ordinary case here -- the KV pool is
                // one buffer and a page copy is two offsets into it -- so the check that matters
                // is OVERLAP, which `hipMemcpyAsync` does not define.
                if dst.addr() == src.addr() && dst_offset < src_end && src_offset < dst_end {
                    return Err(Error::Invalid(format!(
                        "D2D of {bytes} bytes from offset {src_offset} to offset {dst_offset} of \
                         one allocation overlaps itself"
                    )));
                }
                let d = dst.sub_addr(dst_offset).ok_or_else(|| {
                    Error::Invalid(format!(
                        "D2D destination offset {dst_offset} is out of range"
                    ))
                })?;
                let s = src.sub_addr(src_offset).ok_or_else(|| {
                    Error::Invalid(format!("D2D source offset {src_offset} is out of range"))
                })?;
                RawJob::MemcpyD2d {
                    dst: d,
                    src: s,
                    bytes,
                    stream: stream.raw,
                }
            }
            Job::MemcpyPeer {
                dst,
                dst_device,
                src,
                src_device,
                bytes,
                stream,
            } => {
                if bytes > dst.len() || bytes > src.len() {
                    return Err(Error::Invalid(format!(
                        "peer copy of {bytes} bytes between a {} byte source and a {} byte \
                         destination",
                        src.len(),
                        dst.len()
                    )));
                }
                RawJob::MemcpyPeer {
                    dst: dst.addr(),
                    dst_device,
                    src: src.addr(),
                    src_device,
                    bytes,
                    stream: stream.raw,
                }
            }
            Job::MemcpyPinned {
                dev,
                dev_offset,
                host,
                host_offset,
                bytes,
                kind,
                stream,
            } => {
                if !matches!(kind, MemcpyKind::HostToDevice | MemcpyKind::DeviceToHost) {
                    return Err(Error::Invalid(format!(
                        "pinned copy of kind {kind:?}: only HostToDevice / DeviceToHost"
                    )));
                }
                let dev_end = dev_offset.checked_add(bytes);
                let host_end = host_offset.checked_add(bytes);
                if dev_end.is_none_or(|e| e > dev.len())
                    || host_end.is_none_or(|e| e > host.len() as u64)
                {
                    return Err(Error::Invalid(format!(
                        "pinned copy of {bytes} bytes at device offset {dev_offset} (of {}) / host \
                         offset {host_offset} (of {}) runs past an allocation",
                        dev.len(),
                        host.len()
                    )));
                }
                let d = dev.sub_addr(dev_offset).ok_or_else(|| {
                    Error::Invalid(format!(
                        "pinned copy device offset {dev_offset} out of range"
                    ))
                })?;
                let h = host.as_usize() + host_offset as usize;
                let (hip_dst, hip_src) = memcpy_args(kind, d.as_u64() as usize, h);
                RawJob::Memcpy {
                    dst: hip_dst,
                    src: hip_src,
                    bytes,
                    kind,
                    stream: stream.raw,
                }
            }
            Job::Memset { dst, value, stream } => RawJob::Memset {
                dst: dst.addr(),
                value: i32::from(value),
                bytes: dst.len(),
                stream: stream.raw,
            },
            Job::RecordEvent { event, stream } => RawJob::RecordEvent {
                event: event.raw,
                stream: stream.raw,
            },
            Job::StreamWaitEvent { stream, event } => RawJob::StreamWaitEvent {
                stream: stream.raw,
                event: event.raw,
            },
            Job::StreamWaitValueGte {
                stream,
                word,
                value,
                mask,
            } => RawJob::StreamWaitValueGte {
                stream: stream.raw,
                word,
                value,
                mask,
            },
            Job::SyncStream { stream } => RawJob::SyncStream { stream: stream.raw },
            Job::SyncEvent { event } => RawJob::SyncEvent { event: event.raw },
            Job::DeviceSynchronize => RawJob::DeviceSynchronize,
        })
    }

    /// Issue the call. Runs on the HIP thread, inside one `run` or from the background lane.
    pub(crate) fn exec(self, lib: &Lib, ctx: &Ctx) -> Result<()> {
        // SAFETY (every arm): each handle was borrowed by the caller's frame when the batch was
        // lowered, and `run_batch` blocks until this job list is finished, so every allocation,
        // stream and event named here is alive for the duration of the call. For a BACKGROUND
        // batch (FN59) the same holds by FIFO: every handle's destructor is a `Deps::All` message
        // on this channel, posted after the submit, and `Deps::All` drains the lane before it
        // runs. The byte counts were bounds-checked in `lower`.
        match self {
            RawJob::Memcpy {
                dst,
                src,
                bytes,
                kind,
                stream,
            } => unsafe {
                lib.memcpy_async(
                    ctx,
                    dst as *mut c_void,
                    src as *const c_void,
                    bytes,
                    kind,
                    stream.as_ptr(),
                )
            },
            RawJob::MemcpyD2d {
                dst,
                src,
                bytes,
                stream,
            } => unsafe {
                lib.memcpy_async(
                    ctx,
                    dst.as_ptr(),
                    src.as_ptr().cast_const(),
                    bytes,
                    MemcpyKind::DeviceToDevice,
                    stream.as_ptr(),
                )
            },
            RawJob::MemcpyPeer {
                dst,
                dst_device,
                src,
                src_device,
                bytes,
                stream,
            } => unsafe {
                lib.memcpy_peer_async(
                    ctx,
                    dst.as_ptr(),
                    dst_device,
                    src.as_ptr(),
                    src_device,
                    bytes,
                    stream.as_ptr(),
                )
            },
            RawJob::Memset {
                dst,
                value,
                bytes,
                stream,
            } => unsafe { lib.memset_async(ctx, dst.as_ptr(), value, bytes, stream.as_ptr()) },
            RawJob::RecordEvent { event, stream } => unsafe {
                lib.event_record(ctx, event.as_ptr(), stream.as_ptr())
            },
            RawJob::StreamWaitEvent { stream, event } => unsafe {
                lib.stream_wait_event(ctx, stream.as_ptr(), event.as_ptr())
            },
            RawJob::StreamWaitValueGte {
                stream,
                word,
                value,
                mask,
            } => unsafe {
                lib.stream_wait_value32_gte(ctx, stream.as_ptr(), word.as_ptr(), value, mask)
            },
            RawJob::SyncStream { stream } => unsafe {
                lib.stream_synchronize(ctx, stream.as_ptr())
            },
            RawJob::SyncEvent { event } => unsafe { lib.event_synchronize(ctx, event.as_ptr()) },
            RawJob::DeviceSynchronize => lib.device_synchronize(ctx),
        }
    }
}

/// Owns the HIP runtime and the thread it is allowed to be called from.
///
/// Dropping it stops the thread. Any handle still alive at that point logs and leaks rather than
/// calling HIP from the dropping thread.
pub struct HipThread {
    tx: Sender<Msg>,
    join: Option<JoinHandle<()>>,
    info: RuntimeInfo,
    /// The HIP thread's kernel thread id (FN57: the CPU census and the pinning law name it).
    tid: i32,
    /// FN57: a bounded wait passed its deadline and the thread is still inside it -- `Drop` then
    /// detaches instead of joining a thread that will not come back.
    wedged: AtomicBool,
    /// FN57: the schedule this handle last SET (`u32::MAX` = never set: the runtime's default).
    schedule: AtomicU32,
    /// FN106d: the page form of this handle's pinned allocations ([`crate::PinnedPages`]).
    pinned_pages: std::sync::atomic::AtomicU8,
    rccl: Mutex<Option<RcclState>>,
}

impl HipThread {
    /// Load the HIP runtime on a fresh thread and bind every entry point.
    ///
    /// Loading is **lazy and fallible**: on a host with no ROCm -- which is this box outside the
    /// containers -- this returns [`Error::NotAvailable`], and the unit tests depend on that.
    ///
    /// The soname is a parameter, never an environment variable: there is no `getenv` in the
    /// engine, and "swap the ROCm userspace under one binary" is a plan/CLI decision.
    pub fn spawn(soname: &str) -> Result<HipThread> {
        let soname = soname.to_string();
        let (tx, rx) = mpsc::channel::<Msg>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(RuntimeInfo, i32)>>();

        let thread_soname = soname.clone();
        let join = std::thread::Builder::new()
            .name("saber-hip".to_string())
            .spawn(move || {
                let ctx = Ctx::new();
                let lib = match Lib::open(&thread_soname) {
                    Ok(lib) => lib,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let info = (|| {
                    Ok(RuntimeInfo {
                        soname: lib.soname().to_string(),
                        driver_version: lib.driver_version(&ctx)?,
                        runtime_version: lib.runtime_version(&ctx)?,
                    })
                })();
                let ok = info.is_ok();
                let _ = ready_tx.send(info.map(|i| (i, crate::affinity::current_tid())));
                if !ok {
                    return;
                }
                tracing::debug!(soname = %thread_soname, "HIP thread up");
                // FN59: the loop (with the background lane and its drain law) lives in `lane`.
                crate::lane::serve(&rx, &lib, &ctx, |j, lib, ctx| j.exec(lib, ctx));
                tracing::debug!("HIP thread down");
            })
            .map_err(|e| Error::Invalid(format!("could not spawn the HIP thread: {e}")))?;

        match ready_rx.recv() {
            Ok(Ok((info, tid))) => Ok(HipThread {
                tx,
                join: Some(join),
                info,
                tid,
                wedged: AtomicBool::new(false),
                schedule: AtomicU32::new(u32::MAX),
                pinned_pages: std::sync::atomic::AtomicU8::new(0),
                rccl: Mutex::new(None),
            }),
            Ok(Err(e)) => {
                let _ = join.join();
                Err(e)
            }
            Err(_) => {
                let _ = join.join();
                Err(Error::ThreadGone)
            }
        }
    }

    /// Identity and versions of the loaded runtime.
    #[must_use]
    pub fn info(&self) -> &RuntimeInfo {
        &self.info
    }

    /// Run one closure on the HIP thread and block until it returns.
    ///
    /// Crate-private on purpose. Other crates do not get a raw escape hatch into HIP; they get the
    /// typed methods below, and a new HIP operation is added by adding one here. That is the seam
    /// that keeps `unsafe extern "C"` inside this crate.
    pub(crate) fn run<'a, R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Lib, &Ctx) -> Result<R> + Send + 'a,
        R: Send + 'a,
    {
        self.run_with(Deps::All, f)
    }

    /// [`HipThread::run`] with the call's [`Deps`]: what of the background lane it may overtake.
    /// A wrong `Deps` is an ordering bug, so only typed methods that name every stream and event
    /// they touch pass anything but `Deps::All`.
    pub(crate) fn run_with<'a, R, F>(&self, deps: Deps, f: F) -> Result<R>
    where
        F: FnOnce(&Lib, &Ctx) -> Result<R> + Send + 'a,
        R: Send + 'a,
    {
        let (done_tx, done_rx) = mpsc::sync_channel::<Result<R>>(1);
        let job: BorrowedTask<'a> = Box::new(move |lib, ctx| {
            // `f` is consumed here, so everything it borrowed is released before the result is
            // handed back.
            let result = f(lib, ctx);
            let _ = done_tx.send(result);
        });

        // SAFETY: the job is erased to 'static only to cross the channel. This function blocks on
        // `done_rx.recv()` until either the job has produced its result (at which point `f` and
        // everything it captured have already been dropped on the worker) or the channel has been
        // torn down because the worker is gone. So no borrow the closure holds can outlive this
        // stack frame, which is what `'static` would otherwise be asserting.
        let job: Task = unsafe { std::mem::transmute::<BorrowedTask<'a>, Task>(job) };

        self.tx
            .send(Msg::Call(job, deps))
            .map_err(|_| Error::ThreadGone)?;
        done_rx.recv().map_err(|_| Error::ThreadGone)?
    }

    /// FN59: **NON-BLOCKING batch submit** -- the [`HipThread::run_batch`] batch, issued by the
    /// HIP thread's BACKGROUND LANE while the caller goes on.
    ///
    /// Lowered (every bounds check) on the calling thread exactly as `run_batch` lowers it, then
    /// posted; returns at once with a [`BatchTicket`]. The HIP thread issues the jobs in order on
    /// their own streams, one at a time whenever its channel is empty; a later call overtakes
    /// pending jobs only if it names none of their streams or events (`crate::lane` states the
    /// law). Same copies, same streams, same events, same order on every stream as `run_batch` --
    /// only the HOST time at which each is issued moves.
    ///
    /// The device-side wait is unchanged: put a `Job::RecordEvent` in the batch and
    /// `stream_wait_event` on it whenever the consumer is posted -- the wait names the event, so
    /// the lane issues the record first. Host-blocking jobs (`SyncStream`, `SyncEvent`,
    /// `DeviceSynchronize`) are refused: they would block the HIP thread inside the lane. Every
    /// handle named must belong to THIS `HipThread` (its destructor is what the FIFO orders).
    ///
    /// # Errors
    /// A lowering failure or a refused job (nothing posted); `ThreadGone`. A failure while
    /// issuing is reported by the ticket.
    pub fn submit_batch(&self, jobs: &[Job<'_>]) -> Result<BatchTicket> {
        let raw: Vec<RawJob> = jobs.iter().map(RawJob::lower).collect::<Result<_>>()?;
        if let Some(i) = raw.iter().position(RawJob::blocks_host) {
            return Err(Error::Invalid(format!(
                "submit_batch: job {i} ({:?}) blocks the host; a background batch issues \
                 asynchronous work only (wait on its ticket or an event instead)",
                raw[i]
            )));
        }
        post_background(&self.tx, raw)
    }

    /// FN59: block until every background batch submitted so far has been issued (an empty
    /// `Deps::All` call: the lane drains before it runs).
    pub fn flush_background(&self) -> Result<()> {
        self.run(|_, _| Ok(()))
    }

    pub(crate) fn sender(&self) -> Sender<Msg> {
        self.tx.clone()
    }
}

/// Post a lowered batch to the background lane; the ticket's channel is the batch's.
pub(crate) fn post_background<L>(tx: &Sender<Msg<L>>, jobs: Vec<RawJob>) -> Result<BatchTicket> {
    let (done, rx) = mpsc::sync_channel::<Result<usize>>(1);
    let n = jobs.len();
    tx.send(Msg::Background(Background { jobs, done }))
        .map_err(|_| Error::ThreadGone)?;
    Ok(BatchTicket::new(rx, n))
}

impl HipThread {
    // ---- devices ------------------------------------------------------------------------------ //

    /// `hipGetDeviceCount`.
    pub fn device_count(&self) -> Result<i32> {
        self.run(|lib, ctx| lib.device_count(ctx))
    }

    /// `hipSetDevice`. Applies to the HIP thread, which is the only thread that matters.
    pub fn set_device(&self, device: i32) -> Result<()> {
        self.run(move |lib, ctx| lib.set_device(ctx, device))
    }

    /// `hipGetDevice`.
    pub fn current_device(&self) -> Result<i32> {
        self.run(|lib, ctx| lib.current_device(ctx))
    }

    /// `hipDeviceGetAttribute`. See [`crate::DeviceAttribute`] for the attribute ids.
    pub fn device_attribute(&self, device: i32, attr: i32) -> Result<i32> {
        self.run(move |lib, ctx| lib.device_attribute(ctx, device, attr))
    }

    /// `hipDeviceGetName`.
    pub fn device_name(&self, device: i32) -> Result<String> {
        self.run(move |lib, ctx| lib.device_name(ctx, device))
    }

    /// `hipDeviceTotalMem`.
    pub fn device_total_mem(&self, device: i32) -> Result<u64> {
        self.run(move |lib, ctx| lib.device_total_mem(ctx, device))
    }

    /// `hipDeviceGetPCIBusId`.
    pub fn device_pci_bus_id(&self, device: i32) -> Result<String> {
        self.run(move |lib, ctx| lib.device_pci_bus_id(ctx, device))
    }

    /// `hipDeviceCanAccessPeer`: whether `device` can address `peer`'s memory.
    pub fn can_access_peer(&self, device: i32, peer: i32) -> Result<bool> {
        self.run(move |lib, ctx| lib.can_access_peer(ctx, device, peer))
    }

    /// `hipDeviceEnablePeerAccess`, for the *current* device. Startup-only, and a prerequisite for
    /// the 2-rank all-reduce.
    pub fn enable_peer_access(&self, peer: i32) -> Result<()> {
        self.run(move |lib, ctx| lib.enable_peer_access(ctx, peer))
    }

    /// `hipMemGetInfo` for the current device: `(free, total)` bytes.
    pub fn mem_info(&self) -> Result<(u64, u64)> {
        self.run(|lib, ctx| lib.mem_get_info(ctx))
    }

    // ---- memory -------------------------------------------------------------------------------- //

    /// `hipMalloc`. The only way to obtain a [`DevicePtr`].
    pub fn malloc(&self, bytes: u64) -> Result<DevicePtr> {
        if bytes == 0 {
            return Err(Error::Invalid("hipMalloc of 0 bytes".into()));
        }
        let addr = self.run(move |lib, ctx| lib.malloc(ctx, bytes).map(DeviceAddr::from_raw))?;
        Ok(DevicePtr::new(addr, bytes, self.sender()))
    }

    /// `hipExtMallocWithFlags`. The only way to obtain device memory that is not plain
    /// `hipMalloc` -- in practice, the fine-grained memory a peer's stores are coherent in.
    ///
    /// See [`MallocFlags::FineGrained`] for why the all-reduce needs it and why the device
    /// attribute is not what decides.
    pub fn malloc_with_flags(&self, bytes: u64, flags: MallocFlags) -> Result<DevicePtr> {
        if bytes == 0 {
            return Err(Error::Invalid("hipExtMallocWithFlags of 0 bytes".into()));
        }
        let addr = self.run(move |lib, ctx| {
            lib.ext_malloc_with_flags(ctx, bytes, flags)
                .map(DeviceAddr::from_raw)
        })?;
        Ok(DevicePtr::new(addr, bytes, self.sender()))
    }

    /// FN76: import a dma-buf (`fd`, `bytes`) as device memory HIP knows (copies and kernels
    /// address it like a `hipMalloc` range; `hipFree` on drop releases the mapping). The fd is
    /// handed to the runtime (pass a dup the caller does not close).
    pub fn import_dmabuf(&self, fd: i32, bytes: u64) -> Result<DevicePtr> {
        if bytes == 0 || fd < 0 {
            return Err(Error::Invalid(format!(
                "import of dma-buf fd {fd}, {bytes} bytes"
            )));
        }
        // SAFETY: the caller hands over an open dma-buf of `bytes` bytes.
        let addr = self.run(move |lib, ctx| unsafe {
            lib.import_dmabuf(ctx, fd, bytes).map(DeviceAddr::from_raw)
        })?;
        Ok(DevicePtr::new(addr, bytes, self.sender()))
    }

    /// FN106d: set the page form of every later [`HipThread::host_malloc`] (`--fn-pinned-pages`;
    /// set once at boot, before the first pinned allocation).
    pub fn set_pinned_pages(&self, p: crate::PinnedPages) {
        self.pinned_pages.store(p.as_u8(), Ordering::SeqCst);
    }

    /// FN106d: the page form [`HipThread::host_malloc`] uses.
    #[must_use]
    pub fn pinned_pages(&self) -> crate::PinnedPages {
        crate::PinnedPages::from_u8(self.pinned_pages.load(Ordering::SeqCst))
    }

    /// Pinned host memory, zeroed -- THE ONE ALLOCATOR PATH (FN106d): `hipHostMalloc` with
    /// `hipHostMallocDefault` under [`crate::PinnedPages::Shmem`] (the default), the huge form
    /// ([`HipThread::host_malloc_huge`]) under [`crate::PinnedPages::Huge`] for allocations of at
    /// least [`crate::pinned::HUGE_MIN_BYTES`], falling back BY NAME to `hipHostMalloc` (counted
    /// with its reason in [`crate::pinned::stats`]).
    pub fn host_malloc(&self, bytes: u64) -> Result<PinnedBuf> {
        if bytes == 0 {
            return Err(Error::Invalid("hipHostMalloc of 0 bytes".into()));
        }
        if self.pinned_pages().is_huge() && bytes >= crate::pinned::HUGE_MIN_BYTES {
            match self.host_malloc_huge(bytes) {
                Ok(b) => return Ok(b),
                Err(reason) => {
                    tracing::warn!(bytes, %reason, "FN106d: the huge pinned form fell back to hipHostMalloc");
                    crate::pinned::stats().add_fallback(bytes, reason);
                }
            }
        }
        let p = self.run(move |lib, ctx| lib.host_malloc(ctx, bytes, 0).map(|p| p as usize))?;
        Ok(PinnedBuf::new_counted(
            p as *mut c_void,
            bytes as usize,
            self.sender(),
        ))
    }

    /// FN106d: the huge pinned form -- a 2 MiB-aligned private anonymous mapping +
    /// `MADV_HUGEPAGE`, faulted in, then `hipHostRegister` (portable | mapped | coarse-grained),
    /// and the device address checked to BE the host address (UVA identity, which every in-place
    /// reader relies on). The buffer unregisters, then unmaps, on drop.
    ///
    /// # Errors
    /// The failing step by name (nothing is left mapped or registered).
    pub fn host_malloc_huge(&self, bytes: u64) -> std::result::Result<PinnedBuf, String> {
        let t0 = std::time::Instant::now();
        let m = crate::pinned::map_huge(bytes)?;
        let (base, len) = (m.base, m.map_len);
        // FN121 (`huge-collapse`): the 4 KiB pages the THP fault left collapsed BEFORE the
        // register (a registered 4 KiB page is one kcompactd may migrate = a KFD eviction)
        if self.pinned_pages() == crate::PinnedPages::HugeCollapse {
            let tc = std::time::Instant::now();
            let before = crate::pinned::smaps_huge_bytes(&[(base, len)]).unwrap_or(0);
            let (calls, err) = crate::pinned::collapse(&m, 3);
            let after = crate::pinned::smaps_huge_bytes(&[(base, len)]).unwrap_or(0);
            crate::pinned::stats().add_collapse(
                len as u64,
                before,
                after,
                tc.elapsed().as_nanos() as u64,
                (err != 0).then(|| {
                    format!("MADV_COLLAPSE errno {err} after {calls} call(s) (a partial collapse)")
                }),
            );
        }
        let reg = self.run(move |lib, ctx| {
            // SAFETY: [base, base + len) is the live mapping made above; it stays mapped until
            // the unregister below or the buffer's drop.
            unsafe {
                lib.host_register(
                    ctx,
                    base as *mut c_void,
                    len as u64,
                    crate::pinned::REGISTER_FLAGS,
                )
            }?;
            // SAFETY: as above, registered.
            let d = unsafe { lib.host_get_device_pointer(ctx, base as *mut c_void) };
            match d {
                Ok(d) if d as usize == base => Ok(Ok(())),
                other => {
                    // SAFETY: registered just above; unregister before the caller unmaps
                    let _ = unsafe { lib.host_unregister(ctx, base as *mut c_void) };
                    Ok(Err(match other {
                        Ok(d) => format!("device pointer {:#x} != host {base:#x}", d as usize),
                        Err(e) => format!("hipHostGetDevicePointer: {e}"),
                    }))
                }
            }
        });
        let fail = match reg {
            Ok(Ok(())) => None,
            Ok(Err(why)) => Some(why),
            Err(e) => Some(format!("hipHostRegister: {e}")),
        };
        if let Some(why) = fail {
            crate::pinned::unmap(&m);
            return Err(why);
        }
        crate::pinned::stats().add_huge(base, len, t0.elapsed().as_nanos() as u64);
        Ok(PinnedBuf::new_registered(
            base,
            bytes as usize,
            len,
            self.sender(),
        ))
    }

    /// FN53c: `hipHostMalloc` with `hipHostMallocCoherent | hipHostMallocMapped`, zeroed -- fine-
    /// grained pinned memory a running kernel and the host exchange words through (system-scope
    /// atomics on the device, [`PinnedBuf::load_u32_acquire`] / [`PinnedBuf::store_u32_release`]
    /// on the host).
    pub fn host_malloc_coherent(&self, bytes: u64) -> Result<PinnedBuf> {
        if bytes == 0 {
            return Err(Error::Invalid("hipHostMalloc of 0 bytes".into()));
        }
        const COHERENT_MAPPED: u32 = 0x4000_0000 | 0x2;
        let p = self.run(move |lib, ctx| {
            lib.host_malloc(ctx, bytes, COHERENT_MAPPED)
                .map(|p| p as usize)
        })?;
        Ok(PinnedBuf::new_counted(
            p as *mut c_void,
            bytes as usize,
            self.sender(),
        ))
    }

    /// `hipHostGetDevicePointer`: the address a kernel reads `buf` at over UVA (FN8: the expert
    /// tier's pinned RAM slots, which FN4's `xptr` tables point at in place). On this box's ROCm the
    /// mapping is the identity (FN7 hold), which the caller may check against [`PinnedBuf::addr`].
    pub fn host_device_addr(&self, buf: &PinnedBuf) -> Result<u64> {
        let p = buf.as_usize();
        self.run(move |lib, ctx| {
            // SAFETY: `buf` is a live `hipHostMalloc` allocation borrowed by the caller's frame for
            // the duration of this blocking call.
            unsafe { lib.host_get_device_pointer(ctx, p as *mut c_void) }.map(|d| d as u64)
        })
    }

    /// Asynchronous host-to-device copy from pinned staging. Both buffers are owned handles, so
    /// both outlive the call; ordering against the copy is the caller's, via `stream`.
    pub fn memcpy_h2d(&self, dst: &DevicePtr, src: &PinnedBuf, stream: &Stream) -> Result<()> {
        let bytes = src.len() as u64;
        if bytes > dst.len() {
            return Err(Error::Invalid(format!(
                "H2D copy of {bytes} bytes into a {} byte allocation",
                dst.len()
            )));
        }
        let (d, s, st) = (dst.addr(), src.as_usize(), stream.raw);
        self.run_with(Deps::on1(Name::Stream(st)), move |lib, ctx| {
            // SAFETY: both allocations are alive for the duration of this call (they are borrowed
            // by the caller's frame, which blocks until the job returns) and the byte count is
            // bounded by both lengths above.
            unsafe {
                lib.memcpy_async(
                    ctx,
                    d.as_ptr(),
                    s as *const c_void,
                    bytes,
                    MemcpyKind::HostToDevice,
                    st.as_ptr(),
                )
            }
        })
    }

    /// Asynchronous host-to-device copy from a pinned staging *view* into `dst` at `dst_offset`.
    ///
    /// The staging-ring form of [`HipThread::memcpy_h2d`]: a record larger than one slot crosses
    /// the link as several of these, and a record smaller than a slot copies only its own bytes.
    pub fn memcpy_h2d_at(
        &self,
        dst: &DevicePtr,
        dst_offset: u64,
        src: PinnedSlice<'_>,
        stream: &Stream,
    ) -> Result<()> {
        self.run_batch(&[Job::MemcpyH2d {
            dst,
            dst_offset,
            src,
            stream,
        }])
        .map(|_| ())
    }

    /// Run several HIP calls in **one** channel round trip.
    ///
    /// Returns the number of calls issued. The list is issued in order and stops at the first
    /// failure, whose index is logged: a half-issued batch is reported, never silently completed.
    ///
    /// The motivation is measured, not stylistic — a round trip to the HIP thread costs ~3.24 µs
    /// (workstream H), so the cold-start ring's `copy…copy, record-event` wave is posted as one
    /// batch instead of one call at a time.
    pub fn run_batch(&self, jobs: &[Job<'_>]) -> Result<usize> {
        if jobs.is_empty() {
            return Ok(0);
        }
        let raw: Vec<RawJob> = jobs.iter().map(RawJob::lower).collect::<Result<_>>()?;
        let n = raw.len();
        // FN59: a batch of asynchronous jobs overtakes the background lane unless it names one of
        // its streams or events; one holding a host-blocking job drains it.
        let deps = if raw.iter().any(RawJob::blocks_host) {
            Deps::All
        } else {
            let mut v = Vec::with_capacity(2 * n);
            for j in &raw {
                let (a, b) = j.names();
                v.extend(a);
                v.extend(b);
            }
            Deps::Many(v)
        };
        self.run_with(deps, move |lib, ctx| {
            for (i, job) in raw.into_iter().enumerate() {
                if let Err(e) = job.exec(lib, ctx) {
                    tracing::error!(
                        job = i,
                        batch = n,
                        error = %e,
                        "HIP batch failed; the jobs after this one were not issued"
                    );
                    return Err(e);
                }
            }
            Ok(n)
        })
    }

    /// Blocking device-to-device copy: `bytes` from `src + src_offset` to `dst + dst_offset`,
    /// issued on `stream` and waited for.
    ///
    /// **This is what a sequence-group fork is made of** (V4). A branch of an `n > 1` request gets
    /// a copy of the leader's GDN state slot set and of the leader's one partial KV page, and both
    /// are ranges of allocations this process already holds; the alternative — device to host and
    /// back — would move 3 MiB per GDN layer through pinned staging to put it back where it came
    /// from.
    ///
    /// Blocking on purpose: a fork happens once per branch, between two steps, and the branch's
    /// first step must not be able to read a page that is still being written. Everything it
    /// checks (`bytes` against both allocations, an overlap inside one allocation) is checked on
    /// the calling thread before anything is posted, so a bad range is a named `Error::Invalid`
    /// and never a `hipErrorIllegalAddress` in a kernel forty layers later.
    pub fn memcpy_d2d_blocking(
        &self,
        dst: &DevicePtr,
        dst_offset: u64,
        src: &DevicePtr,
        src_offset: u64,
        bytes: u64,
        stream: &Stream,
    ) -> Result<()> {
        self.run_batch(&[
            Job::MemcpyD2d {
                dst,
                dst_offset,
                src,
                src_offset,
                bytes,
                stream,
            },
            Job::SyncStream { stream },
        ])
        .map(|_| ())
    }

    /// `hipMemcpyPeerAsync`: copy between two devices over the P2P link.
    pub fn memcpy_peer_async(
        &self,
        dst: &DevicePtr,
        dst_device: i32,
        src: &DevicePtr,
        src_device: i32,
        bytes: u64,
        stream: &Stream,
    ) -> Result<()> {
        self.run_batch(&[Job::MemcpyPeer {
            dst,
            dst_device,
            src,
            src_device,
            bytes,
            stream,
        }])
        .map(|_| ())
    }

    /// Asynchronous device-to-host copy into pinned staging.
    pub fn memcpy_d2h(&self, dst: &mut PinnedBuf, src: &DevicePtr, stream: &Stream) -> Result<()> {
        let bytes = dst.len() as u64;
        if bytes > src.len() {
            return Err(Error::Invalid(format!(
                "D2H copy of {bytes} bytes out of a {} byte allocation",
                src.len()
            )));
        }
        let (d, s, st) = (dst.as_usize(), src.addr(), stream.raw);
        self.run_with(Deps::on1(Name::Stream(st)), move |lib, ctx| {
            // SAFETY: as above.
            unsafe {
                lib.memcpy_async(
                    ctx,
                    d as *mut c_void,
                    s.as_ptr(),
                    bytes,
                    MemcpyKind::DeviceToHost,
                    st.as_ptr(),
                )
            }
        })
    }

    /// FN112: asynchronous device-to-host copy from any device address into a pinned staging
    /// *view* (the read-back form of [`HipThread::memcpy_h2d_at`]): queued on `stream` behind the
    /// work that writes the range, so the bytes cross the link when that work ends instead of after
    /// the host's next round trip. The caller reads the view only after an event recorded behind
    /// this copy has completed (the staging outlives the copy: it is the caller's buffer).
    pub fn memcpy_d2h_addr(
        &self,
        dst: crate::handles::PinnedSliceMut<'_>,
        src: DeviceAddr,
        stream: &Stream,
    ) -> Result<()> {
        let bytes = dst.len() as u64;
        if bytes == 0 {
            return Ok(());
        }
        if src == DeviceAddr::NULL {
            return Err(Error::Invalid("D2H from a null device address".to_owned()));
        }
        let (d, st) = (dst.as_usize(), stream.raw);
        self.run_with(Deps::on1(Name::Stream(st)), move |lib, ctx| {
            // SAFETY: `d` is a live pinned range of `bytes` bytes owned by the caller's
            // `PinnedBuf`, which the caller keeps alive (and unread) until an event behind this
            // copy completes; the device range is the caller's contract, as
            // [`HipThread::memcpy_d2h_addr_blocking`].
            unsafe {
                lib.memcpy_async(
                    ctx,
                    d as *mut c_void,
                    src.as_ptr(),
                    bytes,
                    MemcpyKind::DeviceToHost,
                    st.as_ptr(),
                )
            }
        })
    }

    /// Blocking host-to-device copy from ordinary (unpinned) host memory.
    ///
    /// Issues the copy and then synchronises the stream, which is what makes borrowing a plain
    /// `&[u8]` sound. It is a *setup* path -- one-shot uploads, tests, the `probe` smoke check --
    /// and is deliberately not what the load path uses.
    /// Blocking host-to-device copy from ordinary host memory into `dst` at `dst_offset`.
    ///
    /// The offset form of [`HipThread::memcpy_h2d_blocking`], and the upload path's declared slow
    /// route: a record too large for the pinned staging ring goes this way, which is one HIP-staged
    /// copy out of pageable memory instead of a DMA out of a staging slot. It is slower on purpose
    /// and it is counted, because a fallback nobody counts is a fallback nobody notices.
    pub fn memcpy_h2d_at_blocking(
        &self,
        dst: &DevicePtr,
        dst_offset: u64,
        src: &[u8],
        stream: &Stream,
    ) -> Result<()> {
        let bytes = src.len() as u64;
        let end = dst_offset.checked_add(bytes).ok_or_else(|| {
            Error::Invalid(format!(
                "H2D at offset {dst_offset} of {bytes} bytes overflows a u64"
            ))
        })?;
        if end > dst.len() {
            return Err(Error::Invalid(format!(
                "H2D of {bytes} bytes at offset {dst_offset} runs past a {} byte allocation",
                dst.len()
            )));
        }
        let addr = dst.sub_addr(dst_offset).ok_or_else(|| {
            Error::Invalid(format!(
                "H2D destination offset {dst_offset} is out of range"
            ))
        })?;
        let st = stream.raw;
        self.run(move |lib, ctx| {
            // SAFETY: `src` is valid for `bytes` for the duration of this job, the destination
            // range was bounds-checked above, and the stream is synchronised before the job
            // returns, so the copy has completed while `src` is still borrowed.
            unsafe {
                lib.memcpy_async(
                    ctx,
                    addr.as_ptr(),
                    src.as_ptr().cast(),
                    bytes,
                    MemcpyKind::HostToDevice,
                    st.as_ptr(),
                )?;
                lib.stream_synchronize(ctx, st.as_ptr())
            }
        })
    }

    /// Blocking host-to-device copy from ordinary (unpinned) host memory into the start of `dst`.
    ///
    /// Issues the copy and then synchronises the stream, which is what makes borrowing a plain
    /// `&[u8]` sound. It is a *setup* path -- one-shot uploads, tests, the `probe` smoke check --
    /// and is deliberately not what the load path's fast route uses.
    pub fn memcpy_h2d_blocking(&self, dst: &DevicePtr, src: &[u8], stream: &Stream) -> Result<()> {
        let bytes = src.len() as u64;
        if bytes > dst.len() {
            return Err(Error::Invalid(format!(
                "H2D copy of {bytes} bytes into a {} byte allocation",
                dst.len()
            )));
        }
        let (d, st) = (dst.addr(), stream.raw);
        self.run(move |lib, ctx| {
            // SAFETY: `src` is valid for `bytes` for the duration of this job, and the stream is
            // synchronised before the job returns, so the copy has completed while `src` is still
            // borrowed.
            unsafe {
                lib.memcpy_async(
                    ctx,
                    d.as_ptr(),
                    src.as_ptr().cast(),
                    bytes,
                    MemcpyKind::HostToDevice,
                    st.as_ptr(),
                )?;
                lib.stream_synchronize(ctx, st.as_ptr())
            }
        })
    }

    /// Blocking device-to-host copy into ordinary host memory. Same trade as
    /// [`HipThread::memcpy_h2d_blocking`].
    pub fn memcpy_d2h_blocking(
        &self,
        dst: &mut [u8],
        src: &DevicePtr,
        stream: &Stream,
    ) -> Result<()> {
        let bytes = dst.len() as u64;
        if bytes > src.len() {
            return Err(Error::Invalid(format!(
                "D2H copy of {bytes} bytes out of a {} byte allocation",
                src.len()
            )));
        }
        let (s, st) = (src.addr(), stream.raw);
        self.run(move |lib, ctx| {
            // SAFETY: as above; the stream is synchronised before `dst`'s borrow ends.
            unsafe {
                lib.memcpy_async(
                    ctx,
                    dst.as_mut_ptr().cast(),
                    s.as_ptr(),
                    bytes,
                    MemcpyKind::DeviceToHost,
                    st.as_ptr(),
                )?;
                lib.stream_synchronize(ctx, st.as_ptr())
            }
        })
    }

    /// Blocking device-to-host copy from a bare device ADDRESS (T4's race checks).
    ///
    /// For a region another owner keeps alive for the whole process -- the `Launcher`'s
    /// peer-visible protocol state, which is never freed while the engine runs -- so a reader that
    /// cannot borrow the owner (the step's pick swap has no `Launcher`) can still read it. The host
    /// side is safe whatever `src` is (the copy lands in `dst`, a live slice of the stated length,
    /// and the stream is synchronised before the borrow ends); an address that is not a live device
    /// range of `dst.len()` bytes is a device fault reported as an error, which is why every caller
    /// takes its addresses from the owner's own list (`Launcher::ar_seq_regions`).
    pub fn memcpy_d2h_addr_blocking(
        &self,
        dst: &mut [u8],
        src: DeviceAddr,
        stream: &Stream,
    ) -> Result<()> {
        let bytes = dst.len() as u64;
        if bytes == 0 {
            return Ok(());
        }
        if src == DeviceAddr::NULL {
            return Err(Error::Invalid("D2H from a null device address".to_owned()));
        }
        let st = stream.raw;
        self.run(move |lib, ctx| {
            // SAFETY: `dst` is borrowed by the caller's frame for the whole blocking job and the
            // stream is synchronised before the borrow ends; the device range is the caller's
            // contract (see above), and a bad one fails in the runtime, not on the host.
            unsafe {
                lib.memcpy_async(
                    ctx,
                    dst.as_mut_ptr().cast(),
                    src.as_ptr(),
                    bytes,
                    MemcpyKind::DeviceToHost,
                    st.as_ptr(),
                )?;
                lib.stream_synchronize(ctx, st.as_ptr())
            }
        })
    }

    /// Blocking device-to-host copy out of the middle of an allocation.
    ///
    /// The read-back form of [`HipThread::memcpy_h2d_at`], and the only way to audit one record
    /// inside a weight arena: a hash over what the device actually holds, rather than over what the
    /// host believes it sent.
    pub fn memcpy_d2h_at_blocking(
        &self,
        dst: &mut [u8],
        src: &DevicePtr,
        src_offset: u64,
        stream: &Stream,
    ) -> Result<()> {
        let bytes = dst.len() as u64;
        let end = src_offset.checked_add(bytes).ok_or_else(|| {
            Error::Invalid(format!(
                "D2H at offset {src_offset} of {bytes} bytes overflows a u64"
            ))
        })?;
        if end > src.len() {
            return Err(Error::Invalid(format!(
                "D2H of {bytes} bytes at offset {src_offset} runs past a {} byte allocation",
                src.len()
            )));
        }
        let addr = src.sub_addr(src_offset).ok_or_else(|| {
            Error::Invalid(format!("D2H source offset {src_offset} out of range"))
        })?;
        let st = stream.raw;
        self.run(move |lib, ctx| {
            // SAFETY: `dst` is borrowed by the caller's frame for the whole blocking job, the
            // source range was bounds-checked above, and the stream is synchronised before the
            // borrow ends, so the copy has completed while `dst` is still alive.
            unsafe {
                lib.memcpy_async(
                    ctx,
                    dst.as_mut_ptr().cast(),
                    addr.as_ptr(),
                    bytes,
                    MemcpyKind::DeviceToHost,
                    st.as_ptr(),
                )?;
                lib.stream_synchronize(ctx, st.as_ptr())
            }
        })
    }

    /// `hipMemsetAsync` over a whole allocation.
    pub fn memset(&self, dst: &DevicePtr, value: u8, stream: &Stream) -> Result<()> {
        let (d, bytes, st) = (dst.addr(), dst.len(), stream.raw);
        self.run_with(Deps::on1(Name::Stream(st)), move |lib, ctx| {
            // SAFETY: the allocation is alive and `bytes` is its own length.
            unsafe { lib.memset_async(ctx, d.as_ptr(), i32::from(value), bytes, st.as_ptr()) }
        })
    }

    /// `hipMemsetAsync` over a RANGE of an allocation, bounded against the allocation itself.
    ///
    /// [`HipThread::memset`] takes the whole allocation, which is the safe shape and the useless
    /// one for a caller that has to blank part of a planned buffer. This takes an offset and a
    /// length and REFUSES one that runs past the end, so the caller stays safe code -- which
    /// matters, because a caller of this is an engine crate and `CLAUDE.md` rule 2 keeps `unsafe`
    /// out of those.
    ///
    /// **It has no caller today.** It was written for ws-l16's `PadBlank`, which zeroed a padded
    /// prefill chunk's unfed query rows between the op that wrote them and the attention that read
    /// them -- 1664 memsets over a multi-turn run. ws-h27 retired that by pointing the attention's
    /// `q_len` at the chunk's live rows instead, so nothing reads those rows at all and there is
    /// nothing to blank. Kept as a bounded primitive, not as a live path.
    ///
    /// # Errors
    /// `at + bytes` past the end of `dst`, and whatever the runtime returns.
    pub fn memset_at(
        &self,
        dst: &DevicePtr,
        at: u64,
        bytes: u64,
        value: u8,
        stream: &Stream,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let len = dst.len();
        if at.saturating_add(bytes) > len {
            return Err(Error::Invalid(format!(
                "hipMemsetAsync: {bytes} B at offset {at} of a {len} B allocation"
            )));
        }
        let Some(d) = dst.addr().offset(at) else {
            return Err(Error::Invalid(format!(
                "hipMemsetAsync: offset {at} overflows the address of a {len} B allocation"
            )));
        };
        let st = stream.raw;
        self.run_with(Deps::on1(Name::Stream(st)), move |lib, ctx| {
            // SAFETY: the range was checked against the allocation's own length above, and the
            // allocation is alive for the duration of the call.
            unsafe { lib.memset_async(ctx, d.as_ptr(), i32::from(value), bytes, st.as_ptr()) }
        })
    }

    // ---- streams and events --------------------------------------------------------------------- //

    /// `hipStreamCreateWithFlags`.
    pub fn create_stream(&self, flags: StreamFlags) -> Result<Stream> {
        let raw =
            self.run(move |lib, ctx| lib.stream_create(ctx, flags).map(RawStream::from_raw))?;
        Ok(Stream {
            raw,
            tx: self.sender(),
            _not_send: PhantomData,
        })
    }

    /// FN72: a stream on a HARDWARE QUEUE OF ITS OWN -- `hipExtStreamCreateWithCUMask` with every
    /// CU of the device set (`cu_count` bits: no CU is excluded, so a kernel on it runs exactly as
    /// on a plain stream). ROCclr pools the plain streams of a priority over `GPU_MAX_HW_QUEUES`
    /// (4) hardware queues and re-assigns them dynamically; a masked stream always gets a new
    /// queue. An AQL queue is in order: a barrier packet on a shared queue blocks every stream
    /// behind it.
    pub fn create_stream_own_queue(&self, cu_count: u32) -> Result<Stream> {
        if cu_count == 0 {
            return Err(Error::Invalid("a CU mask over 0 CUs".into()));
        }
        let mask = full_cu_mask(cu_count);
        let raw = self.run(move |lib, ctx| {
            lib.stream_create_cu_mask(ctx, &mask)
                .map(RawStream::from_raw)
        })?;
        Ok(Stream {
            raw,
            tx: self.sender(),
            _not_send: PhantomData,
        })
    }

    /// `hipStreamSynchronize`.
    pub fn sync_stream(&self, stream: &Stream) -> Result<()> {
        let s = stream.raw;
        // SAFETY: the stream is alive for the duration of the blocking call.
        self.run(move |lib, ctx| unsafe { lib.stream_synchronize(ctx, s.as_ptr()) })
    }

    /// `hipEventSynchronize` with a DEADLINE (FN57): the wait itself runs on the HIP thread as
    /// [`HipThread::sync_event`] does -- an interrupt sleep under [`DeviceSchedule::BlockingSync`],
    /// an active wait otherwise -- and the CALLER waits for its answer at most `deadline`.
    ///
    /// A two-rank step must be able to say "the peer never pushed" instead of holding a card
    /// forever (`saber_exec::tp2::WAIT_DEADLINE`). Past the deadline this returns
    /// [`Error::Timeout`] and marks the handle wedged: the HIP thread is still blocked in the
    /// runtime, nothing queued behind it will run, and `Drop` detaches it. The closure owns only
    /// the raw event and a channel, so returning early leaves no borrow behind; the event's own
    /// destroy message queues behind the stuck wait and never runs before it.
    ///
    /// # Errors
    /// The deadline passing, or the wait failing.
    pub fn sync_event_within(&self, event: &Event, what: &str, deadline: Duration) -> Result<()> {
        let e = event.raw;
        let (done_tx, done_rx) = mpsc::sync_channel::<Result<()>>(1);
        let job: Task = Box::new(move |lib, ctx| {
            // SAFETY: the event outlives the wait: its destroy message queues behind this job.
            let r = unsafe { lib.event_synchronize(ctx, e.as_ptr()) };
            let _ = done_tx.send(r);
        });
        self.tx
            .send(Msg::Call(job, Deps::All))
            .map_err(|_| Error::ThreadGone)?;
        match done_rx.recv_timeout(deadline) {
            Ok(r) => r,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.wedged.store(true, Ordering::SeqCst);
                Err(Error::Timeout {
                    what: what.to_owned(),
                    seconds: deadline.as_secs(),
                })
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(Error::ThreadGone),
        }
    }

    /// `hipSetDeviceFlags` with a schedule (FN57): how the runtime's own host waits wait on the
    /// CURRENT device. Set it right after `set_device`, before the first stream, so every signal
    /// the runtime creates is waited on the same way.
    ///
    /// # Errors
    /// The runtime refusing the flags.
    pub fn set_device_schedule(&self, schedule: DeviceSchedule) -> Result<()> {
        let f = schedule as u32;
        self.run(move |lib, ctx| lib.set_device_flags(ctx, f))?;
        self.schedule.store(f, Ordering::SeqCst);
        Ok(())
    }

    /// The schedule this handle SET with [`HipThread::set_device_schedule`] (`None` = never set:
    /// the runtime's default). A sleeper reads it to choose between polling and one bounded
    /// runtime wait (`saber_exec::tp2::wait_aimed`).
    #[must_use]
    pub fn schedule_set(&self) -> Option<DeviceSchedule> {
        DeviceSchedule::from_flags(self.schedule.load(Ordering::SeqCst))
            .filter(|_| self.schedule.load(Ordering::SeqCst) != u32::MAX)
    }

    /// `hipGetDeviceFlags`' schedule bits for the current device (`None` = bits this crate does
    /// not name).
    ///
    /// # Errors
    /// The runtime call failing.
    pub fn device_schedule(&self) -> Result<Option<DeviceSchedule>> {
        let f = self.run(|lib, ctx| lib.device_flags(ctx))?;
        Ok(DeviceSchedule::from_flags(f))
    }

    /// The HIP thread's kernel thread id (FN57's CPU census names it beside the runtime's own
    /// threads, which inherit its name).
    #[must_use]
    pub fn tid(&self) -> i32 {
        self.tid
    }

    /// Pin the HIP thread to `cpus` (FN57). Threads the runtime creates LATER from it inherit the
    /// mask (the kernel copies it at clone); the ones it already made keep theirs.
    ///
    /// # Errors
    /// An empty set, a CPU past the mask, or the kernel refusing.
    pub fn pin(&self, cpus: &[usize]) -> Result<()> {
        let cpus = cpus.to_vec();
        self.run(move |_lib, _ctx| crate::affinity::pin_current_thread(&cpus))
    }

    /// `hipStreamWaitEvent`: `stream` waits for `event`; the host does not block.
    ///
    /// The device-side dependency between the upload stream and a consumer stream. Waiting on the
    /// host and re-submitting would serialise the two, which is exactly what the cold-start path
    /// must not do.
    pub fn stream_wait_event(&self, stream: &Stream, event: &Event) -> Result<()> {
        self.run_batch(&[Job::StreamWaitEvent { stream, event }])
            .map(|_| ())
    }

    /// `hipDeviceSynchronize`: block until every stream on the current device is idle.
    pub fn device_synchronize(&self) -> Result<()> {
        self.run(|lib, ctx| lib.device_synchronize(ctx))
    }

    /// `hipStreamQuery`: has everything on this stream finished?
    ///
    /// The non-blocking counterpart to [`HipThread::sync_stream`], and the only way a caller can
    /// put a DEADLINE on device work. That matters for exactly one route today: a one-shot
    /// all-reduce spins until its peer has pushed, so a protocol bug between the two ranks does
    /// not produce a wrong answer, it produces a kernel that never retires. A harness that called
    /// `sync_stream` would hang with it; one that polls can give up, say so, and tear the context
    /// down.
    pub fn stream_ready(&self, stream: &Stream) -> Result<bool> {
        let s = stream.raw;
        // SAFETY: the stream is alive for the duration of the call.
        self.run_with(Deps::on1(Name::Stream(s)), move |lib, ctx| unsafe {
            lib.stream_query(ctx, s.as_ptr())
        })
    }

    /// `hipEventCreateWithFlags`.
    pub fn create_event(&self, flags: EventFlags) -> Result<Event> {
        let raw = self.run(move |lib, ctx| lib.event_create(ctx, flags).map(RawEvent::from_raw))?;
        Ok(Event {
            raw,
            tx: self.sender(),
            _not_send: PhantomData,
        })
    }

    /// `hipEventRecord`.
    pub fn record_event(&self, event: &Event, stream: &Stream) -> Result<()> {
        let (e, s) = (event.raw, stream.raw);
        // SAFETY: both handles are alive for the duration of the call.
        self.run_with(
            Deps::on2(Name::Event(e), Name::Stream(s)),
            move |lib, ctx| unsafe { lib.event_record(ctx, e.as_ptr(), s.as_ptr()) },
        )
    }

    /// `hipEventSynchronize`.
    pub fn sync_event(&self, event: &Event) -> Result<()> {
        let e = event.raw;
        // SAFETY: the event is alive for the duration of the blocking call.
        self.run(move |lib, ctx| unsafe { lib.event_synchronize(ctx, e.as_ptr()) })
    }

    /// `hipEventQuery`: has the work before this event completed?
    ///
    /// The non-blocking form, and the one a pipeline wants: [`HipThread::sync_event`] runs the wait
    /// *on the HIP thread*, so while it spins nothing else can be submitted — a consumer that
    /// blocks on every copy serialises its own next copy behind it.
    pub fn query_event(&self, event: &Event) -> Result<bool> {
        let e = event.raw;
        // SAFETY: the event is alive for the duration of the call.
        self.run_with(Deps::on1(Name::Event(e)), move |lib, ctx| unsafe {
            lib.event_query(ctx, e.as_ptr())
        })
    }

    /// `hipEventElapsedTime`, in milliseconds.
    pub fn elapsed_ms(&self, start: &Event, end: &Event) -> Result<f32> {
        let (a, b) = (start.raw, end.raw);
        // SAFETY: both events are alive and were created with timing enabled by default.
        self.run_with(
            Deps::on2(Name::Event(a), Name::Event(b)),
            move |lib, ctx| unsafe { lib.event_elapsed_ms(ctx, a.as_ptr(), b.as_ptr()) },
        )
    }

    // ---- modules and launches -------------------------------------------------------------------- //

    /// `hipModuleLoadData` over a code-object image.
    ///
    /// The image is borrowed for the call only; HIP copies what it needs.
    pub fn load_module(&self, image: &[u8]) -> Result<Module> {
        let raw = self.run(move |lib, ctx| {
            // SAFETY: `image` is a live, complete byte image for the duration of the blocking job.
            unsafe { lib.module_load_data(ctx, image) }.map(RawModule::from_raw)
        })?;
        Ok(Module {
            raw,
            tx: self.sender(),
            _not_send: PhantomData,
        })
    }

    /// `hipModuleGetFunction`. The returned handle borrows the module.
    pub fn get_function<'m>(&self, module: &'m Module, symbol: &str) -> Result<Function<'m>> {
        let name = CString::new(symbol)
            .map_err(|_| Error::Invalid(format!("kernel symbol '{symbol}' contains a NUL")))?;
        let m = module.raw;
        let raw = self.run(move |lib, ctx| {
            // SAFETY: the module is alive (borrowed by the caller) for the duration of the call.
            unsafe { lib.module_get_function(ctx, m.as_ptr(), &name) }.map(RawFunction::from_raw)
        })?;
        Ok(Function {
            raw,
            _module: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// `hipModuleLaunchKernel` with a pre-packed kernarg image.
    ///
    /// `grid` is in **blocks**, not threads. `kernargs` must be exactly the image the code object's
    /// `saber_kernarg_desc` describes -- a layout the host and the kernel disagree about is memory
    /// corruption, not an error, which is why the descriptor is hash-checked against the code
    /// object's own metadata at load.
    pub fn launch(
        &self,
        function: &Function<'_>,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_lds: u32,
        stream: &Stream,
        kernargs: &mut [u8],
    ) -> Result<()> {
        let (f, st) = (function.raw, stream.raw);
        self.run_with(Deps::on1(Name::Stream(st)), move |lib, ctx| {
            // SAFETY: the function, stream and kernarg image are all borrowed by the caller's
            // frame, which blocks until this job returns. Whether the *contents* of the image
            // match the kernel's expectations is the loader's contract, checked at load.
            unsafe {
                lib.module_launch_kernel(
                    ctx,
                    f.as_ptr(),
                    grid,
                    block,
                    dynamic_lds,
                    st.as_ptr(),
                    kernargs,
                )
            }
        })
    }

    // ---- capture and graphs ---------------------------------------------------------------------- //

    /// `hipStreamBeginCapture`.
    ///
    /// [`StreamCaptureMode::ThreadLocal`] is the right mode here and the reason this whole module
    /// exists: with one owning thread it serialises against nothing else in the process.
    pub fn begin_capture(&self, stream: &Stream, mode: StreamCaptureMode) -> Result<()> {
        let s = stream.raw;
        // SAFETY: the stream is alive for the duration of the call.
        self.run_with(Deps::BeginCapture, move |lib, ctx| unsafe {
            lib.stream_begin_capture(ctx, s.as_ptr(), mode)
        })
    }

    /// `hipStreamEndCapture`.
    pub fn end_capture(&self, stream: &Stream) -> Result<Graph> {
        let s = stream.raw;
        let raw = self.run_with(Deps::EndCapture, move |lib, ctx| {
            // SAFETY: the stream is alive and capturing.
            unsafe { lib.stream_end_capture(ctx, s.as_ptr()) }.map(RawGraph::from_raw)
        })?;
        Ok(Graph {
            raw,
            tx: self.sender(),
            _not_send: PhantomData,
        })
    }

    /// How many nodes a captured graph holds.
    ///
    /// The launch budget, measured. A capture that produced a different number of nodes than the
    /// plan declared is a plan that is wrong about what it costs, and rule 7 says that is an error
    /// at capture rather than a surprise in a launch census.
    pub fn graph_node_count(&self, graph: &Graph) -> Result<usize> {
        let g = graph.raw;
        // SAFETY: the graph is alive for the duration of the call.
        self.run(move |lib, ctx| unsafe { lib.graph_get_node_count(ctx, g.as_ptr()) })
    }

    /// HIP node kinds, used to audit external-library graph synchronization nodes.
    pub fn graph_node_types(&self, graph: &Graph) -> Result<Vec<i32>> {
        let g = graph.raw;
        // SAFETY: the borrowed graph stays alive and is not mutated during enumeration.
        self.run(move |lib, ctx| unsafe { lib.graph_node_types(ctx, g.as_ptr()) })
    }

    /// `hipGraphInstantiate`.
    pub fn instantiate(&self, graph: &Graph) -> Result<GraphExec> {
        let g = graph.raw;
        let raw = self.run(move |lib, ctx| {
            // SAFETY: the graph is alive for the duration of the call.
            unsafe { lib.graph_instantiate(ctx, g.as_ptr()) }.map(RawGraphExec::from_raw)
        })?;
        Ok(GraphExec {
            raw,
            tx: self.sender(),
            _not_send: PhantomData,
        })
    }

    /// `hipGraphLaunch`.
    pub fn graph_launch(&self, exec: &GraphExec, stream: &Stream) -> Result<()> {
        let (e, s) = (exec.raw, stream.raw);
        // SAFETY: both handles are alive for the duration of the call.
        self.run_with(Deps::on1(Name::Stream(s)), move |lib, ctx| unsafe {
            lib.graph_launch(ctx, e.as_ptr(), s.as_ptr())
        })
    }

    // ---- IPC ------------------------------------------------------------------------------------- //

    /// `hipIpcGetMemHandle`: export an allocation to the peer rank.
    pub fn ipc_export(&self, ptr: &DevicePtr) -> Result<IpcMemHandle> {
        let p = ptr.addr();
        // SAFETY: the allocation is alive and `p` is its base pointer.
        self.run(move |lib, ctx| unsafe { lib.ipc_get_mem_handle(ctx, p.as_ptr()) })
    }

    /// `hipIpcOpenMemHandle`: map a peer rank's allocation.
    ///
    /// Returns a bare [`DeviceAddr`], not a [`DevicePtr`]: this process does not own the memory and
    /// must not free it. `hipIpcCloseMemHandle` is not bound in v0 -- IPC scratch is startup-only
    /// and lives for the process, which is the only pattern the 2-rank all-reduce needs.
    pub fn ipc_import(&self, handle: IpcMemHandle) -> Result<DeviceAddr> {
        // SAFETY: the handle came from a peer's export of a live allocation; the caller owns that
        // contract, as it does in C.
        self.run(move |lib, ctx| unsafe {
            lib.ipc_open_mem_handle(ctx, handle)
                .map(DeviceAddr::from_raw)
        })
    }

    /// `hipIpcOpenMemHandle`, as an OWNING handle that unmaps itself.
    ///
    /// Prefer this to [`HipThread::ipc_import`]: the raw form leaves the unmap to the caller, and
    /// an unmap nobody performs is a mapping that outlives the peer's allocation.
    pub fn ipc_open(&self, handle: IpcMemHandle) -> Result<ImportedPtr> {
        let addr = self.ipc_import(handle)?;
        Ok(ImportedPtr::new(addr, self.sender()))
    }

    /// `hipIpcCloseMemHandle`: unmap an address obtained from [`HipThread::ipc_import`].
    ///
    /// The import side hands out a bare [`DeviceAddr`] precisely because this process does not own
    /// the memory, so closing it cannot be a `Drop` — it is an explicit call, and calling it while
    /// a kernel still dereferences the address is the caller's contract, as it is in C.
    pub fn ipc_close(&self, addr: DeviceAddr) -> Result<()> {
        // SAFETY: `addr` must have come from `ipc_open_mem_handle` on this runtime; that is the
        // documented contract above, and this crate never manufactures such an address elsewhere.
        self.run(move |lib, ctx| unsafe { lib.ipc_close_mem_handle(ctx, addr.as_ptr()) })
    }
}

impl std::fmt::Debug for HipThread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipThread({:?})", self.info)
    }
}

impl HipThread {
    fn rccl_runtime(&self, path: &str) -> Result<(Arc<RcclLib>, std::sync::Weak<()>)> {
        let mut slot = self
            .rccl
            .lock()
            .map_err(|_| Error::Invalid("RCCL registry poisoned".into()))?;
        if slot.is_none() {
            let lib = self.run(|_, _| RcclLib::load(path))?;
            *slot = Some(RcclState {
                path: path.to_owned(),
                lib,
                owner: Arc::new(()),
                comms: Vec::new(),
            });
        }
        let state = slot.as_ref().expect("runtime initialized above");
        if state.path != path {
            return Err(Error::Invalid(
                "One HIP thread cannot mix RCCL libraries".into(),
            ));
        }
        Ok((Arc::clone(&state.lib), Arc::downgrade(&state.owner)))
    }

    /// Create a bootstrap ID, retaining its library on the owning HIP thread.
    pub fn rccl_unique_id(&self, path: &str) -> Result<RcclUniqueId> {
        let (lib, _) = self.rccl_runtime(path)?;
        self.run(move |_, ctx| lib.unique_id(ctx))
    }

    /// Query the selected RCCL library version on the HIP thread.
    pub fn rccl_library_version(&self, path: &str) -> Result<i32> {
        let (lib, _) = self.rccl_runtime(path)?;
        self.run(move |_, ctx| lib.version(ctx))
    }

    /// Initialize a communicator owned until this HIP thread shuts down.
    pub fn rccl_init_rank(
        &self,
        path: &str,
        world: i32,
        rank: i32,
        id: &RcclUniqueId,
    ) -> Result<RcclComm> {
        let (lib, owner) = self.rccl_runtime(path)?;
        let mut slot = self
            .rccl
            .lock()
            .map_err(|_| Error::Invalid("RCCL registry poisoned".into()))?;
        let raw = self.run(|_, ctx| lib.init_rank(ctx, world, rank, id))?;
        slot.as_mut()
            .expect("runtime initialized above")
            .comms
            .push(raw);
        Ok(RcclComm {
            raw,
            lib,
            owner,
            not_send: PhantomData,
        })
    }

    fn check_rccl_owner(&self, comm: &RcclComm) -> Result<()> {
        let slot = self
            .rccl
            .lock()
            .map_err(|_| Error::Invalid("RCCL registry poisoned".into()))?;
        let valid = slot.as_ref().is_some_and(|state| {
            comm.owner
                .upgrade()
                .is_some_and(|owner| Arc::ptr_eq(&owner, &state.owner))
                && state.comms.contains(&comm.raw)
        });
        if !valid {
            return Err(Error::Invalid(
                "RCCL communicator belongs to another or closed HIP thread".into(),
            ));
        }
        Ok(())
    }

    /// Enqueue an exact BF16 sum on a borrowed stream and device addresses.
    pub fn rccl_all_reduce_bf16(
        &self,
        comm: &RcclComm,
        src: DeviceAddr,
        dst: DeviceAddr,
        count: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.check_rccl_owner(comm)?;
        let lib = Arc::clone(&comm.lib);
        let (raw, s, d, st) = (
            comm.raw,
            src.as_u64() as usize,
            dst.as_u64() as usize,
            stream.raw.as_ptr() as usize,
        );
        self.run_with(Deps::on1(Name::Stream(stream.raw)), move |_, ctx| {
            lib.all_reduce_bf16(ctx, raw, s, d, count, st)
        })
    }

    /// Enqueue a byte all-gather on a borrowed stream and device addresses.
    pub fn rccl_all_gather_bytes(
        &self,
        comm: &RcclComm,
        src: DeviceAddr,
        dst: DeviceAddr,
        count: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.check_rccl_owner(comm)?;
        let lib = Arc::clone(&comm.lib);
        let (raw, s, d, st) = (
            comm.raw,
            src.as_u64() as usize,
            dst.as_u64() as usize,
            stream.raw.as_ptr() as usize,
        );
        self.run_with(Deps::on1(Name::Stream(stream.raw)), move |_, ctx| {
            lib.all_gather_bytes(ctx, raw, s, d, count, st)
        })
    }
}

impl Drop for HipThread {
    fn drop(&mut self) {
        if self.wedged.load(Ordering::SeqCst) {
            // FN57: the thread is inside a wait that passed its deadline; joining would hang
            tracing::error!("the HIP thread is wedged in a timed-out wait; detaching it");
            drop(self.join.take());
            return;
        }
        match self.rccl.get_mut() {
            Ok(slot) => {
                if let Some(state) = slot.take() {
                    let job = Msg::Call(
                        Box::new(move |_, ctx| {
                            for comm in state.comms {
                                if let Err(error) = state.lib.destroy(ctx, comm) {
                                    tracing::error!(%error, "RCCL communicator destruction failed");
                                }
                            }
                        }),
                        Deps::All,
                    );
                    if self.tx.send(job).is_err() {
                        tracing::warn!("HIP thread exited before RCCL teardown");
                    }
                }
            }
            Err(_) => tracing::error!("RCCL registry poisoned during shutdown"),
        }
        if self.tx.send(Msg::Shutdown).is_err() {
            tracing::warn!("HIP thread had already exited when HipThread was dropped");
        }
        if let Some(join) = self.join.take() {
            if join.join().is_err() {
                tracing::error!("the HIP thread panicked");
            }
        }
    }
}

/// FN72: the CU mask with every one of `cu_count` CUs set (words of 32, low bit = CU 0).
pub fn full_cu_mask(cu_count: u32) -> Vec<u32> {
    (0..cu_count.div_ceil(32))
        .map(|w| {
            let left = cu_count - w * 32;
            if left >= 32 {
                u32::MAX
            } else {
                (1u32 << left) - 1
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_full_cu_mask_sets_exactly_the_device_cus() {
        assert_eq!(full_cu_mask(64), vec![u32::MAX, u32::MAX]);
        assert_eq!(full_cu_mask(1), vec![1]);
        assert_eq!(full_cu_mask(40), vec![u32::MAX, 0xff]);
        assert_eq!(full_cu_mask(32), vec![u32::MAX]);
        // the control: a mask one CU short differs
        assert_ne!(full_cu_mask(63), full_cu_mask(64));
        for n in 1..=128u32 {
            let m = full_cu_mask(n);
            assert_eq!(m.iter().map(|w| w.count_ones()).sum::<u32>(), n);
        }
    }

    /// Lowering is where every bounds check in the batch path lives, and it runs with no runtime
    /// loaded — so it is testable on this host, which has no ROCm. The handles below are built from
    /// fake tokens; nothing is dereferenced, because `lower` only reads lengths and addresses.
    #[test]
    fn a_batch_is_bounds_checked_before_anything_is_posted() {
        let (tx, _rx) = mpsc::channel::<Msg>();
        let dst = DevicePtr::new(DeviceAddr::NULL.offset(0x1000).unwrap(), 4096, tx.clone());
        let src_alloc = DevicePtr::new(DeviceAddr::NULL.offset(0x9000).unwrap(), 4096, tx.clone());
        let stream = Stream {
            raw: RawStream::from_raw(0x11 as *mut c_void),
            tx: tx.clone(),
            _not_send: PhantomData,
        };
        let mut storage = vec![0u8; 2048];
        let len = storage.len();
        let mut pinned = PinnedBuf::new(storage.as_mut_ptr().cast(), len, tx.clone());

        // A copy that ends exactly at the allocation's end is legal.
        let head = pinned.slice(0, 2048).expect("view");
        RawJob::lower(&Job::MemcpyH2d {
            dst: &dst,
            dst_offset: 2048,
            src: head,
            stream: &stream,
        })
        .expect("2048 bytes at offset 2048 of a 4096-byte allocation fits");

        // One byte further does not, and it is refused here rather than by the GPU three kernels
        // later with hipErrorIllegalAddress.
        let err = RawJob::lower(&Job::MemcpyH2d {
            dst: &dst,
            dst_offset: 2049,
            src: head,
            stream: &stream,
        })
        .expect_err("must not fit");
        assert!(matches!(err, Error::Invalid(_)), "{err}");

        let err = RawJob::lower(&Job::MemcpyH2d {
            dst: &dst,
            dst_offset: u64::MAX,
            src: head,
            stream: &stream,
        })
        .expect_err("offset overflow must not wrap into another allocation");
        assert!(matches!(err, Error::Invalid(_)), "{err}");

        // Peer copies are checked against both sides.
        RawJob::lower(&Job::MemcpyPeer {
            dst: &dst,
            dst_device: 0,
            src: &src_alloc,
            src_device: 1,
            bytes: 4096,
            stream: &stream,
        })
        .expect("4096 bytes between two 4096-byte allocations");
        assert!(RawJob::lower(&Job::MemcpyPeer {
            dst: &dst,
            dst_device: 0,
            src: &src_alloc,
            src_device: 1,
            bytes: 4097,
            stream: &stream,
        })
        .is_err());

        // ---- D2D (V4): the fork's copy, and the three things it refuses ----------------------
        // A copy between two allocations, in range on both sides, is lowered.
        RawJob::lower(&Job::MemcpyD2d {
            dst: &dst,
            dst_offset: 1024,
            src: &src_alloc,
            src_offset: 2048,
            bytes: 2048,
            stream: &stream,
        })
        .expect("2048 bytes at 1024 of a 4096-byte destination, from 2048 of a 4096-byte source");

        // Out of range on the DESTINATION, and out of range on the SOURCE, are two refusals.
        let err = RawJob::lower(&Job::MemcpyD2d {
            dst: &dst,
            dst_offset: 2049,
            src: &src_alloc,
            src_offset: 0,
            bytes: 2048,
            stream: &stream,
        })
        .expect_err("a destination that overruns must not be posted");
        assert!(matches!(err, Error::Invalid(_)), "{err}");
        let err = RawJob::lower(&Job::MemcpyD2d {
            dst: &dst,
            dst_offset: 0,
            src: &src_alloc,
            src_offset: 2049,
            bytes: 2048,
            stream: &stream,
        })
        .expect_err("a source that overruns must not be posted");
        assert!(matches!(err, Error::Invalid(_)), "{err}");

        // TWO OFFSETS OF ONE ALLOCATION is the ordinary case -- the KV pool is one buffer and a
        // page copy is two offsets into it -- so it is legal while the ranges are disjoint...
        RawJob::lower(&Job::MemcpyD2d {
            dst: &dst,
            dst_offset: 2048,
            src: &dst,
            src_offset: 0,
            bytes: 2048,
            stream: &stream,
        })
        .expect("two disjoint halves of one allocation");
        // ...and refused when they overlap, which `hipMemcpyAsync` does not define.
        let err = RawJob::lower(&Job::MemcpyD2d {
            dst: &dst,
            dst_offset: 1024,
            src: &dst,
            src_offset: 0,
            bytes: 2048,
            stream: &stream,
        })
        .expect_err("an overlapping copy inside one allocation must be refused");
        assert!(matches!(err, Error::Invalid(_)), "{err}");

        // The staging view itself refuses to overrun its buffer.
        assert!(pinned.slice(1, 2048).is_none());
        assert!(pinned.slice_mut(2047, 1).is_some());
        assert!(pinned.slice_mut(2048, 1).is_none());
    }

    #[test]
    fn spawning_without_rocm_reports_not_available() {
        // This is the host path: there is no ROCm outside the containers, so `spawn` must fail
        // cleanly instead of aborting the process at load time.
        let err = HipThread::spawn("libsaber-no-such-runtime.so.999").unwrap_err();
        assert!(err.is_not_available(), "{err}");
    }

    #[test]
    fn the_context_token_is_not_send() {
        struct IsSend<T>(PhantomData<T>);
        trait Fallback {
            const SEND: bool = false;
        }
        impl<T> Fallback for IsSend<T> {}
        impl<T: Send> IsSend<T> {
            #[allow(dead_code)]
            const SEND: bool = true;
        }
        const {
            assert!(!<IsSend<Ctx>>::SEND, "Ctx must not be Send");
            assert!(<IsSend<RuntimeInfo>>::SEND, "negative control");
        }
    }

    #[test]
    fn hip_thread_itself_is_send_so_it_can_be_handed_to_the_step_loop() {
        fn assert_send<T: Send>() {}
        assert_send::<HipThread>();
    }
}

#[cfg(test)]
mod memcpy_direction_tests {
    use super::{memcpy_args, MemcpyKind};

    const DEVICE: usize = 0x7f00_0000_0000;
    const HOST: usize = 0x0000_5555_0000;

    /// **`hipMemcpyAsync` COPIES `src -> dst` AND THE KIND DOES NOT REORDER THEM.** A
    /// device-to-host copy takes the HOST pointer first; a host-to-device copy takes the DEVICE
    /// pointer first. The batch path put the device address first for BOTH until ws-h31, so every
    /// batched `Job::MemcpyD2h` ran backwards — page-locked host memory written INTO the device
    /// range it was meant to read, reported as success.
    ///
    /// Its two callers were the engine's batched logits gather (every `bits` read back as a fresh
    /// page's zero, which `tp2::combine` resolves as a tie to the lower GLOBAL id — always rank
    /// 0's column, right for every token in rank 0's half of the vocabulary and wrong for the
    /// rest) and the ASYNC session park.
    #[test]
    fn a_device_to_host_copy_takes_the_host_pointer_first() {
        assert_eq!(
            memcpy_args(MemcpyKind::DeviceToHost, DEVICE, HOST),
            (HOST, DEVICE),
            "D2H must hand hipMemcpyAsync the host pointer as `dst`"
        );
        assert_eq!(
            memcpy_args(MemcpyKind::HostToDevice, DEVICE, HOST),
            (DEVICE, HOST),
            "H2D must hand hipMemcpyAsync the device pointer as `dst`"
        );
    }

    /// **THE NEGATIVE CONTROL, and it is the bug itself**: the two directions must not resolve to
    /// the same argument order. A lowering that returned `(device, host)` for both would pass the
    /// first half of the test above and is exactly what shipped, so the law is stated as the
    /// DIFFERENCE between the directions rather than as one of them.
    #[test]
    fn the_two_directions_are_not_the_same_argument_order() {
        let h2d = memcpy_args(MemcpyKind::HostToDevice, DEVICE, HOST);
        let d2h = memcpy_args(MemcpyKind::DeviceToHost, DEVICE, HOST);
        assert_ne!(
            h2d, d2h,
            "a batched D2H that takes the same argument order as an H2D runs backwards"
        );
        assert_eq!(
            d2h,
            (h2d.1, h2d.0),
            "and it is exactly the swap, not some third thing"
        );
    }
}
