//! saber-hip — the engine's entire binding to the HIP runtime.
//!
//! **Nothing here is linked at build time.** `libamdhip64.so.7` is `dlopen`ed at run time, so the
//! Rust binary carries zero ROCm: it builds on this host, which has no `/opt/rocm` at all, and it
//! runs against whichever ROCm userspace is mounted into the container. That is not an
//! inconvenience worked around — it is the A/B axis the current stack cannot offer, because torch's
//! ROCm build pins the whole tree. One binary, three userspaces, same paired benchmark.
//!
//! # The three rules this crate exists to enforce
//!
//! 1. **One thread.** [`HipThread`] owns the library and runs every HIP call on a single
//!    `std::thread`; work arrives as closures over a channel and each one is handed a `!Send`
//!    [`Ctx`] token it cannot manufacture. Then `hipStreamCaptureModeThreadLocal` serialises
//!    against nothing else in the process. With the example-default `Global` mode, a `/metrics`
//!    handler calling `hipMemGetInfo` on a tokio worker fails intermittently, under load, only
//!    while a bucket is capturing — which is a bug report nobody ever reproduces.
//!
//! 2. **Handles are `!Send`, and device pointers cannot be forged.** [`Stream`], [`Event`],
//!    [`Module`], [`Function`], [`Graph`], [`GraphExec`], [`DevicePtr`] and [`PinnedBuf`] all carry
//!    `PhantomData<*mut ()>`; [`DevicePtr`] has no constructor reachable from outside this crate
//!    and no conversion from `&[u8]`. Addresses get baked into graph nodes, and Rust's drop order
//!    will happily free one under a captured graph if nothing stops it.
//!
//! 3. **Unknown status codes log and return `Err`.** [`HipError`] is `#[repr(i32)]`, carries an
//!    [`HipError::Unknown`] catch-all, and is `#[non_exhaustive]` so no downstream `match` can be
//!    exhaustive. This box runs a 7.15.0-alpha userspace in some images; a status code this binary
//!    predates must not panic the step loop.
//!
//! # What is here
//!
//! ~40 hand-written `extern "C"` declarations — devices, memory, streams, events, modules and
//! launches, capture and graphs, peer access and IPC — and nothing else. `bindgen` over
//! `hip_runtime_api.h` is ruled out by the same decision that removed ROCm from the build: the
//! header is not on the host, it is 476 KB of C++, and every type this crate needs is `void *` or
//! `int`.
//!
//! # Example
//!
//! ```no_run
//! use saber_hip::{HipThread, Topology, DEFAULT_HIP_SONAME};
//!
//! // Fails with Error::NotAvailable on a host without ROCm; that is the normal outcome here.
//! let hip = HipThread::spawn(DEFAULT_HIP_SONAME)?;
//! let topo = Topology::probe(&hip, /* with_free_mem = */ true)?;
//! for d in &topo.devices {
//!     println!("{} {} {} CUs", d.index, d.arch, d.cu_count);
//! }
//! # Ok::<(), saber_hip::Error>(())
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod affinity;
mod device;
mod error;
mod ffi;
mod handles;
mod lane;
pub mod pinned;
pub mod process;
mod rccl;
pub mod shmline;
mod thread;

pub use device::{ArchSource, DeviceAttribute, DeviceInfo, HipVersion, Topology};
pub use error::{Error, HipError, Result};
pub use ffi::{
    DeviceSchedule, EventFlags, IpcMemHandle, MallocFlags, MemcpyKind, StreamCaptureMode,
    StreamFlags, DEFAULT_HIP_SONAME, IPC_HANDLE_BYTES,
};
pub use handles::RcclComm;
pub use handles::{
    DeviceAddr, DevicePtr, Event, Function, Graph, GraphExec, ImportedPtr, Module, PinnedBuf,
    PinnedChunk, PinnedSlice, PinnedSliceMut, Stream, StreamId,
};
pub use lane::BatchTicket;
pub use pinned::PinnedPages;
pub use rccl::RcclUniqueId;
pub use thread::{full_cu_mask, Ctx, HipThread, Job, RuntimeInfo};

#[cfg(test)]
mod tests {
    use super::*;

    /// The acceptance condition for the host build: the crate is fully usable, and *fails cleanly*,
    /// on a machine with no ROCm anywhere. Every other test in this crate runs under that
    /// condition too — which is why none of them touch a GPU.
    #[test]
    fn the_host_has_no_rocm_and_that_is_an_error_not_a_crash() {
        match HipThread::spawn(DEFAULT_HIP_SONAME) {
            Err(e) if e.is_not_available() => {
                // Expected on the host.
            }
            Err(e) => panic!("expected NotAvailable on a host without ROCm, got: {e}"),
            Ok(hip) => {
                // Inside the dev container this branch is taken, and then the runtime must be
                // coherent enough to answer the two questions the probe starts with.
                let n = hip.device_count().expect("hipGetDeviceCount");
                assert!(n >= 0);
                assert!(hip.info().runtime_version > 0);
            }
        }
    }
}
