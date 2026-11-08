//! `!Send` owning handles for HIP resources, and the `Send` raw tokens that cross into the
//! [`crate::HipThread`].
//!
//! Two properties are structural here, and both come out of the Rust-specific hazards the
//! architecture calls out:
//!
//! 1. **Every owning handle is `!Send`.** `hipStream_t` is `*mut c_void`; left as one it gets
//!    `Send`ed onto a tokio worker the first time someone is in a hurry, and then a `/metrics`
//!    handler is calling into HIP from the wrong thread while a bucket is capturing. Each handle
//!    carries a `PhantomData<*mut ()>` so the compiler refuses that, and the *raw* token that
//!    actually crosses the channel is a plain `usize` newtype.
//!
//! 2. **A device-visible pointer cannot be built from a `&[u8]`.** [`DevicePtr`] has no public
//!    constructor and no conversion from host memory: the only way to get one is
//!    [`crate::HipThread::malloc`]. Addresses get baked into graph nodes, and a host pointer that
//!    reached a kernarg would be a fault at replay time rather than a type error at build time.
//!
//! Drop on every handle posts its destructor back to the `HipThread` rather than calling HIP from
//! whatever thread dropped it. If that thread is gone, the failure is *logged* and the resource is
//! leaked -- a panic in `Drop` during shutdown is how a process loses its error message.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::mpsc::Sender;

use crate::error::Result;
use crate::ffi::Lib;
use crate::thread::{Ctx, Msg};

/// Borrowed communicator token; the HIP thread owns the native RCCL resource.
///
/// Communicators remain alive until thread shutdown, after queued graph teardown.
pub struct RcclComm {
    pub(crate) raw: usize,
    pub(crate) lib: std::sync::Arc<crate::rccl::RcclLib>,
    pub(crate) owner: std::sync::Weak<()>,
    pub(crate) not_send: PhantomData<*mut ()>,
}

impl std::fmt::Debug for RcclComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RcclComm(borrowed from HipThread)")
    }
}

/// A device address.
///
/// `Copy`, `Send` and plain: this is what goes into a kernarg image and what crosses the channel to
/// the HIP thread. It carries no ownership -- [`DevicePtr`] does -- and it is deliberately *not*
/// constructible from a host pointer or a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceAddr(u64);

impl DeviceAddr {
    /// The address as an integer, for packing into a kernarg image.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// A null device address. Some kernels take an optional buffer; this is how a plan expresses
    /// "not provided" without an `Option` in the kernarg layout.
    pub const NULL: DeviceAddr = DeviceAddr(0);

    /// Offset this address by `bytes`. Returns `None` on overflow rather than wrapping into
    /// another allocation.
    #[must_use]
    pub fn offset(self, bytes: u64) -> Option<DeviceAddr> {
        self.0.checked_add(bytes).map(DeviceAddr)
    }

    pub(crate) fn from_raw(p: *mut c_void) -> DeviceAddr {
        DeviceAddr(p as u64)
    }

    pub(crate) fn as_ptr(self) -> *mut c_void {
        self.0 as *mut c_void
    }
}

macro_rules! raw_token {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub(crate) struct $name(usize);

        impl $name {
            pub(crate) fn from_raw(p: *mut c_void) -> Self {
                $name(p as usize)
            }
            pub(crate) fn as_ptr(self) -> *mut c_void {
                self.0 as *mut c_void
            }
        }
    };
}

raw_token!(
    /// `hipStream_t` as a `Send` integer.
    RawStream
);
raw_token!(
    /// `hipEvent_t` as a `Send` integer.
    RawEvent
);
raw_token!(
    /// `hipModule_t` as a `Send` integer.
    RawModule
);
raw_token!(
    /// `hipFunction_t` as a `Send` integer.
    RawFunction
);
raw_token!(
    /// `hipGraph_t` as a `Send` integer.
    RawGraph
);
raw_token!(
    /// `hipGraphExec_t` as a `Send` integer.
    RawGraphExec
);

/// Post a destructor to the HIP thread. Never panics: a failed teardown is logged, not raised.
fn post_destructor<F>(tx: &Sender<Msg>, resource: &'static str, f: F)
where
    F: FnOnce(&Lib, &Ctx) -> Result<()> + Send + 'static,
{
    let job = Msg::Call(
        Box::new(move |lib, ctx| {
            if let Err(e) = f(lib, ctx) {
                tracing::error!(resource, error = %e, "HIP resource destruction failed");
            }
        }),
        crate::lane::Deps::All,
    );
    if tx.send(job).is_err() {
        tracing::warn!(
            resource,
            "HipThread is gone; leaking this handle rather than calling HIP off-thread"
        );
    }
}

macro_rules! owning_handle {
    (
        $(#[$m:meta])*
        $name:ident, $raw:ident, $resource:literal, $destroy:expr
    ) => {
        $(#[$m])*
        pub struct $name {
            pub(crate) raw: $raw,
            pub(crate) tx: Sender<Msg>,
            /// Makes the handle `!Send`: every HIP call belongs to one thread.
            pub(crate) _not_send: PhantomData<*mut ()>,
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), self.raw)
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                let raw = self.raw;
                #[allow(clippy::redundant_closure_call)]
                post_destructor(&self.tx, $resource, move |lib, ctx| ($destroy)(lib, ctx, raw));
            }
        }
    };
}

owning_handle!(
    /// A HIP stream. Created non-blocking by default; dropped by posting `hipStreamDestroy` back
    /// to the HIP thread.
    Stream,
    RawStream,
    "hipStream_t",
    |lib: &Lib, ctx: &Ctx, raw: RawStream| unsafe { lib.stream_destroy(ctx, raw.as_ptr()) }
);

/// Identity of a [`Stream`], without the stream.
///
/// `Copy + Send + Eq + Hash` where [`Stream`] is none of those, so a per-stream table (workspace
/// arenas, a staging ring, a launch census) can be keyed on *which* stream without a handle
/// escaping the HIP thread. It names a stream; it cannot be used to call HIP, and it does not keep
/// the stream alive -- an id outliving its stream keys nothing, which is the intended failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(usize);

impl StreamId {
    /// The underlying `hipStream_t` as an integer, for a log line or a trace.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

impl Stream {
    /// This stream's identity.
    #[must_use]
    pub fn id(&self) -> StreamId {
        StreamId(self.raw.as_ptr() as usize)
    }
}

owning_handle!(
    /// A HIP event.
    Event,
    RawEvent,
    "hipEvent_t",
    |lib: &Lib, ctx: &Ctx, raw: RawEvent| unsafe { lib.event_destroy(ctx, raw.as_ptr()) }
);

owning_handle!(
    /// A loaded code object. Every [`Function`] borrowed from it is bounded by its lifetime, so a
    /// module cannot be unloaded while a function handle into it is still reachable.
    Module,
    RawModule,
    "hipModule_t",
    |lib: &Lib, ctx: &Ctx, raw: RawModule| unsafe { lib.module_unload(ctx, raw.as_ptr()) }
);

owning_handle!(
    /// A captured graph, before instantiation.
    Graph,
    RawGraph,
    "hipGraph_t",
    |lib: &Lib, ctx: &Ctx, raw: RawGraph| unsafe { lib.graph_destroy(ctx, raw.as_ptr()) }
);

owning_handle!(
    /// An instantiated, launchable graph.
    GraphExec,
    RawGraphExec,
    "hipGraphExec_t",
    |lib: &Lib, ctx: &Ctx, raw: RawGraphExec| unsafe { lib.graph_exec_destroy(ctx, raw.as_ptr()) }
);

/// A kernel entry point inside a [`Module`].
///
/// Borrowed, not owned: HIP has no "release function", and the handle is only valid while its
/// module is loaded. The lifetime says so.
pub struct Function<'m> {
    pub(crate) raw: RawFunction,
    pub(crate) _module: PhantomData<&'m Module>,
    pub(crate) _not_send: PhantomData<*mut ()>,
}

impl std::fmt::Debug for Function<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Function({:?})", self.raw)
    }
}

/// An owned device allocation.
///
/// There is deliberately **no** way to build one of these from a `&[u8]`, a host pointer or an
/// integer: `DevicePtr` exists only as the result of [`crate::HipThread::malloc`]. Under invariant
/// I1 these addresses are baked into graph nodes and must outlive every replay, which is why the
/// type is `!Send` (it belongs to the HIP thread's world) and why its `Drop` posts `hipFree` back
/// rather than calling it wherever the value happened to die.
pub struct DevicePtr {
    addr: DeviceAddr,
    len: u64,
    tx: Sender<Msg>,
    _not_send: PhantomData<*mut ()>,
}

impl DevicePtr {
    pub(crate) fn new(addr: DeviceAddr, len: u64, tx: Sender<Msg>) -> DevicePtr {
        DevicePtr {
            addr,
            len,
            tx,
            _not_send: PhantomData,
        }
    }

    /// The device address, for a kernarg image or a copy.
    #[must_use]
    pub fn addr(&self) -> DeviceAddr {
        self.addr
    }

    /// Length of the allocation in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the allocation is zero-length. (A zero-length `hipMalloc` is refused by
    /// [`crate::HipThread::malloc`], so this is always `false` in practice; clippy asks for it and
    /// it costs nothing.)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// An address `bytes` into this allocation, bounds-checked against its length.
    ///
    /// The check is the point: a sub-buffer that runs off the end of its allocation is the kind of
    /// fault that shows up as `hipErrorIllegalAddress` three kernels later.
    #[must_use]
    pub fn sub_addr(&self, bytes: u64) -> Option<DeviceAddr> {
        if bytes > self.len {
            return None;
        }
        self.addr.offset(bytes)
    }
}

impl std::fmt::Debug for DevicePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DevicePtr({:#x}, {} bytes)",
            self.addr.as_u64(),
            self.len
        )
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        let addr = self.addr;
        post_destructor(&self.tx, "hipMalloc", move |lib, ctx| unsafe {
            lib.free(ctx, addr.as_ptr())
        });
    }
}

/// A peer rank's allocation, mapped into this process by `hipIpcOpenMemHandle`.
///
/// Deliberately NOT a [`DevicePtr`]: this process did not allocate the memory and must never
/// `hipFree` it. What it owns is the *mapping*, and dropping it posts `hipIpcCloseMemHandle` back
/// to the HIP thread — the mirror of what `DevicePtr` does with `hipFree`.
///
/// The lifetime rule is the peer's, not ours: the address is valid only while the peer's
/// allocation lives. Under invariant I1 these addresses are baked into the peer's kernarg images
/// and into captured graph nodes, so in practice both sides hold theirs for the life of the
/// process and this `Drop` runs at teardown.
pub struct ImportedPtr {
    addr: DeviceAddr,
    tx: Sender<Msg>,
    _not_send: PhantomData<*mut ()>,
}

impl ImportedPtr {
    pub(crate) fn new(addr: DeviceAddr, tx: Sender<Msg>) -> ImportedPtr {
        ImportedPtr {
            addr,
            tx,
            _not_send: PhantomData,
        }
    }

    /// The mapped address, for a kernarg image.
    #[must_use]
    pub fn addr(&self) -> DeviceAddr {
        self.addr
    }
}

impl std::fmt::Debug for ImportedPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImportedPtr({:#x}, peer-owned)", self.addr.as_u64())
    }
}

impl Drop for ImportedPtr {
    fn drop(&mut self) {
        let addr = self.addr;
        post_destructor(&self.tx, "hipIpcOpenMemHandle", move |lib, ctx| unsafe {
            lib.ipc_close_mem_handle(ctx, addr.as_ptr())
        });
    }
}

/// An owned pinned (page-locked) host allocation, from `hipHostMalloc`.
///
/// Pinned staging is what makes an H2D copy a DMA rather than a bounce. The buffer is zeroed on
/// creation so that reading it is never a read of uninitialised memory.
pub struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
    tx: Sender<Msg>,
    /// FN106d: 0 = a `hipHostMalloc` allocation (freed with `hipHostFree`); otherwise the mapped
    /// length of a REGISTERED private mapping ([`crate::PinnedPages::Huge`]): dropped as
    /// `hipHostUnregister` FIRST, then `munmap` -- never `hipHostFree`.
    reg_len: usize,
    /// FN106d: counted in [`crate::pinned::stats`] (every runtime allocation; not a test seam's).
    counted: bool,
    _not_send: PhantomData<*mut ()>,
}

impl PinnedBuf {
    /// Wrap leaked host memory as a pinned buffer, for tests on a machine with no ROCm.
    ///
    /// This is the seam that lets `saber-mem`'s staging-ring tests run on this host, which has no
    /// `/opt/rocm` at all. The memory must be `'static` (leak it) because nothing frees it, and the
    /// buffer is **not** page-locked and **not** registered with any HIP runtime — an H2D out of it
    /// would be an ordinary pageable copy. It is behind the `test-util` feature so that a real
    /// build cannot reach it.
    #[cfg(feature = "test-util")]
    #[must_use]
    pub fn leaked_for_test(bytes: &'static mut [u8]) -> PinnedBuf {
        let (tx, rx) = std::sync::mpsc::channel();
        // Nothing will ever run the destructor job this buffer posts on drop; keeping the receiver
        // alive is what stops `post_destructor` from logging a warning about a thread that never
        // existed.
        std::mem::forget(rx);
        let len = bytes.len();
        PinnedBuf::new(bytes.as_mut_ptr().cast(), len, tx)
    }

    pub(crate) fn new(ptr: *mut c_void, len: usize, tx: Sender<Msg>) -> PinnedBuf {
        let ptr = ptr.cast::<u8>();
        // hipHostMalloc does not zero. Do it once here so that every later read of this buffer is
        // a read of initialised memory, which is what makes `as_slice` sound.
        // SAFETY: `ptr` is a live host allocation of `len` bytes returned by hipHostMalloc.
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
        PinnedBuf {
            ptr,
            len,
            tx,
            reg_len: 0,
            counted: false,
            _not_send: PhantomData,
        }
    }

    /// FN106d: a `hipHostMalloc` allocation counted in the census.
    pub(crate) fn new_counted(ptr: *mut c_void, len: usize, tx: Sender<Msg>) -> PinnedBuf {
        crate::pinned::stats().add_shmem(len as u64);
        let mut b = PinnedBuf::new(ptr, len, tx);
        b.counted = true;
        b
    }

    /// FN106d: a REGISTERED private mapping of `reg_len` bytes (the huge form), already zero (a
    /// fresh anonymous mapping), `len` of them the caller's.
    pub(crate) fn new_registered(
        ptr: usize,
        len: usize,
        reg_len: usize,
        tx: Sender<Msg>,
    ) -> PinnedBuf {
        PinnedBuf {
            ptr: ptr as *mut u8,
            len,
            tx,
            reg_len,
            counted: true,
            _not_send: PhantomData,
        }
    }

    /// FN106d: whether this buffer is the huge form (registered private mapping).
    #[must_use]
    pub fn is_registered(&self) -> bool {
        self.reg_len > 0
    }

    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer's contents.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is a live, initialised (zeroed at construction) host allocation of `len`
        // bytes, and `&self` prevents concurrent mutation from Rust. A device-side write via an
        // in-flight `hipMemcpyAsync` is the caller's responsibility to order, exactly as it is in
        // C; that is what `Stream` synchronisation is for.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The buffer's contents, mutably.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, with `&mut self` giving exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// FN53c: an ACQUIRE load of the aligned u32 at byte `off` (a word a running kernel writes
    /// at system scope into coherent pinned memory, [`crate::HipThread::host_malloc_coherent`]).
    /// `None` when the word is out of range or misaligned.
    #[must_use]
    pub fn load_u32_acquire(&self, off: usize) -> Option<u32> {
        if off % 4 != 0 || off.checked_add(4)? > self.len {
            return None;
        }
        // SAFETY: an aligned, in-bounds word of a live allocation; an atomic view of plain memory
        // shared with a device that only touches it atomically.
        let a = unsafe { &*(self.ptr.add(off) as *const std::sync::atomic::AtomicU32) };
        Some(a.load(std::sync::atomic::Ordering::Acquire))
    }

    /// FN53c: a RELEASE store of the aligned u32 at byte `off` (every earlier store of this
    /// thread is visible before it). `false` when out of range or misaligned.
    pub fn store_u32_release(&self, off: usize, v: u32) -> bool {
        if off % 4 != 0 || off.checked_add(4).is_none_or(|e| e > self.len) {
            return false;
        }
        // SAFETY: as `load_u32_acquire`.
        let a = unsafe { &*(self.ptr.add(off) as *const std::sync::atomic::AtomicU32) };
        a.store(v, std::sync::atomic::Ordering::Release);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        true
    }

    /// FN53c: a relaxed atomic store of the aligned u64 at byte `off` (ordered by a later
    /// [`Self::store_u32_release`]). `false` when out of range or misaligned.
    pub fn store_u64_relaxed(&self, off: usize, v: u64) -> bool {
        if off % 8 != 0 || off.checked_add(8).is_none_or(|e| e > self.len) {
            return false;
        }
        // SAFETY: as `load_u32_acquire`, an aligned u64.
        let a = unsafe { &*(self.ptr.add(off) as *const std::sync::atomic::AtomicU64) };
        a.store(v, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// A shared view of `[off, off + len)`.
    ///
    /// `None` when the range runs off the end: an H2D whose source overruns its staging buffer
    /// reads whatever the allocator left there, and that is a wrong-weights bug the hash check
    /// cannot see (it hashed the *record*, not the copy).
    #[must_use]
    pub fn slice(&self, off: usize, len: usize) -> Option<PinnedSlice<'_>> {
        let end = off.checked_add(len)?;
        if end > self.len {
            return None;
        }
        Some(PinnedSlice {
            // SAFETY (pointer arithmetic): `off <= self.len`, so this stays inside the allocation.
            ptr: unsafe { self.ptr.add(off) },
            len,
            _buf: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// An exclusive view of `[off, off + len)`. Same bounds rule as [`PinnedBuf::slice`].
    #[must_use]
    pub fn slice_mut(&mut self, off: usize, len: usize) -> Option<PinnedSliceMut<'_>> {
        let end = off.checked_add(len)?;
        if end > self.len {
            return None;
        }
        Some(PinnedSliceMut {
            // SAFETY (pointer arithmetic): as above.
            ptr: unsafe { self.ptr.add(off) },
            len,
            _buf: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// Split the front of the buffer into disjoint exclusive views of `lens[i]` bytes each, in
    /// order (ws-h40: one pinned batch reading several differently sized regions). `None` when a
    /// length is 0 or the lengths sum past the buffer.
    #[must_use]
    pub fn split_mut(&mut self, lens: &[usize]) -> Option<Vec<PinnedSliceMut<'_>>> {
        let mut total = 0usize;
        for &l in lens {
            if l == 0 {
                return None;
            }
            total = total.checked_add(l)?;
        }
        if total > self.len {
            return None;
        }
        let base = self.ptr;
        let mut at = 0usize;
        Some(
            lens.iter()
                .map(|&l| {
                    let v = PinnedSliceMut {
                        // SAFETY: `at + l <= total <= self.len`, so every view is inside the
                        // allocation; consecutive views start where the previous one ends, so they
                        // are disjoint, and the `&mut self` borrow is split across them.
                        ptr: unsafe { base.add(at) },
                        len: l,
                        _buf: PhantomData,
                        _not_send: PhantomData,
                    };
                    at += l;
                    v
                })
                .collect(),
        )
    }

    /// Split the buffer into disjoint exclusive views of `chunk` bytes each — a staging ring.
    ///
    /// The tail is dropped rather than handed out short: a ring whose last slot is a different size
    /// from the others is a slot-size bug waiting to happen, and the caller sizes the allocation.
    /// `None` when `chunk` is 0 or larger than the buffer.
    #[must_use]
    pub fn chunks_mut(&mut self, chunk: usize) -> Option<Vec<PinnedSliceMut<'_>>> {
        if chunk == 0 || chunk > self.len {
            return None;
        }
        let n = self.len / chunk;
        let base = self.ptr;
        Some(
            (0..n)
                .map(|i| PinnedSliceMut {
                    // SAFETY: `i < n` and `n * chunk <= self.len`, so every view is inside the
                    // allocation, and the views are disjoint by construction. The `&mut self`
                    // borrow is split across them, which is what makes handing out N of them sound.
                    ptr: unsafe { base.add(i * chunk) },
                    len: chunk,
                    _buf: PhantomData,
                    _not_send: PhantomData,
                })
                .collect(),
        )
    }

    /// Split the buffer into disjoint, **`Send`** chunks of `chunk` bytes — a staging ring whose
    /// slots can be filled by I/O worker threads.
    ///
    /// The same split as [`PinnedBuf::chunks_mut`], except that the views may cross threads. That
    /// is the whole point: it is what lets an `O_DIRECT` read land in the staging slot instead of
    /// in a pageable buffer that then has to be copied.
    ///
    /// The tail is dropped rather than handed out short, and `None` comes back when `chunk` is 0 or
    /// larger than the buffer.
    #[must_use]
    pub fn send_chunks(&mut self, chunk: usize) -> Option<Vec<PinnedChunk<'_>>> {
        if chunk == 0 || chunk > self.len {
            return None;
        }
        let n = self.len / chunk;
        let base = self.ptr;
        Some(
            (0..n)
                .map(|i| PinnedChunk {
                    // SAFETY: `i < n` and `n * chunk <= self.len`, so every view is inside the
                    // allocation, and the views are disjoint by construction. The `&mut self`
                    // borrow is split across them, which is what makes handing out N of them sound.
                    ptr: unsafe { base.add(i * chunk) },
                    len: chunk,
                    off: i * chunk,
                    _buf: PhantomData,
                })
                .collect(),
        )
    }

    /// Host address of the allocation. `O_DIRECT` needs it 4096-aligned, and a caller that is going
    /// to read into this memory has to be able to check rather than assume.
    #[must_use]
    pub fn addr(&self) -> usize {
        self.ptr as usize
    }

    /// The base address as an integer.
    ///
    /// An integer rather than a pointer because this is what crosses the channel into the HIP
    /// thread, and `*mut c_void` is not `Send` -- which is the whole reason the raw tokens in this
    /// module are integers.
    pub(crate) fn as_usize(&self) -> usize {
        self.ptr as usize
    }
}

/// A shared view into a [`PinnedBuf`]: the source of one H2D copy.
///
/// The staging ring hands these out. It is `!Send` for the same reason every other handle here is,
/// and it borrows its buffer, so a slot cannot outlive the allocation it points into -- the failure
/// mode a ring of raw `(ptr, len)` pairs eventually produces.
#[derive(Clone, Copy)]
pub struct PinnedSlice<'a> {
    ptr: *const u8,
    len: usize,
    _buf: PhantomData<&'a PinnedBuf>,
    _not_send: PhantomData<*mut ()>,
}

impl PinnedSlice<'_> {
    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes behind the view.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the view was built from a live, initialised `PinnedBuf` range that it borrows,
        // and `&self` prevents concurrent mutation from Rust.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_usize(&self) -> usize {
        self.ptr as usize
    }
}

impl std::fmt::Debug for PinnedSlice<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PinnedSlice({:#x}, {} bytes)",
            self.ptr as usize, self.len
        )
    }
}

/// An exclusive view into a [`PinnedBuf`]: one slot of a staging ring, while it is being filled.
///
/// [`PinnedBuf::chunks_mut`] splits one allocation into N of these, which is what "allocate the
/// staging pool once" (I1) looks like in the type system: the slots are views, not allocations, so
/// there is no per-record `hipHostMalloc` to forget.
pub struct PinnedSliceMut<'a> {
    ptr: *mut u8,
    len: usize,
    _buf: PhantomData<&'a mut PinnedBuf>,
    _not_send: PhantomData<*mut ()>,
}

impl PinnedSliceMut<'_> {
    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes behind the view.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: as [`PinnedSlice::as_slice`]; the view uniquely borrows its range.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The bytes behind the view, mutably. This is where a record's payload is staged.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the view uniquely borrows a live, initialised range of its `PinnedBuf`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub(crate) fn as_usize(&self) -> usize {
        self.ptr as usize
    }

    /// A shared view of the first `bytes` of this slot, for the copy that reads it.
    ///
    /// Returns `None` past the end rather than clamping: a copy of the wrong length out of a
    /// staging slot is a wrong-weights bug, not a short read.
    #[must_use]
    pub fn head(&self, bytes: usize) -> Option<PinnedSlice<'_>> {
        if bytes > self.len {
            return None;
        }
        Some(PinnedSlice {
            ptr: self.ptr.cast_const(),
            len: bytes,
            _buf: PhantomData,
            _not_send: PhantomData,
        })
    }
}

impl std::fmt::Debug for PinnedSliceMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PinnedSliceMut({:#x}, {} bytes)",
            self.ptr as usize, self.len
        )
    }
}

/// An exclusive, **`Send`** view of part of a [`PinnedBuf`]: one staging slot that a worker thread
/// may fill.
///
/// This is the one type in this module that crosses threads, and it is deliberate. The `!Send` rule
/// exists so that a HIP *call* cannot happen off the HIP thread ([`Stream`], [`Event`], [`Module`]
/// and friends all carry a handle that could make one). A chunk carries no handle and exposes no
/// HIP call: it is page-locked host memory, which the driver treats the same whichever thread
/// writes it. Making it `Send` is what lets an `O_DIRECT` reader thread land a record *directly* in
/// the staging buffer instead of reading into pageable memory and copying — the copy that measured
/// 66% of the cold-start wall (`docs/LEDGER.md`, workstream K).
///
/// Chunks are produced by [`PinnedBuf::send_chunks`], are disjoint, and borrow their buffer, so
/// none of them can outlive the allocation. [`PinnedChunk::join`] merges *adjacent* chunks so a
/// record larger than one slot can be read into one contiguous buffer;
/// [`PinnedChunk::split_at`] gives the pieces back.
pub struct PinnedChunk<'a> {
    ptr: *mut u8,
    len: usize,
    /// Byte offset within the owning buffer, for adjacency checks and for log lines.
    off: usize,
    _buf: PhantomData<&'a mut PinnedBuf>,
}

// SAFETY: a chunk is an exclusive, disjoint range of ordinary (page-locked) host memory. It owns no
// HIP handle, so no HIP entry point is reachable through it, and nothing in this type is bound to
// the thread that created it. Exclusivity comes from `PinnedBuf::send_chunks`, which splits one
// `&mut` borrow into disjoint views and is the only constructor.
unsafe impl Send for PinnedChunk<'_> {}

impl PinnedChunk<'_> {
    /// Length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the chunk is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte offset of this chunk within the buffer it came from.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.off
    }

    /// The host address, for an alignment check (`O_DIRECT` needs 4096) or a log line.
    #[must_use]
    pub fn addr(&self) -> usize {
        self.ptr as usize
    }

    /// The bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the chunk is a live, initialised (the buffer is zeroed at construction),
        // exclusively-owned range, and `&self` prevents concurrent mutation from Rust.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The bytes, mutably. This is what `preadv` reads into.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, with `&mut self` giving exclusive access to a range no other chunk
        // overlaps.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// A device-copyable view of the first `bytes` of this chunk — the H2D source.
    ///
    /// `None` past the end rather than a clamp: a copy of the wrong length out of a staging slot is
    /// a wrong-weights bug, not a short read.
    #[must_use]
    pub fn head(&self, bytes: usize) -> Option<PinnedSlice<'_>> {
        if bytes > self.len {
            return None;
        }
        Some(PinnedSlice {
            ptr: self.ptr.cast_const(),
            len: bytes,
            _buf: PhantomData,
            _not_send: PhantomData,
        })
    }

    /// Merge two chunks that are adjacent in their buffer, `self` first.
    ///
    /// Returns them untouched if they are not adjacent, which is the only way a pool can ask for a
    /// contiguous run and find out it did its own bookkeeping wrong.
    pub fn join(self, next: Self) -> std::result::Result<Self, (Self, Self)> {
        let contiguous =
            self.off + self.len == next.off && self.ptr.wrapping_add(self.len) == next.ptr;
        if !contiguous {
            return Err((self, next));
        }
        Ok(PinnedChunk {
            ptr: self.ptr,
            len: self.len + next.len,
            off: self.off,
            _buf: PhantomData,
        })
    }

    /// Split at `mid`. `None` when `mid` is 0 or past the end.
    #[must_use]
    pub fn split_at(self, mid: usize) -> Option<(Self, Self)> {
        if mid == 0 || mid >= self.len {
            return None;
        }
        Some((
            PinnedChunk {
                ptr: self.ptr,
                len: mid,
                off: self.off,
                _buf: PhantomData,
            },
            PinnedChunk {
                // SAFETY (pointer arithmetic): `mid < self.len`, so this stays inside the chunk.
                ptr: unsafe { self.ptr.add(mid) },
                len: self.len - mid,
                off: self.off + mid,
                _buf: PhantomData,
            },
        ))
    }
}

impl std::fmt::Debug for PinnedChunk<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PinnedChunk(+{}, {} bytes @ {:#x})",
            self.off, self.len, self.ptr as usize
        )
    }
}

impl std::fmt::Debug for PinnedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PinnedBuf({} bytes)", self.len)
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        let addr = self.ptr as usize;
        if self.reg_len > 0 {
            // FN106d's teardown law: unregister FIRST, unmap only after it returned; a failed
            // unregister leaks the mapping (the driver may still hold its pages)
            let map_len = self.reg_len;
            post_destructor(&self.tx, "hipHostRegister", move |lib, ctx| {
                // SAFETY: `addr` is the base this buffer registered and nothing else unregisters it.
                let r = unsafe { lib.host_unregister(ctx, addr as *mut c_void) };
                let leaked = r.is_err();
                if !leaked {
                    crate::pinned::unmap(&crate::pinned::HugeMap {
                        base: addr,
                        map_len,
                    });
                }
                crate::pinned::stats().sub_huge(addr, map_len, leaked);
                r
            });
            return;
        }
        if self.counted {
            crate::pinned::stats().sub_shmem(self.len as u64);
        }
        post_destructor(&self.tx, "hipHostMalloc", move |lib, ctx| unsafe {
            lib.host_free(ctx, addr as *mut c_void)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `!Send` property is the whole point of this module, and `assert!(!T: Send)` is not
    // expressible directly on stable. This is the standard inherent-impl-beats-trait-impl trick:
    // the inherent `SEND` const only exists when `T: Send`, so for a `!Send` type the blanket
    // trait const (false) is what resolves.
    struct IsSend<T>(PhantomData<T>);
    trait NotSendFallback {
        const SEND: bool = false;
    }
    impl<T> NotSendFallback for IsSend<T> {}
    impl<T: Send> IsSend<T> {
        #[allow(dead_code)]
        const SEND: bool = true;
    }

    // Checked at compile time -- the property is about types, so a run-time check would only be a
    // slower way of asking the same question.
    #[test]
    fn owning_handles_are_not_send() {
        const {
            assert!(!<IsSend<Stream>>::SEND, "Stream must not be Send");
            assert!(!<IsSend<Event>>::SEND, "Event must not be Send");
            assert!(!<IsSend<Module>>::SEND, "Module must not be Send");
            assert!(!<IsSend<Graph>>::SEND, "Graph must not be Send");
            assert!(!<IsSend<GraphExec>>::SEND, "GraphExec must not be Send");
            assert!(!<IsSend<DevicePtr>>::SEND, "DevicePtr must not be Send");
            assert!(!<IsSend<PinnedBuf>>::SEND, "PinnedBuf must not be Send");
            assert!(
                !<IsSend<ImportedPtr>>::SEND,
                "ImportedPtr must not be Send: it unmaps on the HIP thread"
            );
            assert!(
                !<IsSend<Function<'static>>>::SEND,
                "Function must not be Send"
            );
        }
    }

    #[test]
    fn the_not_send_check_has_a_negative_control() {
        // If these read false the assertions above are vacuous and prove nothing.
        const {
            assert!(<IsSend<u64>>::SEND, "negative control: u64 IS Send");
            assert!(<IsSend<DeviceAddr>>::SEND, "raw tokens must stay Send");
            assert!(<IsSend<RawStream>>::SEND, "raw tokens must stay Send");
        }
    }

    #[test]
    fn views_are_not_send_and_stream_ids_are() {
        const {
            assert!(
                !<IsSend<PinnedSlice<'static>>>::SEND,
                "PinnedSlice must not be Send"
            );
            assert!(
                !<IsSend<PinnedSliceMut<'static>>>::SEND,
                "PinnedSliceMut must not be Send"
            );
            // A StreamId exists precisely so that identity CAN cross a thread; if this ever reads
            // false the type has stopped doing its job.
            assert!(<IsSend<StreamId>>::SEND, "StreamId must be Send");
        }
    }

    /// A `PinnedBuf` over ordinary heap memory. Sound for the view arithmetic below -- the type
    /// only ever reads and writes the bytes it was handed -- and the only way to test the ring
    /// split on a host with no ROCm, which is this one.
    fn fake_pinned(storage: &mut [u8]) -> (PinnedBuf, std::sync::mpsc::Receiver<Msg>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let len = storage.len();
        (PinnedBuf::new(storage.as_mut_ptr().cast(), len, tx), rx)
    }

    #[test]
    fn staging_ring_views_are_disjoint_and_bounded() {
        let mut storage = vec![0u8; 4096];
        let (mut buf, _rx) = fake_pinned(&mut storage);

        let mut slots = buf.chunks_mut(1024).expect("4 slots");
        assert_eq!(slots.len(), 4);
        for (i, slot) in slots.iter_mut().enumerate() {
            assert_eq!(slot.len(), 1024);
            slot.as_mut_slice().fill(i as u8 + 1);
        }
        // Disjoint: each slot still reads back only its own fill.
        for (i, slot) in slots.iter().enumerate() {
            assert!(slot.as_slice().iter().all(|&b| b == i as u8 + 1));
        }
        assert_eq!(slots[0].head(1024).map(|v| v.len()), Some(1024));
        assert!(
            slots[0].head(1025).is_none(),
            "head must not overrun a slot"
        );
        drop(slots);

        // A chunk size the buffer cannot hold is a refusal, not a short ring.
        assert!(buf.chunks_mut(0).is_none());
        assert!(buf.chunks_mut(4097).is_none());
        // 3000-byte chunks: one slot, 1096 bytes unused rather than a ragged second slot.
        assert_eq!(buf.chunks_mut(3000).map(|v| v.len()), Some(1));

        assert_eq!(buf.slice(4095, 1).map(|v| v.len()), Some(1));
        assert!(buf.slice(4096, 1).is_none());
        assert!(buf.slice(usize::MAX, 1).is_none(), "no overflow wrap");
        assert!(buf.slice_mut(1, 4096).is_none());
    }

    #[test]
    fn send_chunks_are_send_and_the_other_views_are_not() {
        const {
            // The one type here that crosses threads, because a reader thread has to fill it.
            assert!(
                <IsSend<PinnedChunk<'static>>>::SEND,
                "PinnedChunk must be Send"
            );
            // Negative controls: everything that can reach a HIP call stays put.
            assert!(
                !<IsSend<PinnedSliceMut<'static>>>::SEND,
                "PinnedSliceMut must not be Send"
            );
            assert!(!<IsSend<PinnedBuf>>::SEND, "PinnedBuf must not be Send");
        }
    }

    #[test]
    fn chunks_join_and_split_only_where_they_are_adjacent() {
        let mut storage = vec![0u8; 4096];
        let (mut buf, _rx) = fake_pinned(&mut storage);

        let mut chunks = buf.send_chunks(1024).expect("4 chunks");
        assert_eq!(chunks.len(), 4);
        for (i, c) in chunks.iter_mut().enumerate() {
            assert_eq!(c.offset(), i * 1024);
            c.as_mut_slice().fill(i as u8 + 1);
        }

        let mut it = chunks.into_iter();
        let c0 = it.next().unwrap();
        let c1 = it.next().unwrap();
        let c2 = it.next().unwrap();
        let c3 = it.next().unwrap();

        // Adjacent, in order: joins into one contiguous 2048-byte buffer that still reads back as
        // the two fills, which is what a record spanning two slots depends on.
        let joined = c0.join(c1).expect("adjacent chunks join");
        assert_eq!(joined.len(), 2048);
        assert!(joined.as_slice()[..1024].iter().all(|&b| b == 1));
        assert!(joined.as_slice()[1024..].iter().all(|&b| b == 2));
        assert_eq!(joined.head(2048).map(|v| v.len()), Some(2048));
        assert!(joined.head(2049).is_none());

        // Out of order is not adjacent, and both come back untouched.
        let (c3, c2) = c3.join(c2).expect_err("3 then 2 is not adjacent");
        assert_eq!((c2.offset(), c3.offset()), (2048, 3072));

        // Split puts the pieces back exactly where they came from.
        let (a, b) = joined.split_at(1024).expect("split");
        assert_eq!((a.offset(), a.len()), (0, 1024));
        assert_eq!((b.offset(), b.len()), (1024, 1024));
        assert_eq!(a.addr() + 1024, b.addr());
        assert!(b.split_at(0).is_none());
    }

    #[test]
    fn device_addresses_are_checked_arithmetic() {
        let a = DeviceAddr(0x1000);
        assert_eq!(a.as_u64(), 0x1000);
        assert_eq!(a.offset(0x10).unwrap().as_u64(), 0x1010);
        assert_eq!(DeviceAddr(u64::MAX).offset(1), None);
        assert_eq!(DeviceAddr::NULL.as_u64(), 0);
    }

    #[test]
    fn raw_tokens_round_trip_through_pointers() {
        let p = 0xdead_beef_usize as *mut c_void;
        assert_eq!(RawStream::from_raw(p).as_ptr(), p);
        assert_eq!(DeviceAddr::from_raw(p).as_u64(), 0xdead_beef);
    }
}
