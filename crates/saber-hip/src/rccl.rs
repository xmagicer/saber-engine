//! Raw RCCL bindings: `dlopen` plus hand-written `nccl*` `extern "C"` declarations -- the
//! **compat mode's** collective transport, for the R9700 pair whose `peer` probe reads false.
//!
//! The on-machine C++ probe (RCCL 2.30.4, `NCCL_P2P_DISABLE=1`, Simple protocol, SHM on) ran
//! two-process BF16 AllReduce/AllGather plus HIPGraph capture/replay 20x each, bit-exact, with
//! RCCL capture nodes = 1. That is SHM transport, **not** hardware P2P: nothing here calls
//! `hipDeviceEnablePeerAccess`, touches no environment variable, claims no P2P.
//!
//! ABI, confirmed against the machine's RCCL 2.30.4 headers and probe: `ncclUniqueId_t` is
//! `char[128]` and `ncclCommInitRank` takes it **by value**; `ncclComm_t`/`hipStream_t` are
//! `void *`; `ncclUint8 = 1`, `ncclBfloat16 = 9`, `ncclSum = 0`. Raw comms cross this module as
//! `usize`; the `!Send` borrowed token lives in `crate::handles`; `HipThread` owns native teardown. The `Library` is held in
//! an `Arc` and must outlive every comm. Every method takes `&Ctx`: HIP thread only, provably.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::Arc;

use libloading::Library;

use crate::error::{Error, Result};
use crate::thread::Ctx;

/// `NCCL_UNIQUE_ID_BYTES` from RCCL 2.30.4's `nccl.h`.
const UNIQUE_ID_BYTES: usize = 128;
/// `ncclUint8 = 1` (probe-verified on RCCL 2.30.4).
const NCCL_UINT8: c_int = 1;
/// `ncclBfloat16 = 9` (probe-verified on RCCL 2.30.4).
const NCCL_BF16: c_int = 9;
/// `ncclSum = 0` (probe-verified on RCCL 2.30.4).
const NCCL_SUM: c_int = 0;

type FnGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type FnGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type FnGetUniqueId = unsafe extern "C" fn(*mut RcclUniqueId) -> c_int;
type FnCommInitRankConfig =
    unsafe extern "C" fn(*mut *mut c_void, c_int, RcclUniqueId, c_int, *mut RcclConfig) -> c_int;

/// The public v2.30.4 config ABI, verified against the deployment's C header.
#[repr(C)]
struct RcclConfig {
    size: usize,
    magic: u32,
    version: u32,
    blocking: c_int,
    cga_cluster_size: c_int,
    min_ctas: c_int,
    max_ctas: c_int,
    net_name: *const c_char,
    split_share: c_int,
    traffic_class: c_int,
    comm_name: *const c_char,
    collnet_enable: c_int,
    cta_policy: c_int,
    shrink_share: c_int,
    nvls_ctas: c_int,
    channels_per_net_peer: c_int,
    nvlink_centric_sched: c_int,
    graph_usage_mode: c_int,
    num_rma_ctx: c_int,
    max_p2p_peers: c_int,
}

impl RcclConfig {
    fn mixed_graphs() -> Self {
        let unset = c_int::MIN;
        Self {
            size: std::mem::size_of::<Self>(),
            magic: 0xcafebeef,
            version: 23004,
            blocking: 1,
            cga_cluster_size: unset,
            min_ctas: unset,
            max_ctas: unset,
            net_name: std::ptr::null(),
            split_share: unset,
            traffic_class: unset,
            comm_name: std::ptr::null(),
            collnet_enable: unset,
            cta_policy: unset,
            shrink_share: unset,
            nvls_ctas: unset,
            channels_per_net_peer: unset,
            nvlink_centric_sched: unset,
            graph_usage_mode: 2,
            num_rma_ctx: unset,
            max_p2p_peers: unset,
        }
    }
}
// AllReduce/AllGather args: sendbuff, recvbuff, count, [datatype, op,] comm, stream.
// `count` is size_t end-to-end: passed through, never narrowed or widened.
type FnAllReduce = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    c_int,
    c_int,
    *mut c_void,
    *mut c_void,
) -> c_int;
type FnAllGather = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    c_int,
    *mut c_void,
    *mut c_void,
) -> c_int;
type FnCommDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;

/// A `ncclUniqueId_t`: an opaque 128-byte bootstrap token. Rank 0 generates it; the others
/// receive it over Saber's rank bootstrap exchange and hand it back verbatim. `repr(C)` over
/// `[u8; 128]` is layout-identical to `char[128]`, which matters because `ncclCommInitRank`
/// takes the struct **by value**.
#[repr(C)]
#[derive(Clone)]
pub struct RcclUniqueId {
    /// The opaque bytes. Never interpreted or logged by Saber.
    bytes: [u8; UNIQUE_ID_BYTES],
}

impl RcclUniqueId {
    /// Rebuilds an id from bytes received from a peer. Exactly 128 bytes accepted: a short
    /// buffer would leave the tail uninitialised, a long one means the peer is not speaking
    /// RCCL 2.30.4's ABI.
    pub fn from_bytes(src: &[u8]) -> Result<Self> {
        if src.len() != UNIQUE_ID_BYTES {
            return Err(Error::Invalid(format!(
                "ncclUniqueId must be exactly {UNIQUE_ID_BYTES} bytes, got {}",
                src.len()
            )));
        }
        let mut bytes = [0u8; UNIQUE_ID_BYTES];
        bytes.copy_from_slice(src);
        Ok(RcclUniqueId { bytes })
    }

    /// The 128 bytes, for sending to a peer rank.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl std::fmt::Debug for RcclUniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A magic token, not data for a human; the bytes themselves are never printed.
        write!(f, "RcclUniqueId({} bytes, opaque)", self.bytes.len())
    }
}

/// The loaded RCCL library: the `Library` handle plus one function pointer per entry point.
/// Held as `Arc<RcclLib>` so the `dlopen` outlives every comm -- the raw pointers RCCL hands
/// back are only valid while the library is. Every method takes a `&Ctx` proving the caller is
/// the HIP thread; this type never spawns or blocks on its own.
pub(crate) struct RcclLib {
    soname: String,
    get_version: FnGetVersion,
    get_error_string: FnGetErrorString,
    get_unique_id: FnGetUniqueId,
    comm_init_rank: FnCommInitRankConfig,
    all_reduce: FnAllReduce,
    all_gather: FnAllGather,
    comm_destroy: FnCommDestroy,
    /// Declared last so it is dropped last: the function pointers above are only valid while
    /// this is loaded. (Rust drops struct fields in declaration order.)
    _lib: Library,
}

impl std::fmt::Debug for RcclLib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RcclLib({})", self.soname)
    }
}

macro_rules! load {
    ($lib:expr, $soname:expr, $name:literal) => {{
        let sym = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }.map_err(|e| {
            Error::Invalid(format!(
                "RCCL library '{}' does not export '{}': {}",
                $soname, $name, e
            ))
        })?;
        *sym
    }};
}

impl RcclLib {
    /// `dlopen`s RCCL and binds every entry point, all-or-nothing: a partially bound
    /// collective library is how numerics degrade silently. RCCL exports the `nccl*` names.
    /// Load failures are `Error::Invalid` because `Error::NotAvailable`'s message is
    /// HIP-specific and would mislabel this library.
    pub(crate) fn load(path: &str) -> Result<Arc<Self>> {
        // SAFETY: `Library::new` runs the library's initialisers. Loading RCCL is exactly what
        // any ROCm collective program does at startup; nothing else in the process is touched.
        #[cfg(target_os = "linux")]
        let lib: Library = unsafe {
            libloading::os::unix::Library::open(
                Some(path),
                libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NODELETE,
            )
        }
        .map(Library::from)
        .map_err(|e| Error::Invalid(format!("RCCL library '{path}' could not be loaded: {e}")))?;
        #[cfg(not(target_os = "linux"))]
        let lib = unsafe { Library::new(path) }.map_err(|e| {
            Error::Invalid(format!("RCCL library '{path}' could not be loaded: {e}"))
        })?;
        Ok(Arc::new(RcclLib {
            soname: path.to_string(),
            get_version: load!(lib, path, "ncclGetVersion"),
            get_error_string: load!(lib, path, "ncclGetErrorString"),
            get_unique_id: load!(lib, path, "ncclGetUniqueId"),
            comm_init_rank: load!(lib, path, "ncclCommInitRankConfig"),
            all_reduce: load!(lib, path, "ncclAllReduce"),
            all_gather: load!(lib, path, "ncclAllGather"),
            comm_destroy: load!(lib, path, "ncclCommDestroy"),
            _lib: lib,
        }))
    }

    /// `ncclGetVersion()`: the compiled-in version, e.g. `23004` for 2.30.4. Returns the int
    /// through an out-pointer; the function itself returns `ncclResult_t`.
    pub(crate) fn version(&self, _ctx: &Ctx) -> Result<i32> {
        let mut version = 0;
        // SAFETY: `version` is a writable int for the duration of the call.
        let status = unsafe { (self.get_version)(&mut version) };
        self.check("ncclGetVersion", status)?;
        Ok(version)
    }

    /// `ncclGetUniqueId()`: rank 0's bootstrap token, shipped to peers verbatim.
    pub(crate) fn unique_id(&self, _ctx: &Ctx) -> Result<RcclUniqueId> {
        let mut id = RcclUniqueId {
            bytes: [0; UNIQUE_ID_BYTES],
        };
        // SAFETY: writes 128 bytes through the out-pointer, which aliases `id` and is valid
        // for the whole call; `RcclUniqueId` is layout-identical to `ncclUniqueId_t`.
        let r = unsafe { (self.get_unique_id)(&mut id) };
        self.check("ncclGetUniqueId", r)?;
        Ok(id)
    }

    /// `ncclCommInitRankConfig`: the blocking constructor, permitting eager and graph calls.
    /// Returns the raw comm as a `usize`; `HipThread` retains and destroys it after queued graph drops.
    pub(crate) fn init_rank(
        &self,
        _ctx: &Ctx,
        world: i32,
        rank: i32,
        id: &RcclUniqueId,
    ) -> Result<usize> {
        if world < 2 {
            return Err(Error::Invalid(format!(
                "ncclCommInitRank refused: world {world} < 2 (a one-rank comm is pointless)"
            )));
        }
        if rank < 0 || rank >= world {
            return Err(Error::Invalid(format!(
                "ncclCommInitRank refused: rank {rank} out of range 0..{world}"
            )));
        }
        if self.version(_ctx)? < 23004 {
            return Err(Error::Invalid(
                "RCCL compatibility requires the validated v2.30.4 config API or newer".into(),
            ));
        }
        let mut comm: *mut c_void = std::ptr::null_mut();
        let mut config = RcclConfig::mixed_graphs();
        // SAFETY: `comm` out-pointer is valid for the call; `id` is passed **by value**
        // (cloned for the move), exactly what the nccl ABI demands for its 128-byte struct.
        let r = unsafe { (self.comm_init_rank)(&mut comm, world, id.clone(), rank, &mut config) };
        self.check("ncclCommInitRankConfig", r)?;
        if comm.is_null() {
            return Err(Error::Invalid(
                "ncclCommInitRank succeeded but returned a null comm".into(),
            ));
        }
        Ok(comm as usize)
    }

    /// `ncclAllReduce` over BF16 with `ncclSum`: `count` bf16 elements, in-place when
    /// `src == dst`. Enqueues on `stream`; completion is the caller's stream-ordering problem.
    pub(crate) fn all_reduce_bf16(
        &self,
        _ctx: &Ctx,
        comm: usize,
        src: usize,
        dst: usize,
        count: usize,
        stream: usize,
    ) -> Result<()> {
        self.check_args("ncclAllReduce", comm, src, dst, count, stream)?;
        // SAFETY: `src`/`dst` are device addresses of `count * 2` valid bytes owned by the
        // caller; comm and stream are live handles; the call only enqueues work, never waits.
        let r = unsafe {
            (self.all_reduce)(
                src as *const c_void,
                dst as *mut c_void,
                count,
                NCCL_BF16,
                NCCL_SUM,
                comm as *mut c_void,
                stream as *mut c_void,
            )
        };
        self.check("ncclAllReduce", r)
    }

    /// `ncclAllGather` over Uint8: `count` bytes from every rank concatenated into `dst`,
    /// which must therefore be `count * world` bytes. Byte-typed so buffer payloads need no
    /// element-type lies.
    pub(crate) fn all_gather_bytes(
        &self,
        _ctx: &Ctx,
        comm: usize,
        src: usize,
        dst: usize,
        count: usize,
        stream: usize,
    ) -> Result<()> {
        self.check_args("ncclAllGather", comm, src, dst, count, stream)?;
        // SAFETY: as `all_reduce_bf16`; additionally `dst` must hold `count * world` bytes,
        // which only the caller (who knows `world`) can guarantee.
        let r = unsafe {
            (self.all_gather)(
                src as *const c_void,
                dst as *mut c_void,
                count,
                NCCL_UINT8,
                comm as *mut c_void,
                stream as *mut c_void,
            )
        };
        self.check("ncclAllGather", r)
    }

    /// `ncclCommDestroy`: releases the comm. Call only once the caller's stream ordering has
    /// let every enqueued collective on it finish.
    pub(crate) fn destroy(&self, _ctx: &Ctx, comm: usize) -> Result<()> {
        if comm == 0 {
            return Err(Error::Invalid("ncclCommDestroy refused: null comm".into()));
        }
        // SAFETY: `comm` is a live handle from `init_rank` that the caller will not use again.
        let r = unsafe { (self.comm_destroy)(comm as *mut c_void) };
        self.check("ncclCommDestroy", r)
    }

    /// Shared precondition block for the collectives: every handle and buffer non-null, and a
    /// non-empty transfer. Refused before any FFI call is made.
    fn check_args(
        &self,
        op: &str,
        comm: usize,
        src: usize,
        dst: usize,
        count: usize,
        stream: usize,
    ) -> Result<()> {
        for (what, v) in [
            ("comm", comm),
            ("src", src),
            ("dst", dst),
            ("stream", stream),
        ] {
            if v == 0 {
                return Err(Error::Invalid(format!("{op} refused: null {what}")));
            }
        }
        if count == 0 {
            return Err(Error::Invalid(format!("{op} refused: count 0")));
        }
        Ok(())
    }

    /// Turns an `ncclResult_t` into `Ok(())` when zero, otherwise an error carrying the op,
    /// the code and `ncclGetErrorString`'s text. Nothing RCCL reports is ever swallowed.
    fn check(&self, op: &str, r: c_int) -> Result<()> {
        if r == 0 {
            return Ok(());
        }
        // SAFETY: `ncclGetErrorString` only reads `code`; it returns a pointer to a string
        // constant owned by the library, which outlives this borrow.
        let ptr = unsafe { (self.get_error_string)(r) };
        let text = if ptr.is_null() {
            "<null>".to_string()
        } else {
            // SAFETY: a non-null return of a C string function: NUL-terminated and readable.
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Error::Invalid(format!(
            "{op} failed with ncclResult {r}: {text}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_layout_matches_the_independently_compiled_c_probe() {
        let c = RcclConfig::mixed_graphs();
        assert_eq!(std::mem::size_of::<RcclConfig>(), 96);
        assert_eq!(std::mem::align_of::<RcclConfig>(), 8);
        assert_eq!(std::mem::offset_of!(RcclConfig, graph_usage_mode), 80);
        assert_eq!(c.magic, 3405692655);
        assert_eq!(c.version, 23004);
        assert_eq!(c.graph_usage_mode, 2);
        assert_eq!(c.blocking, 1);
    }

    #[test]
    fn unique_id_round_trips_exactly_128_bytes() {
        let bytes = [0xA5u8; UNIQUE_ID_BYTES];
        assert_eq!(
            RcclUniqueId::from_bytes(&bytes).unwrap().as_bytes(),
            &bytes[..]
        );
    }

    #[test]
    fn unique_id_rejects_wrong_lengths() {
        for n in [0usize, 1, 127, 129, 256] {
            assert!(
                RcclUniqueId::from_bytes(&vec![0u8; n]).is_err(),
                "{n} bytes"
            );
        }
    }

    #[test]
    fn debug_never_prints_the_bytes() {
        let id = RcclUniqueId::from_bytes(&[b'S'; UNIQUE_ID_BYTES]).unwrap();
        let s = format!("{id:?}");
        assert!(!s.contains('S'), "Debug leaked id content: {s}");
        assert!(s.contains("128"), "Debug must state the length: {s}");
    }

    #[test]
    fn unique_id_abi_is_128_bytes() {
        assert_eq!(std::mem::size_of::<RcclUniqueId>(), 128);
        assert_eq!(std::mem::align_of::<RcclUniqueId>(), 1);
    }
}
