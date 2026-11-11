//! The raw HIP surface: `dlopen` plus the hand-written `extern "C"` declarations.
//!
//! **Nothing here is linked at build time.** The Rust binary carries zero ROCm: kernels are
//! compiled to `.hsaco` ahead of time inside a container and loaded with `hipModuleLoadData`, so
//! nothing in this build touches a HIP header or a ROCm library. That is also why these ~40
//! declarations are hand-written rather than `bindgen`ed from `hip_runtime_api.h`: the header is
//! 476 KB of C++, it is not on the host, and the twelve types this crate actually needs are
//! `void *` and `int`.
//!
//! Types in these signatures are deliberately opaque. `hipStream_t`, `hipEvent_t`, `hipModule_t`,
//! `hipFunction_t`, `hipGraph_t` and `hipGraphExec_t` are all `void *` in HIP; they appear here as
//! `*mut c_void` and are wrapped by the `!Send` newtypes in [`crate::handles`] before any other
//! crate can see them, because a bare `*mut c_void` gets `Send`ed the first time someone is in a
//! hurry.
//!
//! This module and `saber-hsa` are the only places in the engine allowed to contain
//! `unsafe extern "C"`.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

use libloading::Library;

use crate::error::{Error, HipError, Result};
use crate::thread::Ctx;

/// The soname this engine binds to. HIP 7.x. Passed explicitly rather than read from the
/// environment: there is no `getenv` anywhere in the engine, so an alternative userspace is
/// selected by the plan or by a CLI flag, never by an ambient variable.
pub const DEFAULT_HIP_SONAME: &str = "libamdhip64.so.7";

/// Bytes of a `hipIpcMemHandle_t`, from `HIP_IPC_HANDLE_SIZE`.
pub const IPC_HANDLE_BYTES: usize = 64;

/// A `hipIpcMemHandle_t`: an opaque 64-byte token that names a device allocation to another
/// process. Passed to `hipIpcOpenMemHandle` **by value**, which is why it is a real struct here.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IpcMemHandle {
    /// The opaque bytes. Not interpreted by anything in Saber; they are exchanged with the peer
    /// rank verbatim.
    pub reserved: [c_char; IPC_HANDLE_BYTES],
}

impl Default for IpcMemHandle {
    fn default() -> Self {
        IpcMemHandle {
            reserved: [0; IPC_HANDLE_BYTES],
        }
    }
}

impl std::fmt::Debug for IpcMemHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IpcMemHandle({} bytes)", IPC_HANDLE_BYTES)
    }
}

/// Direction argument of `hipMemcpyAsync`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum MemcpyKind {
    /// `hipMemcpyHostToHost`
    HostToHost = 0,
    /// `hipMemcpyHostToDevice`
    HostToDevice = 1,
    /// `hipMemcpyDeviceToHost`
    DeviceToHost = 2,
    /// `hipMemcpyDeviceToDevice` -- including peer-to-peer over the P2P link.
    DeviceToDevice = 3,
    /// `hipMemcpyDefault` -- inferred from the pointers' address ranges.
    Default = 4,
}

/// `hipStreamCreateWithFlags` flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum StreamFlags {
    /// `hipStreamDefault` -- synchronises with the legacy null stream.
    Default = 0,
    /// `hipStreamNonBlocking` -- does not synchronise with the null stream. Everything the engine
    /// creates uses this; an implicit dependency on the null stream is a serialisation nobody
    /// asked for.
    NonBlocking = 1,
}

/// `hipExtMallocWithFlags` allocation types.
///
/// The one that matters here is [`MallocFlags::FineGrained`]. A one-shot push all-reduce writes
/// into the *peer's* scratch and then spins on a flag the peer wrote, so both sides have to see
/// each other's stores: coarse-grained memory may hold them in a cache the peer cannot snoop, and
/// the symptom is not a wrong answer but a **hang**, which is the failure mode the ledger warns
/// about. The reference allocates its IPC scratch and flags this way
/// (`saber_ar_ipc_alloc(size, finegrained=1)`), and Saber matches it.
///
/// Note that `hipDeviceAttributeFineGrainSupport` reads **0** on these cards while this allocation
/// succeeds: the attribute is not the gate, the allocation is. Do not "optimise" this to
/// `hipMalloc` on the strength of the attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MallocFlags {
    /// `hipDeviceMallocDefault` -- what `hipMalloc` gives.
    Default = 0x0,
    /// `hipDeviceMallocFinegrained` -- device memory the peer's stores are coherent in.
    FineGrained = 0x1,
    /// `hipMallocSignalMemory` -- for `hipStreamWaitValue*`.
    SignalMemory = 0x2,
    /// `hipDeviceMallocUncached` -- bypasses the cache entirely.
    Uncached = 0x3,
}

/// `hipEventCreateWithFlags` flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EventFlags {
    /// `hipEventDefault` -- timing enabled, spin wait.
    Default = 0,
    /// `hipEventBlockingSync` -- `hipEventSynchronize` sleeps instead of spinning.
    BlockingSync = 1,
    /// `hipEventDisableTiming` -- cheaper; required for events used only as dependencies.
    DisableTiming = 2,
}

/// `hipSetDeviceFlags` schedule flags (FN57): how the HIP runtime's OWN host waits wait --
/// `hipEventSynchronize`, `hipStreamSynchronize`, a synchronous `hipMemcpy`.
///
/// ROCclr maps them onto one switch, the device's active wait: `Auto`, `Spin` and `Yield` all
/// busy-poll the completion signal on the waiting thread; `BlockingSync` polls briefly and then
/// sleeps in the kernel on the signal's interrupt (ROCr's `HSA_WAIT_STATE_BLOCKED`), the only form
/// that gives the core back while the card works.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DeviceSchedule {
    /// `hipDeviceScheduleAuto` -- the runtime's default (an active wait on ROCm).
    Auto = 0,
    /// `hipDeviceScheduleSpin` -- an active wait.
    Spin = 1,
    /// `hipDeviceScheduleYield` -- an active wait on ROCm too.
    Yield = 2,
    /// `hipDeviceScheduleBlockingSync` -- the interrupt wait.
    BlockingSync = 4,
}

impl DeviceSchedule {
    /// The schedule bits of a `hipGetDeviceFlags` word (`hipDeviceScheduleMask` = 0x7).
    #[must_use]
    pub fn from_flags(flags: u32) -> Option<DeviceSchedule> {
        match flags & 0x7 {
            0 => Some(DeviceSchedule::Auto),
            1 => Some(DeviceSchedule::Spin),
            2 => Some(DeviceSchedule::Yield),
            4 => Some(DeviceSchedule::BlockingSync),
            _ => None,
        }
    }
}

/// `hipStreamBeginCapture` mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum StreamCaptureMode {
    /// `hipStreamCaptureModeGlobal` -- serialises against *every* thread in the process. The
    /// example default, and the reason a `/metrics` handler calling `hipMemGetInfo` on a tokio
    /// worker fails intermittently, under load, only while a bucket is capturing.
    Global = 0,
    /// `hipStreamCaptureModeThreadLocal` -- serialises only against the capturing thread. Correct
    /// here precisely because every HIP call already runs on one thread.
    ThreadLocal = 1,
    /// `hipStreamCaptureModeRelaxed`
    Relaxed = 2,
}

// --------------------------------------------------------------------------------------------- //
// Signatures                                                                                      //
// --------------------------------------------------------------------------------------------- //

type FnI32Out = unsafe extern "C" fn(*mut c_int) -> c_int;
type FnSetDevice = unsafe extern "C" fn(c_int) -> c_int;
type FnSetDeviceFlags = unsafe extern "C" fn(c_uint) -> c_int;
type FnGetDeviceFlags = unsafe extern "C" fn(*mut c_uint) -> c_int;
type FnDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type FnDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type FnDeviceTotalMem = unsafe extern "C" fn(*mut usize, c_int) -> c_int;
type FnDeviceGetPciBusId = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type FnMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type FnExtMallocWithFlags = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int;
type FnFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnHostMalloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int;
type FnHostFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnHostRegister = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int;
type FnHostUnregister = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnHostGetDevicePointer = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> c_int;
type FnMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, *mut c_void) -> c_int;
type FnMemsetAsync = unsafe extern "C" fn(*mut c_void, c_int, usize, *mut c_void) -> c_int;
type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;
type FnStreamCreateWithFlags = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> c_int;
type FnExtStreamCreateWithCuMask = unsafe extern "C" fn(*mut *mut c_void, u32, *const u32) -> c_int;
type FnPtrIn = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnEventCreateWithFlags = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> c_int;
type FnEventRecord = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type FnEventElapsedTime = unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> c_int;
type FnModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
type FnModuleGetFunction =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
#[allow(clippy::type_complexity)]
type FnModuleLaunchKernel = unsafe extern "C" fn(
    *mut c_void,      // function
    c_uint,           // gridDimX  (HIP counts BLOCKS here, not threads)
    c_uint,           // gridDimY
    c_uint,           // gridDimZ
    c_uint,           // blockDimX
    c_uint,           // blockDimY
    c_uint,           // blockDimZ
    c_uint,           // sharedMemBytes (dynamic LDS)
    *mut c_void,      // stream
    *mut *mut c_void, // kernelParams
    *mut *mut c_void, // extra
) -> c_int;
type FnStreamBeginCapture = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type FnStreamEndCapture = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> c_int;
type FnGraphGetNodes = unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut usize) -> c_int;
type FnGraphNodeGetType = unsafe extern "C" fn(*mut c_void, *mut c_int) -> c_int;
type FnGraphInstantiate = unsafe extern "C" fn(
    *mut *mut c_void, // pGraphExec
    *mut c_void,      // graph
    *mut *mut c_void, // pErrorNode
    *mut c_char,      // pLogBuffer
    usize,            // bufferSize
) -> c_int;
type FnGraphLaunch = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type FnDeviceEnablePeerAccess = unsafe extern "C" fn(c_int, c_uint) -> c_int;
type FnDeviceCanAccessPeer = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
type FnDeviceSynchronize = unsafe extern "C" fn() -> c_int;
type FnStreamWaitEvent = unsafe extern "C" fn(*mut c_void, *mut c_void, c_uint) -> c_int;
type FnStreamWaitValue32 =
    unsafe extern "C" fn(*mut c_void, *mut c_void, u32, c_uint, u32) -> c_int;
type FnMemcpyPeerAsync =
    unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize, *mut c_void) -> c_int;
type FnIpcGetMemHandle = unsafe extern "C" fn(*mut IpcMemHandle, *mut c_void) -> c_int;
type FnIpcOpenMemHandle = unsafe extern "C" fn(*mut *mut c_void, IpcMemHandle, c_uint) -> c_int;
type FnImportExternalMemory =
    unsafe extern "C" fn(*mut *mut c_void, *const ExtMemHandleDesc) -> c_int;
type FnExternalMemoryGetMappedBuffer =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const ExtMemBufferDesc) -> c_int;

/// FN76: `hipExternalMemoryHandleDesc` (104 B: the type, the 16-byte handle union at 8, the size
/// at 24, the flags at 32, 16 reserved words).
#[repr(C)]
pub(crate) struct ExtMemHandleDesc {
    kind: c_int,
    pad0: c_int,
    fd: c_int,
    pad1: c_int,
    name: u64,
    size: u64,
    flags: c_uint,
    reserved: [c_uint; 16],
}

/// FN76: `hipExternalMemoryBufferDesc` (88 B).
#[repr(C)]
pub(crate) struct ExtMemBufferDesc {
    offset: u64,
    size: u64,
    flags: c_uint,
    reserved: [c_uint; 16],
}

/// `hipExternalMemoryHandleTypeOpaqueFd`.
const EXT_MEM_OPAQUE_FD: c_int = 1;
type FnGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

/// The loaded HIP runtime: the `Library` handle plus one function pointer per entry point.
///
/// Private to this crate and never shared between threads. It is owned by the thread
/// [`crate::HipThread`] spawns, and its methods all take a `&`[`Ctx`] proving the caller is on that
/// thread.
pub(crate) struct Lib {
    soname: String,

    get_device_count: FnI32Out,
    set_device: FnSetDevice,
    set_device_flags: FnSetDeviceFlags,
    get_device_flags: FnGetDeviceFlags,
    get_device: FnI32Out,
    device_get_attribute: FnDeviceGetAttribute,
    device_get_name: FnDeviceGetName,
    device_total_mem: FnDeviceTotalMem,
    device_get_pci_bus_id: FnDeviceGetPciBusId,
    device_synchronize: FnDeviceSynchronize,

    malloc: FnMalloc,
    ext_malloc_with_flags: FnExtMallocWithFlags,
    free: FnFree,
    host_malloc: FnHostMalloc,
    host_free: FnHostFree,
    host_register: FnHostRegister,
    host_unregister: FnHostUnregister,
    host_get_device_pointer: FnHostGetDevicePointer,
    memcpy_async: FnMemcpyAsync,
    memset_async: FnMemsetAsync,
    memcpy_peer_async: FnMemcpyPeerAsync,
    mem_get_info: FnMemGetInfo,

    stream_create_with_flags: FnStreamCreateWithFlags,
    ext_stream_create_with_cu_mask: FnExtStreamCreateWithCuMask,
    stream_destroy: FnPtrIn,
    stream_synchronize: FnPtrIn,
    stream_query: FnPtrIn,
    stream_wait_event: FnStreamWaitEvent,
    stream_wait_value32: FnStreamWaitValue32,

    event_create_with_flags: FnEventCreateWithFlags,
    event_record: FnEventRecord,
    event_synchronize: FnPtrIn,
    event_query: FnPtrIn,
    event_elapsed_time: FnEventElapsedTime,
    event_destroy: FnPtrIn,

    module_load_data: FnModuleLoadData,
    module_get_function: FnModuleGetFunction,
    module_launch_kernel: FnModuleLaunchKernel,
    module_unload: FnPtrIn,

    stream_begin_capture: FnStreamBeginCapture,
    stream_end_capture: FnStreamEndCapture,
    graph_get_nodes: FnGraphGetNodes,
    graph_node_get_type: FnGraphNodeGetType,
    graph_instantiate: FnGraphInstantiate,
    graph_launch: FnGraphLaunch,
    graph_exec_destroy: FnPtrIn,
    graph_destroy: FnPtrIn,

    device_enable_peer_access: FnDeviceEnablePeerAccess,
    device_can_access_peer: FnDeviceCanAccessPeer,
    ipc_get_mem_handle: FnIpcGetMemHandle,
    ipc_open_mem_handle: FnIpcOpenMemHandle,
    import_external_memory: FnImportExternalMemory,
    external_memory_get_mapped_buffer: FnExternalMemoryGetMappedBuffer,
    ipc_close_mem_handle: FnPtrIn,

    get_error_string: FnGetErrorString,
    driver_get_version: FnI32Out,
    runtime_get_version: FnI32Out,

    /// Declared last so it is dropped last: the function pointers above are only valid while this
    /// is loaded. (Rust drops struct fields in declaration order.)
    _lib: Library,
}

impl std::fmt::Debug for Lib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lib({})", self.soname)
    }
}

macro_rules! load {
    ($lib:expr, $soname:expr, $name:literal) => {{
        let sym: libloading::Symbol<'_, _> = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
            .map_err(|e| Error::MissingSymbol {
            soname: $soname.to_string(),
            symbol: $name,
            reason: e.to_string(),
        })?;
        *sym
    }};
}

impl Lib {
    /// `dlopen` the HIP runtime and bind every entry point the engine uses.
    ///
    /// Binding is all-or-nothing on purpose: a partially bound runtime is how a route degrades
    /// silently, and there is no fallback soname list -- the caller names the library it wants.
    pub(crate) fn open(soname: &str) -> Result<Lib> {
        // SAFETY: `Library::new` runs the library's initialisers. libamdhip64 is the ROCm runtime;
        // loading it is exactly what a HIP program does at startup.
        let lib = unsafe { Library::new(soname) }.map_err(|e| Error::NotAvailable {
            soname: soname.to_string(),
            reason: e.to_string(),
        })?;

        let out = Lib {
            soname: soname.to_string(),

            get_device_count: load!(lib, soname, "hipGetDeviceCount"),
            set_device: load!(lib, soname, "hipSetDevice"),
            set_device_flags: load!(lib, soname, "hipSetDeviceFlags"),
            get_device_flags: load!(lib, soname, "hipGetDeviceFlags"),
            get_device: load!(lib, soname, "hipGetDevice"),
            device_get_attribute: load!(lib, soname, "hipDeviceGetAttribute"),
            device_get_name: load!(lib, soname, "hipDeviceGetName"),
            device_total_mem: load!(lib, soname, "hipDeviceTotalMem"),
            device_get_pci_bus_id: load!(lib, soname, "hipDeviceGetPCIBusId"),
            device_synchronize: load!(lib, soname, "hipDeviceSynchronize"),

            malloc: load!(lib, soname, "hipMalloc"),
            ext_malloc_with_flags: load!(lib, soname, "hipExtMallocWithFlags"),
            free: load!(lib, soname, "hipFree"),
            host_malloc: load!(lib, soname, "hipHostMalloc"),
            host_free: load!(lib, soname, "hipHostFree"),
            host_register: load!(lib, soname, "hipHostRegister"),
            host_unregister: load!(lib, soname, "hipHostUnregister"),
            host_get_device_pointer: load!(lib, soname, "hipHostGetDevicePointer"),
            memcpy_async: load!(lib, soname, "hipMemcpyAsync"),
            memset_async: load!(lib, soname, "hipMemsetAsync"),
            memcpy_peer_async: load!(lib, soname, "hipMemcpyPeerAsync"),
            mem_get_info: load!(lib, soname, "hipMemGetInfo"),

            stream_create_with_flags: load!(lib, soname, "hipStreamCreateWithFlags"),
            ext_stream_create_with_cu_mask: load!(lib, soname, "hipExtStreamCreateWithCUMask"),
            stream_destroy: load!(lib, soname, "hipStreamDestroy"),
            stream_synchronize: load!(lib, soname, "hipStreamSynchronize"),
            stream_query: load!(lib, soname, "hipStreamQuery"),
            stream_wait_event: load!(lib, soname, "hipStreamWaitEvent"),
            stream_wait_value32: load!(lib, soname, "hipStreamWaitValue32"),

            event_create_with_flags: load!(lib, soname, "hipEventCreateWithFlags"),
            event_record: load!(lib, soname, "hipEventRecord"),
            event_synchronize: load!(lib, soname, "hipEventSynchronize"),
            event_query: load!(lib, soname, "hipEventQuery"),
            event_elapsed_time: load!(lib, soname, "hipEventElapsedTime"),
            event_destroy: load!(lib, soname, "hipEventDestroy"),

            module_load_data: load!(lib, soname, "hipModuleLoadData"),
            module_get_function: load!(lib, soname, "hipModuleGetFunction"),
            module_launch_kernel: load!(lib, soname, "hipModuleLaunchKernel"),
            module_unload: load!(lib, soname, "hipModuleUnload"),

            stream_begin_capture: load!(lib, soname, "hipStreamBeginCapture"),
            stream_end_capture: load!(lib, soname, "hipStreamEndCapture"),
            graph_get_nodes: load!(lib, soname, "hipGraphGetNodes"),
            graph_node_get_type: load!(lib, soname, "hipGraphNodeGetType"),
            graph_instantiate: load!(lib, soname, "hipGraphInstantiate"),
            graph_launch: load!(lib, soname, "hipGraphLaunch"),
            graph_exec_destroy: load!(lib, soname, "hipGraphExecDestroy"),
            graph_destroy: load!(lib, soname, "hipGraphDestroy"),

            device_enable_peer_access: load!(lib, soname, "hipDeviceEnablePeerAccess"),
            device_can_access_peer: load!(lib, soname, "hipDeviceCanAccessPeer"),
            ipc_get_mem_handle: load!(lib, soname, "hipIpcGetMemHandle"),
            ipc_open_mem_handle: load!(lib, soname, "hipIpcOpenMemHandle"),
            import_external_memory: load!(lib, soname, "hipImportExternalMemory"),
            external_memory_get_mapped_buffer: load!(
                lib,
                soname,
                "hipExternalMemoryGetMappedBuffer"
            ),
            ipc_close_mem_handle: load!(lib, soname, "hipIpcCloseMemHandle"),

            get_error_string: load!(lib, soname, "hipGetErrorString"),
            driver_get_version: load!(lib, soname, "hipDriverGetVersion"),
            runtime_get_version: load!(lib, soname, "hipRuntimeGetVersion"),

            _lib: lib,
        };
        Ok(out)
    }

    /// The soname that was loaded.
    pub(crate) fn soname(&self) -> &str {
        &self.soname
    }

    /// Turn a raw status into a `Result`, attaching `hipGetErrorString`'s text.
    fn check(&self, op: &'static str, code: c_int) -> Result<()> {
        let status = HipError::from_raw(code);
        if status.is_success() {
            return Ok(());
        }
        Err(Error::hip(op, status, self.error_string(code)))
    }

    fn error_string(&self, code: c_int) -> Option<String> {
        // SAFETY: hipGetErrorString returns a pointer to a static string owned by the runtime, or
        // null for a code it does not know.
        let p = unsafe { (self.get_error_string)(code) };
        if p.is_null() {
            return None;
        }
        // SAFETY: non-null, NUL-terminated, static lifetime per the HIP contract.
        let s = unsafe { CStr::from_ptr(p) };
        Some(s.to_string_lossy().into_owned())
    }

    // ---- device ----------------------------------------------------------------------------- //

    pub(crate) fn device_count(&self, _c: &Ctx) -> Result<i32> {
        let mut n: c_int = 0;
        // SAFETY: `n` is a valid, writable c_int for the duration of the call.
        self.check("hipGetDeviceCount", unsafe {
            (self.get_device_count)(&mut n)
        })?;
        Ok(n)
    }

    pub(crate) fn set_device(&self, _c: &Ctx, device: i32) -> Result<()> {
        // SAFETY: plain scalar argument.
        self.check("hipSetDevice", unsafe { (self.set_device)(device) })
    }

    pub(crate) fn set_device_flags(&self, _c: &Ctx, flags: u32) -> Result<()> {
        // SAFETY: plain scalar argument.
        self.check("hipSetDeviceFlags", unsafe {
            (self.set_device_flags)(flags as c_uint)
        })
    }

    pub(crate) fn device_flags(&self, _c: &Ctx) -> Result<u32> {
        let mut f: c_uint = 0;
        // SAFETY: `f` is a valid, writable c_uint.
        self.check("hipGetDeviceFlags", unsafe {
            (self.get_device_flags)(&mut f)
        })?;
        Ok(f)
    }

    pub(crate) fn current_device(&self, _c: &Ctx) -> Result<i32> {
        let mut d: c_int = 0;
        // SAFETY: `d` is a valid, writable c_int.
        self.check("hipGetDevice", unsafe { (self.get_device)(&mut d) })?;
        Ok(d)
    }

    pub(crate) fn device_attribute(&self, _c: &Ctx, device: i32, attr: i32) -> Result<i32> {
        let mut v: c_int = 0;
        // SAFETY: `v` is a valid, writable c_int; `attr` and `device` are scalars.
        self.check("hipDeviceGetAttribute", unsafe {
            (self.device_get_attribute)(&mut v, attr, device)
        })?;
        Ok(v)
    }

    pub(crate) fn device_name(&self, _c: &Ctx, device: i32) -> Result<String> {
        let mut buf = [0 as c_char; 256];
        // SAFETY: `buf` is 256 writable bytes and the length passed matches it.
        self.check("hipDeviceGetName", unsafe {
            (self.device_get_name)(buf.as_mut_ptr(), buf.len() as c_int, device)
        })?;
        Ok(cstr_to_string(&buf))
    }

    pub(crate) fn device_total_mem(&self, _c: &Ctx, device: i32) -> Result<u64> {
        let mut bytes: usize = 0;
        // SAFETY: `bytes` is a valid, writable usize.
        self.check("hipDeviceTotalMem", unsafe {
            (self.device_total_mem)(&mut bytes, device)
        })?;
        Ok(bytes as u64)
    }

    pub(crate) fn device_pci_bus_id(&self, _c: &Ctx, device: i32) -> Result<String> {
        let mut buf = [0 as c_char; 64];
        // SAFETY: `buf` is 64 writable bytes and the length passed matches it.
        self.check("hipDeviceGetPCIBusId", unsafe {
            (self.device_get_pci_bus_id)(buf.as_mut_ptr(), buf.len() as c_int, device)
        })?;
        Ok(cstr_to_string(&buf))
    }

    /// `hipDeviceSynchronize`: block until every stream on the current device is idle.
    ///
    /// A blunt instrument, kept for teardown and for the end of a bulk transfer; the step loop
    /// synchronises a stream or an event, never the device.
    pub(crate) fn device_synchronize(&self, _c: &Ctx) -> Result<()> {
        // SAFETY: no arguments, and the call is valid on any thread that has a HIP context -- which
        // this one does by construction.
        self.check("hipDeviceSynchronize", unsafe {
            (self.device_synchronize)()
        })
    }

    pub(crate) fn can_access_peer(&self, _c: &Ctx, device: i32, peer: i32) -> Result<bool> {
        let mut v: c_int = 0;
        // SAFETY: `v` is a valid, writable c_int.
        self.check("hipDeviceCanAccessPeer", unsafe {
            (self.device_can_access_peer)(&mut v, device, peer)
        })?;
        Ok(v != 0)
    }

    pub(crate) fn enable_peer_access(&self, _c: &Ctx, peer: i32) -> Result<()> {
        // SAFETY: plain scalar arguments. Applies to the *current* device.
        self.check("hipDeviceEnablePeerAccess", unsafe {
            (self.device_enable_peer_access)(peer, 0)
        })
    }

    pub(crate) fn driver_version(&self, _c: &Ctx) -> Result<i32> {
        let mut v: c_int = 0;
        // SAFETY: `v` is a valid, writable c_int.
        self.check("hipDriverGetVersion", unsafe {
            (self.driver_get_version)(&mut v)
        })?;
        Ok(v)
    }

    pub(crate) fn runtime_version(&self, _c: &Ctx) -> Result<i32> {
        let mut v: c_int = 0;
        // SAFETY: `v` is a valid, writable c_int.
        self.check("hipRuntimeGetVersion", unsafe {
            (self.runtime_get_version)(&mut v)
        })?;
        Ok(v)
    }

    // ---- memory ----------------------------------------------------------------------------- //

    pub(crate) fn mem_get_info(&self, _c: &Ctx) -> Result<(u64, u64)> {
        let (mut free, mut total): (usize, usize) = (0, 0);
        // SAFETY: both out-params are valid, writable usizes.
        self.check("hipMemGetInfo", unsafe {
            (self.mem_get_info)(&mut free, &mut total)
        })?;
        Ok((free as u64, total as u64))
    }

    pub(crate) fn malloc(&self, _c: &Ctx, bytes: u64) -> Result<*mut c_void> {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: `p` is a valid, writable pointer slot.
        self.check("hipMalloc", unsafe {
            (self.malloc)(&mut p, bytes as usize)
        })?;
        Ok(p)
    }

    pub(crate) fn ext_malloc_with_flags(
        &self,
        _c: &Ctx,
        bytes: u64,
        flags: MallocFlags,
    ) -> Result<*mut c_void> {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: `p` is a valid, writable pointer slot; `flags` is one of the four values the
        // runtime documents, by construction of the enum.
        self.check("hipExtMallocWithFlags", unsafe {
            (self.ext_malloc_with_flags)(&mut p, bytes as usize, flags as c_uint)
        })?;
        Ok(p)
    }

    /// # Safety
    /// `p` must be a pointer returned by [`Lib::malloc`] on this runtime and not yet freed.
    pub(crate) unsafe fn free(&self, _c: &Ctx, p: *mut c_void) -> Result<()> {
        self.check("hipFree", unsafe { (self.free)(p) })
    }

    pub(crate) fn host_malloc(&self, _c: &Ctx, bytes: u64, flags: u32) -> Result<*mut c_void> {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: `p` is a valid, writable pointer slot.
        self.check("hipHostMalloc", unsafe {
            (self.host_malloc)(&mut p, bytes as usize, flags)
        })?;
        Ok(p)
    }

    /// `hipHostGetDevicePointer`: the device-visible address of a pinned allocation (FN8: the
    /// expert tier's RAM slots are read by the MoE kernel over UVA at this address).
    ///
    /// # Safety
    /// `p` must be a pointer returned by [`Lib::host_malloc`] on this runtime and not yet freed.
    pub(crate) unsafe fn host_get_device_pointer(
        &self,
        _c: &Ctx,
        p: *mut c_void,
    ) -> Result<*mut c_void> {
        let mut d: *mut c_void = std::ptr::null_mut();
        self.check("hipHostGetDevicePointer", unsafe {
            (self.host_get_device_pointer)(&mut d, p, 0)
        })?;
        Ok(d)
    }

    /// FN106d: `hipHostRegister` -- page-lock (and map for the devices) host memory this process
    /// mapped itself (the 2 MiB-page pinned form, [`crate::PinnedPages::Huge`]).
    ///
    /// # Safety
    /// `p .. p + bytes` must be a live mapping of this process that stays mapped until
    /// [`Lib::host_unregister`] returned for it.
    pub(crate) unsafe fn host_register(
        &self,
        _c: &Ctx,
        p: *mut c_void,
        bytes: u64,
        flags: u32,
    ) -> Result<()> {
        self.check("hipHostRegister", unsafe {
            (self.host_register)(p, bytes as usize, flags)
        })
    }

    /// FN106d: `hipHostUnregister` -- the runtime lets go of a registered range; the CALLER unmaps
    /// it afterwards (registered memory is never `hipHostFree`d: the unregister-before-unmap law).
    ///
    /// # Safety
    /// `p` must be a base passed to [`Lib::host_register`] on this runtime and not yet unregistered.
    pub(crate) unsafe fn host_unregister(&self, _c: &Ctx, p: *mut c_void) -> Result<()> {
        self.check("hipHostUnregister", unsafe { (self.host_unregister)(p) })
    }

    /// # Safety
    /// `p` must be a pointer returned by [`Lib::host_malloc`] on this runtime and not yet freed.
    pub(crate) unsafe fn host_free(&self, _c: &Ctx, p: *mut c_void) -> Result<()> {
        self.check("hipHostFree", unsafe { (self.host_free)(p) })
    }

    /// # Safety
    /// `dst` and `src` must be valid for `bytes` in the address spaces implied by `kind`, and must
    /// stay alive until the copy completes on `stream`.
    pub(crate) unsafe fn memcpy_async(
        &self,
        _c: &Ctx,
        dst: *mut c_void,
        src: *const c_void,
        bytes: u64,
        kind: MemcpyKind,
        stream: *mut c_void,
    ) -> Result<()> {
        self.check("hipMemcpyAsync", unsafe {
            (self.memcpy_async)(dst, src, bytes as usize, kind as c_int, stream)
        })
    }

    /// # Safety
    /// `dst` must be a live allocation of at least `bytes` on `dst_device`, `src` the same on
    /// `src_device`, and both must stay alive until the copy completes on `stream`.
    // The argument list is `hipMemcpyPeerAsync`'s; shortening it here would mean inventing a struct
    // that exists nowhere in HIP and re-deriving the mapping at every call site.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn memcpy_peer_async(
        &self,
        _c: &Ctx,
        dst: *mut c_void,
        dst_device: i32,
        src: *const c_void,
        src_device: i32,
        bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        self.check("hipMemcpyPeerAsync", unsafe {
            (self.memcpy_peer_async)(dst, dst_device, src, src_device, bytes as usize, stream)
        })
    }

    /// # Safety
    /// `dst` must be a device allocation valid for `bytes`, alive until the fill completes.
    pub(crate) unsafe fn memset_async(
        &self,
        _c: &Ctx,
        dst: *mut c_void,
        value: i32,
        bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        self.check("hipMemsetAsync", unsafe {
            (self.memset_async)(dst, value, bytes as usize, stream)
        })
    }

    // ---- streams and events ------------------------------------------------------------------ //

    pub(crate) fn stream_create(&self, _c: &Ctx, flags: StreamFlags) -> Result<*mut c_void> {
        let mut s: *mut c_void = std::ptr::null_mut();
        // SAFETY: `s` is a valid, writable pointer slot.
        self.check("hipStreamCreateWithFlags", unsafe {
            (self.stream_create_with_flags)(&mut s, flags as c_uint)
        })?;
        Ok(s)
    }

    /// FN72: `hipExtStreamCreateWithCUMask` (a stream with a CU mask gets a hardware queue of its
    /// own: ROCclr never hands a masked stream a pooled queue another stream holds).
    pub(crate) fn stream_create_cu_mask(&self, _c: &Ctx, mask: &[u32]) -> Result<*mut c_void> {
        let mut s: *mut c_void = std::ptr::null_mut();
        let n = u32::try_from(mask.len())
            .map_err(|_| Error::Invalid(format!("a CU mask of {} words", mask.len())))?;
        // SAFETY: `s` is a valid, writable pointer slot; `mask` holds `n` words for the call.
        self.check("hipExtStreamCreateWithCUMask", unsafe {
            (self.ext_stream_create_with_cu_mask)(&mut s, n, mask.as_ptr())
        })?;
        Ok(s)
    }

    /// # Safety
    /// `s` must be a live stream from this runtime.
    /// `hipStreamQuery`: `Ok(true)` when everything on the stream has completed.
    ///
    /// `hipErrorNotReady` is the *expected* answer while work is outstanding, so it maps to
    /// `Ok(false)` rather than to an error. Everything else is an error.
    ///
    /// # Safety
    /// `stream` must be a live stream on this runtime.
    pub(crate) unsafe fn stream_query(&self, _c: &Ctx, stream: *mut c_void) -> Result<bool> {
        let rc = unsafe { (self.stream_query)(stream) };
        if crate::error::HipError::from_raw(rc) == crate::error::HipError::NotReady {
            return Ok(false);
        }
        self.check("hipStreamQuery", rc)?;
        Ok(true)
    }

    pub(crate) unsafe fn stream_destroy(&self, _c: &Ctx, s: *mut c_void) -> Result<()> {
        self.check("hipStreamDestroy", unsafe { (self.stream_destroy)(s) })
    }

    /// # Safety
    /// `s` must be a live stream from this runtime.
    pub(crate) unsafe fn stream_synchronize(&self, _c: &Ctx, s: *mut c_void) -> Result<()> {
        self.check("hipStreamSynchronize", unsafe {
            (self.stream_synchronize)(s)
        })
    }

    /// `hipStreamWaitEvent`: make `s` wait for `e` without blocking the host.
    ///
    /// The cross-stream dependency the upload path is built on: a consumer stream waits on the copy
    /// stream's event instead of the host waiting and then re-submitting.
    ///
    /// # Safety
    /// `s` and `e` must be live handles from this runtime.
    pub(crate) unsafe fn stream_wait_event(
        &self,
        _c: &Ctx,
        s: *mut c_void,
        e: *mut c_void,
    ) -> Result<()> {
        // `flags` is 0 (`hipEventWaitDefault`); the external-node form is a graph concern and is
        // deliberately not bound here.
        self.check("hipStreamWaitEvent", unsafe {
            (self.stream_wait_event)(s, e, 0)
        })
    }

    /// `hipStreamWaitValue32(s, ptr, value, hipStreamWaitValueGte, mask)`: the COMMAND PROCESSOR
    /// holds `s` until `*ptr & mask >= value` (P27-1d). No CU spins; the host does not block.
    ///
    /// # Safety
    /// `s` must be a live stream from this runtime and `ptr` a device-visible 4-byte word that
    /// outlives every command queued behind the wait.
    pub(crate) unsafe fn stream_wait_value32_gte(
        &self,
        _c: &Ctx,
        s: *mut c_void,
        ptr: *mut c_void,
        value: u32,
        mask: u32,
    ) -> Result<()> {
        // `hipStreamWaitValueGte` is 0x0 in hip_runtime_api.h.
        self.check("hipStreamWaitValue32", unsafe {
            (self.stream_wait_value32)(s, ptr, value, 0, mask)
        })
    }

    pub(crate) fn event_create(&self, _c: &Ctx, flags: EventFlags) -> Result<*mut c_void> {
        let mut e: *mut c_void = std::ptr::null_mut();
        // SAFETY: `e` is a valid, writable pointer slot.
        self.check("hipEventCreateWithFlags", unsafe {
            (self.event_create_with_flags)(&mut e, flags as c_uint)
        })?;
        Ok(e)
    }

    /// # Safety
    /// `e` and `s` must be live handles from this runtime.
    pub(crate) unsafe fn event_record(
        &self,
        _c: &Ctx,
        e: *mut c_void,
        s: *mut c_void,
    ) -> Result<()> {
        self.check("hipEventRecord", unsafe { (self.event_record)(e, s) })
    }

    /// # Safety
    /// `e` must be a live event from this runtime.
    pub(crate) unsafe fn event_synchronize(&self, _c: &Ctx, e: *mut c_void) -> Result<()> {
        self.check("hipEventSynchronize", unsafe {
            (self.event_synchronize)(e)
        })
    }

    /// `hipEventQuery`: has this event completed? `hipErrorNotReady` is the *answer* "no", not a
    /// failure, and is the only status this maps instead of raising.
    ///
    /// # Safety
    /// `e` must be a live event from this runtime.
    pub(crate) unsafe fn event_query(&self, _c: &Ctx, e: *mut c_void) -> Result<bool> {
        let code = unsafe { (self.event_query)(e) };
        if HipError::from_raw(code) == HipError::NotReady {
            return Ok(false);
        }
        self.check("hipEventQuery", code)?;
        Ok(true)
    }

    /// # Safety
    /// Both events must be live, recorded, and created with timing enabled.
    pub(crate) unsafe fn event_elapsed_ms(
        &self,
        _c: &Ctx,
        start: *mut c_void,
        end: *mut c_void,
    ) -> Result<f32> {
        let mut ms: f32 = 0.0;
        self.check("hipEventElapsedTime", unsafe {
            (self.event_elapsed_time)(&mut ms, start, end)
        })?;
        Ok(ms)
    }

    /// # Safety
    /// `e` must be a live event from this runtime.
    pub(crate) unsafe fn event_destroy(&self, _c: &Ctx, e: *mut c_void) -> Result<()> {
        self.check("hipEventDestroy", unsafe { (self.event_destroy)(e) })
    }

    // ---- modules ----------------------------------------------------------------------------- //

    /// # Safety
    /// `image` must point at a complete, valid code-object image that outlives the call.
    pub(crate) unsafe fn module_load_data(&self, _c: &Ctx, image: &[u8]) -> Result<*mut c_void> {
        let mut m: *mut c_void = std::ptr::null_mut();
        self.check("hipModuleLoadData", unsafe {
            (self.module_load_data)(&mut m, image.as_ptr().cast())
        })?;
        Ok(m)
    }

    /// # Safety
    /// `module` must be a live module from this runtime.
    pub(crate) unsafe fn module_get_function(
        &self,
        _c: &Ctx,
        module: *mut c_void,
        name: &CStr,
    ) -> Result<*mut c_void> {
        let mut f: *mut c_void = std::ptr::null_mut();
        self.check("hipModuleGetFunction", unsafe {
            (self.module_get_function)(&mut f, module, name.as_ptr())
        })?;
        Ok(f)
    }

    /// # Safety
    /// `module` must be a live module from this runtime, and no function obtained from it may be
    /// launched afterwards.
    pub(crate) unsafe fn module_unload(&self, _c: &Ctx, module: *mut c_void) -> Result<()> {
        self.check("hipModuleUnload", unsafe { (self.module_unload)(module) })
    }

    /// # Safety
    /// `f` must be a live function handle; `kernargs` must be the exact kernarg image the code
    /// object declares, as validated against its `saber_kernarg_desc`; and every device pointer in
    /// it must be live for the duration of the launch. A kernarg layout the host and the kernel
    /// disagree about is memory corruption, not an error.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn module_launch_kernel(
        &self,
        _c: &Ctx,
        f: *mut c_void,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_lds: u32,
        stream: *mut c_void,
        kernargs: &mut [u8],
    ) -> Result<()> {
        // HIP's "extra" mechanism: HIP_LAUNCH_PARAM_BUFFER_POINTER (1), the buffer,
        // HIP_LAUNCH_PARAM_BUFFER_SIZE (2), &size, HIP_LAUNCH_PARAM_END (3). It is the only way to
        // pass a pre-packed kernarg image rather than an array of per-argument pointers, and a
        // pre-packed image is exactly what the ABI's kernarg descriptor describes.
        const HIP_LAUNCH_PARAM_BUFFER_POINTER: usize = 1;
        const HIP_LAUNCH_PARAM_BUFFER_SIZE: usize = 2;
        const HIP_LAUNCH_PARAM_END: usize = 3;
        let mut size = kernargs.len();
        let mut extra: [*mut c_void; 5] = [
            HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut c_void,
            kernargs.as_mut_ptr().cast(),
            HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut c_void,
            (&mut size as *mut usize).cast(),
            HIP_LAUNCH_PARAM_END as *mut c_void,
        ];
        self.check("hipModuleLaunchKernel", unsafe {
            (self.module_launch_kernel)(
                f,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                dynamic_lds,
                stream,
                std::ptr::null_mut(),
                extra.as_mut_ptr(),
            )
        })
    }

    // ---- capture and graphs ------------------------------------------------------------------ //

    /// # Safety
    /// `s` must be a live stream from this runtime.
    pub(crate) unsafe fn stream_begin_capture(
        &self,
        _c: &Ctx,
        s: *mut c_void,
        mode: StreamCaptureMode,
    ) -> Result<()> {
        self.check("hipStreamBeginCapture", unsafe {
            (self.stream_begin_capture)(s, mode as c_int)
        })
    }

    /// # Safety
    /// `s` must be a live stream currently capturing.
    pub(crate) unsafe fn stream_end_capture(
        &self,
        _c: &Ctx,
        s: *mut c_void,
    ) -> Result<*mut c_void> {
        let mut g: *mut c_void = std::ptr::null_mut();
        self.check("hipStreamEndCapture", unsafe {
            (self.stream_end_capture)(s, &mut g)
        })?;
        Ok(g)
    }

    /// # Safety
    /// `graph` must be a live graph from this runtime.
    /// `hipGraphGetNodes` with a null array: how many nodes the capture produced.
    ///
    /// The count is the launch budget, measured rather than assumed. `CLAUDE.md` rule 7 says the
    /// count per bucket is declared statically and **asserted at capture**; this is the call that
    /// makes the assertion possible.
    ///
    /// # Safety
    /// `graph` must be a live graph on this runtime.
    pub(crate) unsafe fn graph_get_node_count(
        &self,
        _c: &Ctx,
        graph: *mut c_void,
    ) -> Result<usize> {
        let mut n: usize = 0;
        self.check("hipGraphGetNodes", unsafe {
            (self.graph_get_nodes)(graph, std::ptr::null_mut(), &mut n)
        })?;
        Ok(n)
    }

    /// Enumerate graph node kinds without launching or changing the graph.
    /// # Safety
    /// `graph` must remain alive and unmodified for the duration of this call.
    pub(crate) unsafe fn graph_node_types(
        &self,
        ctx: &Ctx,
        graph: *mut c_void,
    ) -> Result<Vec<i32>> {
        let mut count = unsafe { self.graph_get_node_count(ctx, graph) }?;
        let mut nodes = vec![std::ptr::null_mut(); count];
        self.check("hipGraphGetNodes", unsafe {
            (self.graph_get_nodes)(graph, nodes.as_mut_ptr(), &mut count)
        })?;
        if count != nodes.len() {
            return Err(Error::Invalid(
                "Graph node count changed during enumeration".into(),
            ));
        }
        nodes
            .into_iter()
            .map(|node| {
                let mut kind = -1;
                self.check("hipGraphNodeGetType", unsafe {
                    (self.graph_node_get_type)(node, &mut kind)
                })?;
                Ok(kind)
            })
            .collect()
    }

    pub(crate) unsafe fn graph_instantiate(
        &self,
        _c: &Ctx,
        graph: *mut c_void,
    ) -> Result<*mut c_void> {
        let mut exec: *mut c_void = std::ptr::null_mut();
        self.check("hipGraphInstantiate", unsafe {
            (self.graph_instantiate)(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        })?;
        Ok(exec)
    }

    /// # Safety
    /// `exec` and `stream` must be live handles from this runtime.
    pub(crate) unsafe fn graph_launch(
        &self,
        _c: &Ctx,
        exec: *mut c_void,
        stream: *mut c_void,
    ) -> Result<()> {
        self.check("hipGraphLaunch", unsafe {
            (self.graph_launch)(exec, stream)
        })
    }

    /// # Safety
    /// `exec` must be a live executable graph from this runtime.
    pub(crate) unsafe fn graph_exec_destroy(&self, _c: &Ctx, exec: *mut c_void) -> Result<()> {
        self.check("hipGraphExecDestroy", unsafe {
            (self.graph_exec_destroy)(exec)
        })
    }

    /// # Safety
    /// `graph` must be a live graph from this runtime.
    pub(crate) unsafe fn graph_destroy(&self, _c: &Ctx, graph: *mut c_void) -> Result<()> {
        self.check("hipGraphDestroy", unsafe { (self.graph_destroy)(graph) })
    }

    // ---- IPC --------------------------------------------------------------------------------- //

    /// # Safety
    /// `p` must be the base pointer of a live `hipMalloc` allocation.
    pub(crate) unsafe fn ipc_get_mem_handle(
        &self,
        _c: &Ctx,
        p: *mut c_void,
    ) -> Result<IpcMemHandle> {
        let mut h = IpcMemHandle::default();
        self.check("hipIpcGetMemHandle", unsafe {
            (self.ipc_get_mem_handle)(&mut h, p)
        })?;
        Ok(h)
    }

    /// # Safety
    /// `handle` must come from a peer process's [`Lib::ipc_get_mem_handle`] for a live allocation,
    /// and the resulting pointer is only valid while that allocation lives.
    pub(crate) unsafe fn ipc_open_mem_handle(
        &self,
        _c: &Ctx,
        handle: IpcMemHandle,
    ) -> Result<*mut c_void> {
        // hipIpcMemLazyEnablePeerAccess == 1; the only flag HIP defines here.
        const LAZY_ENABLE_PEER_ACCESS: c_uint = 1;
        let mut p: *mut c_void = std::ptr::null_mut();
        self.check("hipIpcOpenMemHandle", unsafe {
            (self.ipc_open_mem_handle)(&mut p, handle, LAZY_ENABLE_PEER_ACCESS)
        })?;
        Ok(p)
    }

    /// FN76: `hipImportExternalMemory` (opaque fd = a dma-buf; ROCclr maps it with
    /// `hsa_amd_interop_map_buffer`) + `hipExternalMemoryGetMappedBuffer` over its whole length.
    /// Returns the mapped device pointer (released by `hipFree`); the external-memory handle lives
    /// for the process.
    ///
    /// # Safety
    /// `fd` is an open dma-buf of at least `bytes` bytes; the runtime takes it over.
    pub(crate) unsafe fn import_dmabuf(
        &self,
        _c: &Ctx,
        fd: c_int,
        bytes: u64,
    ) -> Result<*mut c_void> {
        let desc = ExtMemHandleDesc {
            kind: EXT_MEM_OPAQUE_FD,
            pad0: 0,
            fd,
            pad1: 0,
            name: 0,
            size: bytes,
            flags: 0,
            reserved: [0; 16],
        };
        let mut ext: *mut c_void = std::ptr::null_mut();
        self.check("hipImportExternalMemory", unsafe {
            (self.import_external_memory)(&mut ext, &desc)
        })?;
        let bd = ExtMemBufferDesc {
            offset: 0,
            size: bytes,
            flags: 0,
            reserved: [0; 16],
        };
        let mut p: *mut c_void = std::ptr::null_mut();
        self.check("hipExternalMemoryGetMappedBuffer", unsafe {
            (self.external_memory_get_mapped_buffer)(&mut p, ext, &bd)
        })?;
        Ok(p)
    }

    /// `hipIpcCloseMemHandle`: unmap a peer's allocation this process imported.
    ///
    /// # Safety
    /// `p` must be an address returned by [`Lib::ipc_open_mem_handle`] on this runtime, not yet
    /// closed, with no kernel still dereferencing it.
    pub(crate) unsafe fn ipc_close_mem_handle(&self, _c: &Ctx, p: *mut c_void) -> Result<()> {
        self.check("hipIpcCloseMemHandle", unsafe {
            (self.ipc_close_mem_handle)(p)
        })
    }
}

fn cstr_to_string(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_a_nonexistent_runtime_is_not_available_not_a_panic() {
        let err = Lib::open("libsaber-no-such-runtime.so.999").unwrap_err();
        assert!(err.is_not_available(), "{err}");
        assert!(err.to_string().contains("libsaber-no-such-runtime"));
    }

    #[test]
    fn ipc_handle_is_sixty_four_bytes() {
        // hipIpcOpenMemHandle takes this BY VALUE; if the size is wrong the stack is wrong.
        assert_eq!(std::mem::size_of::<IpcMemHandle>(), 64);
        // FN76: hip_runtime_api.h's external-memory descriptors
        assert_eq!(std::mem::size_of::<ExtMemHandleDesc>(), 104);
        assert_eq!(std::mem::offset_of!(ExtMemHandleDesc, fd), 8);
        assert_eq!(std::mem::offset_of!(ExtMemHandleDesc, size), 24);
        assert_eq!(std::mem::size_of::<ExtMemBufferDesc>(), 88);
        assert_eq!(std::mem::align_of::<IpcMemHandle>(), 1);
    }

    #[test]
    fn enum_reprs_match_hip() {
        assert_eq!(MemcpyKind::HostToDevice as i32, 1);
        assert_eq!(MemcpyKind::DeviceToHost as i32, 2);
        assert_eq!(MemcpyKind::DeviceToDevice as i32, 3);
        assert_eq!(StreamFlags::NonBlocking as u32, 1);
        assert_eq!(EventFlags::DisableTiming as u32, 2);
        assert_eq!(StreamCaptureMode::ThreadLocal as i32, 1);
    }

    #[test]
    fn cstr_to_string_stops_at_nul() {
        let mut buf = [0 as c_char; 8];
        buf[0] = b'g' as c_char;
        buf[1] = b'f' as c_char;
        buf[2] = b'x' as c_char;
        buf[3] = 0;
        buf[4] = b'Z' as c_char;
        assert_eq!(cstr_to_string(&buf), "gfx");
    }
}
