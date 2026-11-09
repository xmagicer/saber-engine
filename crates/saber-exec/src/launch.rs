//! Turning a [`LaunchDescriptor`] into a dispatch: pack the kernarg image, assign the workspaces
//! by kind, launch, and assert the declared launch count.
//!
//! `saber-kernels` decides *what* to run — `resolve(op, shape, rules)` is pure and has no branch
//! that is not a table row. This module does exactly what the descriptor says and re-derives
//! nothing: not a tile, not a split-K rule, not a kernarg offset. If the descriptor is wrong the
//! launch is wrong, which is the property the bit-exact gate tests.
//!
//! Three things are policy here and are stated as such:
//!
//! * **Workspaces have kinds, not just byte counts.** `Scratch` is aliasable within a stream — the
//!   split-K partial slab is dead between launches, so one arena serves every op on that stream.
//!   `Persistent` is not: the split-K block counter must be zeroed exactly once and survive graph
//!   replay (the last block of a split resets it in place, so a replay that re-zeroed it would be
//!   correct only by luck, and one that reallocated it would corrupt a captured graph). It is
//!   therefore keyed on `(plugin, op, shape)` and allocated once. `IpcPeerVisible` is refused with
//!   a counter rather than silently degraded — invariant "fail closed".
//! * **One `Launcher` serves one stream.** Aliasing Scratch across streams would be a data race
//!   the type system cannot see, and `saber_hip::Stream` exposes no identity to key an arena on.
//! * **The launch count is asserted, not observed.** `CLAUDE.md` rule 7: launch count per bucket is
//!   declared statically and asserted at capture. A descriptor claiming `n_launches = 1` that
//!   issues two dispatches fails here, not in a launch census three phases later.

use std::collections::BTreeMap;

use saber_hip::{DeviceAddr, DevicePtr, HipError, HipThread, ImportedPtr, MallocFlags, Stream};
use saber_kernels::manifest::KernelDecl;
use saber_kernels::rules::{ArgValue, Shape, Slot, WorkspaceKind, WorkspaceReq};
use saber_kernels::LaunchDescriptor;

use crate::counters::LaunchCounters;
use crate::error::{Error, Result};
use crate::loader::FunctionTable;
use crate::peer::{ProtocolDigest, RankPeer};
use std::sync::Arc;

/// What the launcher needs that is not in the descriptor: the HIP thread and the resolved symbols.
pub struct LaunchCtx<'a> {
    /// The one thread every HIP call runs on.
    pub hip: &'a HipThread,
    /// Entry points resolved at plugin load.
    pub functions: &'a FunctionTable<'a>,
}

impl<'a> LaunchCtx<'a> {
    /// A launch context over a loaded plugin set.
    pub fn new(hip: &'a HipThread, functions: &'a FunctionTable<'a>) -> Self {
        LaunchCtx { hip, functions }
    }
}

/// The caller's side of a descriptor: the device addresses its named buffers resolve to.
///
/// Workspaces are *not* bound here — the launcher owns those, by kind. A descriptor slot naming a
/// buffer that is neither bound nor a declared workspace is a refusal with a counter, because the
/// alternative is a kernel dereferencing whatever was in that kernarg slot.
#[derive(Debug, Clone)]
pub struct ArgBindings {
    shape: Shape,
    buffers: BTreeMap<String, DeviceAddr>,
}

impl ArgBindings {
    /// Bindings for a call at this shape. The shape is part of the Persistent workspace key.
    #[must_use]
    pub fn new(shape: Shape) -> Self {
        ArgBindings {
            shape,
            buffers: BTreeMap::new(),
        }
    }

    /// Bind one named buffer.
    pub fn bind(&mut self, name: &str, addr: DeviceAddr) -> &mut Self {
        self.buffers.insert(name.to_owned(), addr);
        self
    }

    /// Builder form of [`ArgBindings::bind`].
    #[must_use]
    pub fn with(mut self, name: &str, addr: DeviceAddr) -> Self {
        self.bind(name, addr);
        self
    }

    /// The address bound to `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<DeviceAddr> {
        self.buffers.get(name).copied()
    }

    /// The shape these bindings were built for.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.shape
    }
}

/// The key a Persistent workspace is tracked under: `(plugin, op, shape)` plus its name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PersistentKey {
    plugin: String,
    op: String,
    m: u32,
    n: u32,
    k: u32,
    name: String,
}

/// Every byte the host writes into a kernel's argument buffer, built from the descriptor alone.
///
/// Pure, and public, for two reasons. The first is that this is the function the bit-exact gate is
/// really testing — a kernarg layout the host and the kernel disagree about is memory corruption,
/// not an error — so it has to be testable without a GPU. The second is by-value structs: for
/// The reference's attention kernels the descriptor's slots are the struct's *fields*, laid over the
/// struct's byte range by `saber_kernels::abi::pack_fields_at`, and this loop writes them at their
/// absolute offsets exactly as it writes an ordinary scalar. There is no struct-specific branch
/// here, and that is the point: the descriptor has already done the layout arithmetic.
///
/// Bytes not covered by any slot stay zero. That is the ABI's rule for padding, and it is what
/// makes the 4 bytes of tail padding inside `SABERArgs` — real memory the host must leave alone —
/// deterministic rather than whatever the staging buffer last held.
pub fn pack_kernarg(
    desc: &LaunchDescriptor,
    args: &ArgBindings,
    workspaces: &BTreeMap<String, DeviceAddr>,
    counters: Option<&LaunchCounters>,
) -> Result<Vec<u8>> {
    let mut image = vec![0u8; desc.kernarg_bytes as usize];
    let mut covered: Vec<(u32, u32)> = Vec::with_capacity(desc.args.len());
    for slot in &desc.args {
        let (off, size) = (slot.offset as usize, slot.size as usize);
        if off + size > image.len() {
            return Err(Error::launch(
                &desc.kernel,
                format!(
                    "slot `{}` at offset {off} size {size} runs past the {}-byte kernarg image",
                    slot.name,
                    image.len()
                ),
            ));
        }
        // Two slots writing the same byte would make the image depend on descriptor order, which
        // is exactly the class of bug a by-value struct's field list could introduce.
        if let Some((o, s)) = covered
            .iter()
            .find(|(o, s)| (*o as usize) < off + size && off < (*o + *s) as usize)
        {
            return Err(Error::launch(
                &desc.kernel,
                format!(
                    "slot `{}` at offset {off} size {size} overlaps another slot at offset {o} \
                     size {s}",
                    slot.name
                ),
            ));
        }
        covered.push((slot.offset, slot.size));
        let bytes = slot_bytes(desc, slot, args, workspaces, counters)?;
        if bytes.len() != size {
            return Err(Error::launch(
                &desc.kernel,
                format!(
                    "slot `{}` declares size {size} but its value packs to {} bytes",
                    slot.name,
                    bytes.len()
                ),
            ));
        }
        image[off..off + size].copy_from_slice(&bytes);
    }
    Ok(image)
}

/// Every argument the manifest declares is covered by the descriptor's slots, and every slot lands
/// inside one of them.
///
/// A scalar argument is matched exactly. A **by-value struct** — the reference's `SABERArgs`, which the code
/// object reports as one 136-byte `by_value` argument — is instead *covered* by the field slots the
/// descriptor lays over its byte range. Comparing slot counts one for one, as this check used to,
/// would refuse every attention kernel: 22 descriptor slots against 2 declared arguments.
///
/// What is still checked, and is what actually matters: no descriptor slot may fall outside every
/// declared argument (that would be a write into padding or past the segment), and no declared
/// argument may be left entirely unwritten (that would be a kernel reading whatever the staging
/// buffer held).
fn cover_declared_args(
    desc: &LaunchDescriptor,
    decl: &KernelDecl,
) -> std::result::Result<(), String> {
    let mut written = vec![0u32; decl.arg.len()];
    for slot in &desc.args {
        let end = slot.offset + slot.size;
        let owner = decl
            .arg
            .iter()
            .position(|a| slot.offset >= a.offset && end <= a.offset + a.size);
        let Some(owner) = owner else {
            return Err(format!(
                "slot `{}` at (offset {}, size {}) is inside no argument the manifest declares; \
                 the manifest has {}",
                slot.name,
                slot.offset,
                slot.size,
                decl.arg
                    .iter()
                    .map(|a| format!("(offset {}, size {})", a.offset, a.size))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        };
        written[owner] += slot.size;
    }
    for (i, (a, w)) in decl.arg.iter().zip(&written).enumerate() {
        if *w == 0 {
            return Err(format!(
                "the manifest declares argument {i} at (offset {}, size {}) and the descriptor \
                 writes none of it",
                a.offset, a.size
            ));
        }
        if *w > a.size {
            return Err(format!(
                "the descriptor writes {w} B into argument {i}, which is {} B",
                a.size
            ));
        }
    }
    Ok(())
}

fn slot_bytes(
    desc: &LaunchDescriptor,
    slot: &Slot,
    args: &ArgBindings,
    workspaces: &BTreeMap<String, DeviceAddr>,
    counters: Option<&LaunchCounters>,
) -> Result<Vec<u8>> {
    Ok(match &slot.value {
        ArgValue::Buffer { buffer } => {
            let addr = args
                .get(buffer)
                .or_else(|| workspaces.get(buffer).copied())
                .ok_or_else(|| {
                    if let Some(c) = counters {
                        c.hit_refused_unbound_buffer();
                    }
                    Error::launch(
                        &desc.kernel,
                        format!(
                            "buffer `{buffer}` is neither bound by the caller nor declared as a \
                             workspace of this descriptor"
                        ),
                    )
                })?;
            addr.as_u64().to_le_bytes().to_vec()
        }
        // A pointer the chosen instantiation provably never dereferences. Writing an explicit
        // null is the descriptor's claim; a kernel that did touch it faults here rather than
        // corrupting somebody else's allocation.
        ArgValue::NullBuffer { .. } => DeviceAddr::NULL.as_u64().to_le_bytes().to_vec(),
        ArgValue::I32 { ival } => {
            let v = i32::try_from(*ival).map_err(|_| {
                Error::launch(
                    &desc.kernel,
                    format!("slot `{}` value {ival} does not fit in i32", slot.name),
                )
            })?;
            v.to_le_bytes().to_vec()
        }
        ArgValue::I64 { ival } => ival.to_le_bytes().to_vec(),
        ArgValue::F32 { fval } => fval.to_le_bytes().to_vec(),
    })
}

/// The plugin whose protocol regions a collective's `IpcPeerVisible` workspaces key onto.
///
/// P27-1d: the SPLIT rotated all-reduce (`saber_ar_wht_split`: the one-shot's push and reduce halves)
/// runs on the ONE-SHOT's regions -- the same scratch slots, per-block flags and counters, so a
/// split call and a one-shot call alternate on one protocol (P27-1b/1c's byte law, and the reason
/// the two ranks' slot parity cannot drift between forms). CMP27S: the WAVE-PARALLEL body
/// (`saber_ar_wht_wpar`, `--engine-ar-wpar`) and its race-check twin are the one-shot and its twin
/// with another work assignment -- the same slots, flags and counters -- so they key onto the
/// served objects' regions too (a call routed to either body advances the one sequence; no second
/// 2 x 84 MiB scratch). CMP27AE: the GEMM + wht6 fold (`mxfp4_gemm_decode_arw`, `--engine-ar-fuse`)
/// keys onto the same regions ([`protocol_op`] maps its op too) and keeps its OWN words in them
/// (counters / flags from `saber::AR_FUSE_WORD0`, payload at `saber::ar_fuse_off`), so it adds no
/// region (CMP27S g1's `engine.ipc` FATAL) and never shares a word with a one-shot. Every other
/// plugin keys as itself.
#[must_use]
pub fn protocol_family(plugin: &str) -> &str {
    match plugin {
        "saber_ar_wht_split" | "saber_ar_wht_wpar" | "mxfp4_gemm_decode_arw" => "saber_ar_wht",
        "saber_ar_wht_wpar_chk" => "saber_ar_wht_chk",
        p => p,
    }
}

/// The op a collective's `IpcPeerVisible` workspaces key under: CMP27AE's GEMM + wht6 fold keys as
/// the rotated reduce it carries (`allreduce_wht`), so [`protocol_family`] + this name ONE region
/// set; every other descriptor keys as its own op.
#[must_use]
pub fn protocol_op<'a>(plugin: &str, op: &'a str) -> &'a str {
    if plugin == "mxfp4_gemm_decode_arw" {
        "allreduce_wht"
    } else {
        op
    }
}

/// `ar_peer_scratch` -> `ar_scratch`: the region a `_peer_` name is the peer's view of.
///
/// One substring, applied once. Not a heuristic over the whole name: `peer_` has to appear as a
/// path component between underscores, so `ar_peer_scratch` matches and a hypothetical
/// `ar_peering` does not.
fn peer_base_name(name: &str) -> Option<String> {
    name.find("_peer_")
        .map(|i| format!("{}_{}", &name[..i], &name[i + "_peer_".len()..]))
}

/// One stream's dispatcher: the workspace arenas, the counters, and the kernarg staging buffer.
pub struct Launcher {
    rccl: Option<std::rc::Rc<saber_hip::RcclComm>>,
    host_ar: bool,
    stream_label: String,
    counters: Arc<LaunchCounters>,
    /// Aliasable slabs, indexed by the order a descriptor asks for them. Index `i` is handed to
    /// the `i`-th Scratch workspace of *every* descriptor: that is what "aliasable" means.
    scratch: Vec<DevicePtr>,
    persistent: BTreeMap<PersistentKey, DevicePtr>,
    /// T8: the Persistent regions allocated `zero_once` -- protocol state (split-K counters,
    /// barriers, tile counters) whose rest value is its boot zeros. The boot census reads them; the
    /// boot poison leaves them alone.
    zero_once: std::collections::BTreeSet<PersistentKey>,
    /// Peer-visible regions, keyed like Persistent ones. Allocated ONCE and never resized: their
    /// addresses are exchanged with the other rank and baked into its kernarg images, so a
    /// reallocation here is corruption over there.
    ipc: BTreeMap<PersistentKey, IpcRegion>,
    /// The other rank, when this launcher serves a collective. `None` means no route it launches
    /// may declare an `IpcPeerVisible` workspace.
    peer: Option<Arc<RankPeer>>,
    kernarg: Vec<u8>,
    /// FN63: when armed, every dispatch's launch parameters and packed kernarg image, in issue
    /// order -- the node list a non-HIP backend replays over the same buffers. Off by default; a
    /// diagnostic, never on a served path.
    recorder: Option<Vec<DispatchRecord>>,
}

/// FN63: one dispatch as the launcher issued it.
#[derive(Debug, Clone)]
pub struct DispatchRecord {
    /// Code object (plugin) stem.
    pub plugin: String,
    /// Kernel name.
    pub kernel: String,
    /// The op that issued it.
    pub op: String,
    /// Grid in work-groups.
    pub grid: [u32; 3],
    /// Work-group size.
    pub block: [u32; 3],
    /// Dynamic LDS bytes.
    pub shared_bytes: u32,
    /// The explicit kernarg image, exactly as passed to `hipModuleLaunchKernel`.
    pub kernarg: Vec<u8>,
    /// Byte offsets of the 8-byte pointer slots in `kernarg` (buffers and workspaces).
    pub pointer_slots: Vec<u32>,
}

/// One peer-visible region: what this rank owns, or what it borrowed from the other rank.
enum IpcRegion {
    /// Allocated here. Fine-grained when the peer reads it, because a coarse-grained store the
    /// peer cannot snoop turns a push into a hang. Kept alive for the process: the peer holds its
    /// address.
    Mine {
        ptr: DevicePtr,
        /// How many times this rank has zeroed it. Both ranks must agree, because `slot = seq & 1`
        /// picks which half of the double-buffered scratch they use.
        epoch: u64,
    },
    /// Opened from the peer's handle. NOT owned: `hipIpcCloseMemHandle` on drop, never `hipFree`.
    Peer { ptr: ImportedPtr },
    Host {
        ptr: saber_hip::SharedHost,
        mine: bool,
        epoch: u64,
    },
}

impl IpcRegion {
    fn addr(&self) -> DeviceAddr {
        match self {
            IpcRegion::Mine { ptr, .. } => ptr.addr(),
            IpcRegion::Peer { ptr } => ptr.addr(),
            IpcRegion::Host { ptr, .. } => ptr.addr(),
        }
    }

    fn bytes(&self) -> u64 {
        match self {
            IpcRegion::Mine { ptr, .. } => ptr.len(),
            // The peer's region is mapped, not owned; its size is the peer's business.
            IpcRegion::Peer { .. } | IpcRegion::Host { .. } => 0,
        }
    }
}

/// What one `launch` did, for the census and for tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchReport {
    /// Dispatches actually issued. Asserted equal to the descriptor's `n_launches`.
    pub launches: u32,
    /// Bytes of Scratch newly allocated by this call (0 once the arena is warm).
    pub scratch_allocated: u64,
    /// Bytes of Persistent newly allocated by this call (0 after the first call at this key).
    pub persistent_allocated: u64,
}

/// T8: one region of the boot census ([`Launcher::rest_census`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestRegion {
    /// `persistent:<plugin>/<op>@m,n,k:<name>` or `ipc:<plugin>/<op>@m,n,k:<name>`.
    pub label: String,
    /// Bytes read back.
    pub bytes: u64,
    /// Non-zero 32-bit words among them.
    pub nonzero: u64,
    /// FNV-1a 64 of the bytes.
    pub digest: u64,
}

/// FNV-1a 64 of `b` (the census' digest; the host's own, no device code).
#[must_use]
pub fn fnv1a64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in b {
        h ^= u64::from(*x);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl Launcher {
    /// **T8 (a diagnostic, `--poison-boot ws|scratch`)**: fill this launcher's non-protocol
    /// Persistent regions (every one allocated WITHOUT `zero_once`) and/or its Scratch slabs with
    /// `byte`, on `stream`. The zero-once regions (counters, barriers) are protocol state whose
    /// rest value is zero by design; [`Launcher::rest_census`] reads those instead. Returns how many
    /// buffers were filled.
    ///
    /// # Errors
    /// A memset that fails.
    pub fn poison(
        &self,
        ctx: &LaunchCtx<'_>,
        stream: &Stream,
        byte: u8,
        persistent: bool,
        scratch: bool,
    ) -> Result<u32> {
        let mut n = 0u32;
        if persistent {
            for (k, p) in &self.persistent {
                if self.zero_once.contains(k) {
                    continue;
                }
                ctx.hip.memset(p, byte, stream)?;
                n += 1;
            }
        }
        if scratch {
            for p in &self.scratch {
                ctx.hip.memset(p, byte, stream)?;
                n += 1;
            }
        }
        Ok(n)
    }

    /// **T8: the protocol state at rest** (a diagnostic, read at the end of boot under the shadow):
    /// every zero-once Persistent region and every IPC region this rank OWNS of at most `max_bytes`,
    /// read back whole (blocking, once, before any request). A clean boot leaves every one of them
    /// at the same bytes (a split-K counter at zero); a region a first request reads before any
    /// launch of this boot restored it is named by a digest that moves between boots.
    ///
    /// # Errors
    /// A read-back that fails.
    pub fn rest_census(
        &self,
        ctx: &LaunchCtx<'_>,
        stream: &Stream,
        max_bytes: u64,
    ) -> Result<Vec<RestRegion>> {
        let label = |kind: &str, k: &PersistentKey| {
            format!(
                "{kind}:{}/{}@{},{},{}:{}",
                k.plugin, k.op, k.m, k.n, k.k, k.name
            )
        };
        let mut out = Vec::new();
        let mut read = |lab: String, p: &DevicePtr| -> Result<()> {
            let bytes = p.len().min(max_bytes);
            let mut buf = vec![0u8; bytes as usize];
            ctx.hip.memcpy_d2h_blocking(&mut buf, p, stream)?;
            let nonzero = buf.chunks(4).filter(|w| w.iter().any(|b| *b != 0)).count() as u64;
            out.push(RestRegion {
                label: lab,
                bytes,
                nonzero,
                digest: fnv1a64(&buf),
            });
            Ok(())
        };
        for k in &self.zero_once {
            if let Some(p) = self.persistent.get(k) {
                read(label("persistent", k), p)?;
            }
        }
        for (k, r) in &self.ipc {
            if let IpcRegion::Mine { ptr, .. } = r {
                if ptr.len() <= max_bytes {
                    read(label("ipc", k), ptr)?;
                }
            }
        }
        drop(read);
        for (k, r) in &self.ipc {
            if let IpcRegion::Host {
                ptr, mine: true, ..
            } = r
            {
                if ptr.len() <= max_bytes {
                    let mut buf = vec![0u8; ptr.len() as usize];
                    ctx.hip
                        .memcpy_d2h_addr_blocking(&mut buf, ptr.addr(), stream)?;
                    out.push(RestRegion {
                        label: label("host-ipc", k),
                        bytes: ptr.len(),
                        nonzero: buf.chunks(4).filter(|w| w.iter().any(|b| *b != 0)).count() as u64,
                        digest: fnv1a64(&buf),
                    });
                }
            }
        }
        Ok(out)
    }

    /// A launcher for one stream. `label` is only for messages and the census.
    #[must_use]
    pub fn new(label: impl Into<String>, counters: &Arc<LaunchCounters>) -> Self {
        Launcher {
            rccl: None,
            host_ar: false,
            stream_label: label.into(),
            counters: Arc::clone(counters),
            scratch: Vec::new(),
            persistent: BTreeMap::new(),
            zero_once: std::collections::BTreeSet::new(),
            ipc: BTreeMap::new(),
            peer: None,
            kernarg: Vec::new(),
            recorder: None,
        }
    }

    /// FN63: arm (`true`, clearing any earlier records) or disarm the dispatch recorder.
    pub fn record_dispatches(&mut self, on: bool) {
        self.recorder = on.then(Vec::new);
    }

    /// FN63: the records since the recorder was armed (the recorder stays armed, now empty).
    pub fn take_dispatch_records(&mut self) -> Vec<DispatchRecord> {
        self.recorder
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Give this launcher the other rank, enabling `IpcPeerVisible` workspaces.
    ///
    /// Without it, a descriptor that declares one is refused — which is the right answer for a
    /// single-rank engine, and is what every route did before the all-reduce landed.
    #[must_use]
    pub fn with_peer(mut self, peer: Arc<RankPeer>) -> Self {
        self.peer = Some(peer);
        self
    }

    /// Bind the HIP-thread-owned collective implementation selected by the resolved plan.
    #[must_use]
    pub fn with_rccl(mut self, comm: std::rc::Rc<saber_hip::RcclComm>) -> Self {
        self.rccl = Some(comm);
        self
    }

    /// Use registered shared host pages for native exact collectives.
    #[must_use]
    pub fn with_host_ar(mut self) -> Self {
        self.host_ar = true;
        self
    }

    fn rccl_args(
        &self,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
    ) -> Result<(DeviceAddr, DeviceAddr, usize)> {
        use saber_kernels::saber::{RCCL_BF16_KERNEL, RCCL_PLUGIN};
        if desc.plugin != RCCL_PLUGIN
            || desc.kernel != RCCL_BF16_KERNEL
            || desc.op != "allreduce_rccl_bf16"
            || !desc.workspaces.is_empty()
            || desc.args.len() != 3
        {
            return Err(Error::Refused(
                "Invalid external RCCL collective descriptor".into(),
            ));
        }
        if self.rccl.is_none() {
            return Err(Error::Refused(
                "RCCL descriptor has no initialized communicator".into(),
            ));
        }
        let buffer = |name: &str| -> Result<DeviceAddr> {
            match desc.arg(name).map(|s| &s.value) {
                Some(ArgValue::Buffer { buffer }) => args
                    .get(buffer)
                    .filter(|a| *a != DeviceAddr::NULL)
                    .ok_or_else(|| Error::Refused(format!("RCCL buffer {buffer} is unbound"))),
                _ => Err(Error::Refused(format!(
                    "RCCL descriptor has no buffer argument {name}"
                ))),
            }
        };
        let count = match desc.arg("count").map(|s| &s.value) {
            Some(ArgValue::I32 { ival }) if *ival > 0 => {
                usize::try_from(*ival).map_err(|_| Error::Refused("RCCL count overflow".into()))?
            }
            _ => {
                return Err(Error::Refused(
                    "RCCL descriptor has no positive element count".into(),
                ))
            }
        };
        Ok((buffer("in")?, buffer("out")?, count))
    }

    /// The other rank, if this launcher serves a collective.
    #[must_use]
    pub fn peer(&self) -> Option<&Arc<RankPeer>> {
        self.peer.as_ref()
    }

    /// The stream this launcher's Scratch arena belongs to.
    #[must_use]
    pub fn stream_label(&self) -> &str {
        &self.stream_label
    }

    /// Bytes currently held by the two arenas: `(scratch, persistent)`.
    #[must_use]
    pub fn arena_bytes(&self) -> (u64, u64) {
        (
            self.scratch.iter().map(DevicePtr::len).sum(),
            self.persistent.values().map(DevicePtr::len).sum(),
        )
    }

    /// Issue one op: every dispatch its head descriptor declares.
    ///
    /// **This is where rule 7 lives.** `n_launches` is a property of the *op*, not of one
    /// dispatch: the reference's attention decode declares 2 because the split-KV combine is a second
    /// kernel over the partials the first one wrote, and that second kernel has its own
    /// descriptor. So the budget is asserted here, across the whole sequence, and a caller that
    /// hands over one descriptor for a two-launch op is refused by name rather than quietly
    /// issuing half an op.
    ///
    /// `descs[0]` is the head: its `n_launches` is the declaration the rest are measured against.
    pub fn launch_op(
        &mut self,
        ctx: &LaunchCtx<'_>,
        descs: &[&LaunchDescriptor],
        args: &ArgBindings,
        stream: &Stream,
    ) -> Result<LaunchReport> {
        let Some(head) = descs.first() else {
            return Err(Error::Refused(
                "launch_op was handed no descriptors at all".to_owned(),
            ));
        };
        let declared = head.n_launches;
        let mut report = LaunchReport {
            launches: 0,
            scratch_allocated: 0,
            persistent_allocated: 0,
        };
        for desc in descs {
            let one = self.dispatch(ctx, desc, args, stream)?;
            report.launches += one.launches;
            report.scratch_allocated += one.scratch_allocated;
            report.persistent_allocated += one.persistent_allocated;
        }
        if report.launches != declared {
            self.counters.hit_refused_launch_count();
            return Err(Error::launch(
                &head.kernel,
                format!(
                    "op `{}` declares n_launches={declared} but {} dispatch(es) were issued; the \
                     descriptors handed over were [{}]",
                    head.op,
                    report.launches,
                    descs
                        .iter()
                        .map(|d| d.kernel.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
        Ok(report)
    }

    /// Pack, assign, launch — one op that is one dispatch.
    ///
    /// Every byte of the kernarg image comes from the descriptor: the offsets and sizes are the
    /// descriptor's, the pointer values come from `args` (or from a workspace this launcher owns),
    /// and the scalars are already bound in the descriptor's slots. Nothing is re-derived.
    ///
    /// An op that declares more than one launch is refused here; use [`Launcher::launch_op`] with
    /// its whole descriptor sequence.
    pub fn launch(
        &mut self,
        ctx: &LaunchCtx<'_>,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
        stream: &Stream,
    ) -> Result<LaunchReport> {
        self.launch_op(ctx, &[desc], args, stream)
    }

    /// Issue exactly ONE dispatch from a descriptor, without the op-level launch budget.
    ///
    /// For a harness that replays individual dispatches rather than whole ops — the bit-exact
    /// gate, whose reference capture ran each kernel on its own and dumped its own buffers, so
    /// there is no second dispatch to issue and nothing for [`Launcher::launch_op`] to count. The
    /// budget is still declared on the descriptor and is still asserted wherever an op is run as
    /// an op; this route is counted separately so that "a dispatch escaped its budget" stays a
    /// number rather than an assumption.
    pub fn dispatch_one(
        &mut self,
        ctx: &LaunchCtx<'_>,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
        stream: &Stream,
    ) -> Result<LaunchReport> {
        if desc.n_launches != 1 {
            self.counters.hit_dispatched_unbudgeted();
        }
        self.dispatch(ctx, desc, args, stream)
    }

    fn dispatch(
        &mut self,
        ctx: &LaunchCtx<'_>,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
        stream: &Stream,
    ) -> Result<LaunchReport> {
        if desc.plugin == saber_kernels::saber::RCCL_PLUGIN {
            if self.recorder.is_some() {
                return Err(Error::Refused(
                    "Dispatch recording cannot replay external RCCL collectives".into(),
                ));
            }
            let (input, output, count) = self.rccl_args(desc, args)?;
            ctx.hip.rccl_all_reduce_bf16(
                self.rccl.as_ref().expect("checked above"),
                input,
                output,
                count,
                stream,
            )?;
            self.counters.hit_launched();
            self.counters.hit_ar_plain();
            return Ok(LaunchReport {
                launches: 1,
                scratch_allocated: 0,
                persistent_allocated: 0,
            });
        }
        let Some((function, decl)) = ctx.functions.get(&desc.plugin, &desc.kernel) else {
            return Err(Error::NotFound(format!(
                "`{}` has no kernel `{}` loaded (plugin not loaded, or the symbol is not in it)",
                desc.plugin, desc.kernel
            )));
        };
        self.check_against_manifest(desc, decl)?;

        let mut report = LaunchReport {
            launches: 0,
            scratch_allocated: 0,
            persistent_allocated: 0,
        };
        let workspaces = self.assign_workspaces(ctx, desc, args, stream, &mut report)?;

        // ---- pack --------------------------------------------------------------------------
        self.kernarg = pack_kernarg(desc, args, &workspaces, Some(&self.counters))?;
        if let Some(rec) = self.recorder.as_mut() {
            rec.push(DispatchRecord {
                plugin: desc.plugin.clone(),
                kernel: desc.kernel.clone(),
                op: desc.op.clone(),
                grid: desc.grid,
                block: desc.block,
                shared_bytes: desc.shared_bytes,
                kernarg: self.kernarg.clone(),
                pointer_slots: desc
                    .args
                    .iter()
                    .filter(|a| a.size == 8 && matches!(a.value, ArgValue::Buffer { .. }))
                    .map(|a| a.offset)
                    .collect(),
            });
        }

        // ---- launch ------------------------------------------------------------------------
        ctx.hip.launch(
            function,
            desc.grid,
            desc.block,
            desc.shared_bytes,
            stream,
            &mut self.kernarg,
        )?;
        report.launches += 1;
        self.counters.hit_launched();
        // Which path a reduce took, counted ON the path: the fused `_nq` epilogue does the norm
        // and the quant inside the reduce, the plain arms leave them as their own dispatch.
        match desc.op.as_str() {
            "allreduce_exact_nq" => self.counters.hit_ar_fused_nq(),
            "gemm_decode_ar_nq" => self.counters.hit_ar_fused_gemm(),
            "allreduce_exact" | "allreduce_wht" => self.counters.hit_ar_plain(),
            _ => {}
        }
        Ok(report)
    }

    /// The descriptor's byte layout against the manifest the code object was validated to.
    ///
    /// The loader has already proved that the manifest matches the `.hsaco`. This proves that the
    /// *descriptor* matches the manifest — which closes the loop, because the descriptor is what
    /// the launch is actually built from.
    fn check_against_manifest(&self, desc: &LaunchDescriptor, decl: &KernelDecl) -> Result<()> {
        if desc.kernarg_bytes > decl.kernarg_segment_size {
            self.counters.hit_refused_kernarg_drift();
            return Err(Error::launch(
                &desc.kernel,
                format!(
                    "descriptor packs {} B of kernargs, the code object's segment is {} B",
                    desc.kernarg_bytes, decl.kernarg_segment_size
                ),
            ));
        }
        if let Err(detail) = cover_declared_args(desc, decl) {
            self.counters.hit_refused_kernarg_drift();
            return Err(Error::launch(&desc.kernel, detail));
        }
        let threads =
            u64::from(desc.block[0]) * u64::from(desc.block[1]) * u64::from(desc.block[2]);
        if threads > u64::from(decl.max_flat_workgroup_size) {
            self.counters.hit_refused_kernarg_drift();
            return Err(Error::launch(
                &desc.kernel,
                format!(
                    "block {:?} is {threads} threads, the code object's max flat workgroup is {}",
                    desc.block, decl.max_flat_workgroup_size
                ),
            ));
        }
        Ok(())
    }

    fn assign_workspaces(
        &mut self,
        ctx: &LaunchCtx<'_>,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
        stream: &Stream,
        report: &mut LaunchReport,
    ) -> Result<BTreeMap<String, DeviceAddr>> {
        let mut out = BTreeMap::new();
        let mut scratch_slot = 0usize;
        for w in &desc.workspaces {
            let addr = match w.kind {
                WorkspaceKind::Scratch => {
                    let a = self.scratch_slab(ctx, scratch_slot, w, report)?;
                    scratch_slot += 1;
                    a
                }
                WorkspaceKind::Persistent => {
                    self.persistent_region(ctx, desc, args, w, stream, report)?
                }
                WorkspaceKind::IpcPeerVisible => {
                    self.ipc_region(ctx, desc, args, w, stream, report)?
                }
            };
            out.insert(w.name.clone(), addr);
        }
        Ok(out)
    }

    /// A peer-visible region: allocate mine, exchange the handle, or open the peer's.
    ///
    /// # How a descriptor says which is which
    ///
    /// There is no flag for it, and there should not be: what the host must do is a property of
    /// the *pair* of names, not of one region. `ar_workspaces` declares five regions —
    /// `ar_scratch`, `ar_peer_scratch`, `ar_flags`, `ar_peer_flags`, `ar_seq` — and the rule that
    /// reads them is:
    ///
    /// * a name containing `_peer_` is the **peer's** view of the region named with `peer_`
    ///   removed. It is opened from the peer's handle, never allocated, and closed (not freed) on
    ///   drop;
    /// * any other name is **mine**. It is allocated here, and exported to the peer only if this
    ///   descriptor also names its `_peer_` counterpart — which is exactly why `ar_seq` is
    ///   allocated but never exported. It carries the peer-visible KIND because it is protocol
    ///   state a replay must not re-zero and a second allocation would desynchronise, not because
    ///   a handle is taken on it. The reference allocates it with a plain `hipMalloc` and so does this.
    ///
    /// # Fine-grained, and why the device attribute is not the question
    ///
    /// An exported region is allocated with [`MallocFlags::FineGrained`], matching
    /// `saber_ar_ipc_alloc(size, finegrained=1)`. `hipDeviceAttributeFineGrainSupport` reads 0 on
    /// these cards and the allocation succeeds anyway; the attribute is not the gate. A
    /// coarse-grained region would let a rank's push sit in a cache its peer cannot snoop, and the
    /// symptom is a kernel that never retires.
    fn ipc_region(
        &mut self,
        ctx: &LaunchCtx<'_>,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
        w: &WorkspaceReq,
        stream: &Stream,
        report: &mut LaunchReport,
    ) -> Result<DeviceAddr> {
        let Some(peer) = self.peer.clone() else {
            self.counters.hit_refused_workspace_kind();
            return Err(Error::UnsupportedWorkspaceKind {
                kind: "ipc_peer_visible".to_owned(),
                name: w.name.clone(),
                kernel: desc.kernel.clone(),
                detail: "this Launcher has no peer; build it with `Launcher::with_peer` before \
                         launching a collective. A single-rank engine cannot serve a route that \
                         reads another rank's memory, and pretending otherwise is how a one-shot \
                         all-reduce spins forever"
                    .to_owned(),
            });
        };
        let key = self.ipc_key(desc, args, &w.name);
        if let Some(region) = self.ipc.get(&key) {
            self.counters.hit_ipc_reused();
            return Ok(region.addr());
        }

        if self.host_ar {
            if !matches!(
                desc.plugin.as_str(),
                "saber_ar_host_exact"
                    | "saber_ar_host_exact_chk"
                    | "mxfp4_gemm_decode_host_arnq"
                    | "mxfp4_gemm_decode_host_arnq_chk"
            ) {
                return Err(Error::launch(
                    &desc.kernel,
                    "host mapping requires a bounded exact code object",
                ));
            }
            if let Some(base) = peer_base_name(&w.name) {
                let base_key = self.ipc_key(desc, args, &base);
                let path = match self.ipc.get(&base_key) {
                    Some(IpcRegion::Host {
                        ptr, mine: true, ..
                    }) => ptr
                        .path()
                        .ok_or_else(|| {
                            Error::launch(
                                &desc.kernel,
                                "shared file already unlinked before exchange",
                            )
                        })?
                        .to_path_buf(),
                    _ => {
                        return Err(Error::launch(
                            &desc.kernel,
                            "shared peer workspace precedes its base",
                        ))
                    }
                };
                let bytes = path
                    .to_str()
                    .ok_or_else(|| Error::launch(&desc.kernel, "non-UTF8 shared file name"))?
                    .as_bytes();
                let theirs = peer
                    .exchange()
                    .xchg_bytes(&format!("host-ar.{base}"), bytes)?;
                let theirs = std::str::from_utf8(&theirs)
                    .map_err(|_| Error::launch(&desc.kernel, "non-UTF8 peer file name"))?;
                let ptr = ctx
                    .hip
                    .shared_host_open(std::path::Path::new(theirs), w.bytes.max(1))?;
                peer.rendezvous(&format!("host-ar.mapped.{base}"))?;
                if let Some(IpcRegion::Host { ptr, .. }) = self.ipc.get_mut(&base_key) {
                    ptr.unlink()?;
                }
                let addr = ptr.addr();
                self.ipc.insert(
                    key,
                    IpcRegion::Host {
                        ptr,
                        mine: false,
                        epoch: 0,
                    },
                );
                self.counters.hit_ipc_opened();
                return Ok(addr);
            }
            let exported = desc
                .workspaces
                .iter()
                .any(|o| peer_base_name(&o.name).as_deref() == Some(w.name.as_str()));
            if exported {
                let rounded = w.bytes.max(1).div_ceil(4096) * 4096;
                // Both mappings count against a per-process 1 GiB pinned-page reservation.
                if self.host_ipc_bytes().saturating_add(2 * rounded) > 1024 * 1024 * 1024 {
                    return Err(Error::launch(
                        &desc.kernel,
                        "host collective pages exceed the reserved 1 GiB cap",
                    ));
                }
                let ptr = ctx.hip.shared_host_create(w.bytes.max(1))?;
                let addr = ptr.addr();
                self.ipc.insert(
                    key,
                    IpcRegion::Host {
                        ptr,
                        mine: true,
                        epoch: 1,
                    },
                );
                self.counters.hit_ipc_exported();
                return Ok(addr);
            }
            // Sequence counters are local device atomics; only payload and flags use host pages.
        }

        if let Some(base) = peer_base_name(&w.name) {
            // The peer's region. Its handle comes from the exchange, and the base must be a
            // region this descriptor also declares — otherwise the peer has nothing to export.
            if !desc.workspaces.iter().any(|o| o.name == base) {
                self.counters.hit_refused_workspace_kind();
                return Err(Error::launch(
                    &desc.kernel,
                    format!(
                        "workspace `{}` is the peer's view of `{base}`, which this descriptor does \
                         not declare; there is nothing for the peer to export",
                        w.name
                    ),
                ));
            }
            // Peer access before the open: `hipIpcOpenMemHandle` needs it, and "already enabled"
            // is success, not a failure to paper over.
            match ctx.hip.enable_peer_access(peer.peer_rank() as i32) {
                Ok(()) => self.counters.hit_peer_access_enabled(),
                Err(e) if e.hip_status() == Some(HipError::PeerAccessAlreadyEnabled) => {
                    self.counters.hit_peer_access_already();
                }
                Err(e) => return Err(e.into()),
            }
            // The base region must already exist here: this rank exports it in the same pass, and
            // `assign_workspaces` walks the descriptor in declaration order with the base first.
            let base_key = self.ipc_key(desc, args, &base);
            let handle = match self.ipc.get(&base_key) {
                Some(IpcRegion::Mine { ptr, .. }) => ctx.hip.ipc_export(ptr)?,
                _ => {
                    return Err(Error::launch(
                        &desc.kernel,
                        format!(
                            "workspace `{}` was resolved before its base `{base}`; the descriptor \
                             must declare a region before the peer's view of it",
                            w.name
                        ),
                    ))
                }
            };
            let theirs = peer.swap_handle(&base, handle)?;
            let ptr = ctx.hip.ipc_open(theirs)?;
            let addr = ptr.addr();
            self.counters.hit_ipc_opened();
            self.ipc.insert(key, IpcRegion::Peer { ptr });
            return Ok(addr);
        }

        // Mine. Exported only if the peer's view of it is also declared here.
        let exported = desc
            .workspaces
            .iter()
            .any(|o| peer_base_name(&o.name).as_deref() == Some(w.name.as_str()));
        let ptr = if exported {
            ctx.hip
                .malloc_with_flags(w.bytes.max(1), MallocFlags::FineGrained)?
        } else {
            ctx.hip.malloc(w.bytes.max(1))?
        };
        // Zeroed exactly once, here, and synchronised: the very first launch must not race the
        // memset, and every later launch inherits the protocol state the last one left.
        ctx.hip.memset(&ptr, 0, stream)?;
        ctx.hip.sync_stream(stream)?;
        report.persistent_allocated += ptr.len();
        if exported {
            self.counters.hit_ipc_exported();
        } else {
            self.counters.hit_ipc_local();
        }
        let addr = ptr.addr();
        self.ipc.insert(key, IpcRegion::Mine { ptr, epoch: 1 });
        Ok(addr)
    }

    fn ipc_key(&self, desc: &LaunchDescriptor, args: &ArgBindings, name: &str) -> PersistentKey {
        let shape = args.shape();
        PersistentKey {
            plugin: protocol_family(&desc.plugin).to_owned(),
            op: protocol_op(&desc.plugin, &desc.op).to_owned(),
            m: shape.m,
            n: shape.n,
            k: shape.k,
            name: name.to_owned(),
        }
    }

    /// The digest both ranks compare before a collective launch.
    ///
    /// Everything the protocol depends on and nothing that it does not: the kernel symbol, the
    /// launch geometry, the kernarg size, and each peer-visible region's declared bytes and reset
    /// epoch. Buffer ADDRESSES are deliberately absent — they differ between ranks by
    /// construction.
    #[must_use]
    pub fn collective_digest(&self, desc: &LaunchDescriptor, args: &ArgBindings) -> u64 {
        let mut d = ProtocolDigest::new();
        d.text(&desc.plugin)
            .text(&desc.kernel)
            .text(&desc.op)
            .num(u64::from(desc.grid[0]))
            .num(u64::from(desc.grid[1]))
            .num(u64::from(desc.grid[2]))
            .num(u64::from(desc.block[0]))
            .num(u64::from(desc.shared_bytes))
            .num(u64::from(desc.kernarg_bytes))
            .num(u64::from(desc.n_launches));
        for slot in &desc.args {
            d.text(&slot.name).num(u64::from(slot.offset));
            // Scalars are part of the contract (n_groups, the slot stride, drain/acq); pointers
            // are not, because they are per rank.
            match &slot.value {
                ArgValue::I32 { ival } | ArgValue::I64 { ival } => {
                    d.num(*ival as u64);
                }
                ArgValue::F32 { fval } => {
                    d.num(u64::from(fval.to_bits()));
                }
                ArgValue::Buffer { .. } | ArgValue::NullBuffer { .. } => {}
            }
        }
        for w in &desc.workspaces {
            d.text(&w.name).num(w.bytes);
            if w.kind == WorkspaceKind::IpcPeerVisible {
                let key = self.ipc_key(desc, args, &w.name);
                let epoch = match self.ipc.get(&key) {
                    Some(IpcRegion::Mine { epoch, .. }) => *epoch,
                    // The peer's view has no epoch of its own; the peer accounts for it.
                    Some(IpcRegion::Host {
                        mine: true, epoch, ..
                    }) => *epoch,
                    Some(IpcRegion::Peer { .. }) | Some(IpcRegion::Host { .. }) | None => 0,
                };
                d.num(epoch);
            }
        }
        d.finish()
    }

    /// Re-zero every peer-visible region this rank owns, and count the reset.
    ///
    /// The bit-exact gate needs this between cases, because the reference run leaves the flags and
    /// the sequence counters advanced. It is NOT part of the serve path: there the protocol state
    /// is zeroed once and carried forward, which is the whole reason the kind exists. Both ranks
    /// must call it the same number of times, and [`Launcher::collective_digest`] is what makes a
    /// rank that did not say so.
    pub fn reset_ipc_state(&mut self, ctx: &LaunchCtx<'_>, stream: &Stream) -> Result<usize> {
        let mut n = 0;
        ctx.hip.sync_stream(stream)?;
        if self.host_ar {
            self.peer
                .as_ref()
                .expect("host transport peer")
                .rendezvous("host-ar.reset-idle")?;
        }
        for region in self.ipc.values_mut() {
            if let IpcRegion::Mine { ptr, epoch } = region {
                ctx.hip.memset(ptr, 0, stream)?;
                *epoch += 1;
                n += 1;
            } else if let IpcRegion::Host {
                ptr,
                mine: true,
                epoch,
            } = region
            {
                // Both ranks synchronized their streams and met the reset barrier above.
                ctx.hip.memset_shared_zero(ptr, stream)?;
                *epoch += 1;
                n += 1;
            }
        }
        ctx.hip.sync_stream(stream)?;
        if self.host_ar {
            self.peer
                .as_ref()
                .expect("host transport peer")
                .rendezvous("host-ar.reset-done")?;
        }
        self.counters.hit_ipc_reset();
        Ok(n)
    }

    /// Allocate and exchange every workspace this descriptor needs, without launching.
    ///
    /// The collective path needs this as a separate step: the IPC handles have to be exchanged
    /// **before** the two ranks compare their protocol digest, or the digest would be comparing
    /// sizes that do not exist yet. It is also what a plan-bind would call, since an
    /// `IpcPeerVisible` region must exist before capture (invariant I1) and a handle exchange
    /// inside a capture is not a thing.
    pub fn prepare_workspaces(
        &mut self,
        ctx: &LaunchCtx<'_>,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
        stream: &Stream,
    ) -> Result<()> {
        if desc.plugin == saber_kernels::saber::RCCL_PLUGIN {
            self.rccl_args(desc, args)?;
            return Ok(());
        }
        let mut report = LaunchReport {
            launches: 0,
            scratch_allocated: 0,
            persistent_allocated: 0,
        };
        self.assign_workspaces(ctx, desc, args, stream, &mut report)?;
        Ok(())
    }

    /// Size every **local** workspace from the MAX over a whole plan, before anything launches.
    ///
    /// `assign_workspaces` restarts its `scratch_slot` counter at 0 for every op, so slab 0 is
    /// shared by every op's first `Scratch` region and `scratch_slab` grows it — a `hipMalloc` plus
    /// the dropped slab's `hipFree` — whenever an op wants more than the last one left. At decode
    /// that is invisible: the split-K partials are kilobytes and one slab fits every op. At a
    /// 2048-row plan a partial is `DKS · M · N` **floats**, 142 MB for one `gate_up`, and the grow
    /// chain on the first eager step is tens of milliseconds — which is what made I15's eager
    /// prefill read 125 ms slower than the graph's and, divided by nodes, look like 165 µs of
    /// per-node overhead that submission cannot explain.
    ///
    /// A graph never paid it: its caller prepares every op's workspaces BEFORE the capture, so the
    /// grows happen outside the timed region and the replay reuses baked addresses. This gives the
    /// eager path the same property: one allocation per slab at the plan's maximum, and a
    /// `scratch(grown=…)` of zero on every step afterwards.
    ///
    /// `IpcPeerVisible` regions are deliberately NOT touched — they need both ranks in lockstep and
    /// their callers already exchange them explicitly. Persistent regions ARE prepared here,
    /// because they are one `hipMalloc` **and a `memset`** per distinct `(plugin, op, shape)` and a
    /// large-M plan has many.
    ///
    /// Returns the bytes newly allocated, for the report.
    ///
    /// # Errors
    /// An allocation failing, or a Persistent region whose key cannot be formed.
    pub fn reserve_local(
        &mut self,
        ctx: &LaunchCtx<'_>,
        descs: &[(&LaunchDescriptor, &ArgBindings)],
        stream: &Stream,
    ) -> Result<u64> {
        // Per-slot maxima first, so the slab is allocated once at the size the widest op needs
        // rather than climbing to it.
        let mut want: Vec<u64> = Vec::new();
        for (desc, _) in descs {
            let mut slot = 0usize;
            for w in &desc.workspaces {
                if w.kind != WorkspaceKind::Scratch {
                    continue;
                }
                if want.len() <= slot {
                    want.resize(slot + 1, 0);
                }
                want[slot] = want[slot].max(w.bytes);
                slot += 1;
            }
        }
        let mut report = LaunchReport {
            launches: 0,
            scratch_allocated: 0,
            persistent_allocated: 0,
        };
        for (index, bytes) in want.iter().enumerate() {
            let w = WorkspaceReq {
                name: format!("reserve.slab{index}"),
                kind: WorkspaceKind::Scratch,
                bytes: *bytes,
                zero_once: false,
            };
            self.scratch_slab(ctx, index, &w, &mut report)?;
        }
        // The Persistent regions, in plan order, so their allocation and zeroing also happen once
        // and outside any timed step.
        for (desc, args) in descs {
            for w in &desc.workspaces {
                if w.kind != WorkspaceKind::Persistent {
                    continue;
                }
                self.persistent_region(ctx, desc, args, w, stream, &mut report)?;
            }
        }
        Ok(report.scratch_allocated + report.persistent_allocated)
    }

    /// Bytes held by the peer-visible regions this rank owns.
    #[must_use]
    pub fn ipc_bytes(&self) -> u64 {
        self.ipc.values().map(IpcRegion::bytes).sum()
    }

    /// Page-rounded registered host mappings held by this process (both rank views).
    #[must_use]
    pub fn host_ipc_bytes(&self) -> u64 {
        self.ipc
            .values()
            .map(|r| match r {
                IpcRegion::Host { ptr, .. } => ptr.len().div_ceil(4096) * 4096,
                _ => 0,
            })
            .sum()
    }

    /// The Persistent regions this launcher owns: `(key, bytes)`, in key order.
    ///
    /// Persistent state is the one thing a captured graph carries ACROSS replays — the split-K
    /// block counter the fused decode reduce elects on is the whole of it today — so a caller
    /// that replays a bucket has to be able to see what survives, and to say so in a report.
    #[must_use]
    pub fn persistent_regions(&self) -> Vec<(String, u64)> {
        self.persistent
            .iter()
            .map(|(k, p)| {
                (
                    format!(
                        "{}/{} m={} n={} k={} {}",
                        k.plugin, k.op, k.m, k.n, k.k, k.name
                    ),
                    p.len(),
                )
            })
            .collect()
    }

    /// **T10: the address of the peer-visible region `name`** (`ar_scratch`, `ar_loc_pack`,
    /// `ar_flags`, `ar_seq`, ...) a collective `desc` resolves to under `args` -- the SAME key its
    /// launch uses (plugin, op, the bound shape, the name), so a reader sees exactly the region the
    /// kernel handshakes through. `None` before the first launch of that key allocated it.
    #[must_use]
    pub fn ipc_region_addr(
        &self,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
        name: &str,
    ) -> Option<DeviceAddr> {
        self.ipc
            .get(&self.ipc_key(desc, args, name))
            .map(IpcRegion::addr)
    }

    /// **T10: the address of the `index`-th Scratch slab** (a descriptor's `index`-th Scratch
    /// workspace binds it; the slabs are shared by every op, so its bytes are the LAST op's that
    /// used it). `None` before a launch allocated it.
    #[must_use]
    pub fn scratch_slab_addr(&self, index: usize) -> Option<DeviceAddr> {
        self.scratch.get(index).map(DevicePtr::addr)
    }

    /// Every `ar_seq` region this rank owns (T4): `(key, address, bytes)`, in key order.
    ///
    /// The all-reduce's per-block sequence counters are the protocol's lockstep — both ranks issue
    /// the same calls, so after the same step both hold the same counters, region for region — and
    /// under the race checks their tail carries the `_chk` kernels' diagnostic words
    /// (`saber_kernels::saber::AR_CHECK_DIAG_OFFSET`). The regions live as long as this launcher and
    /// are never freed while it runs, which is what lets a reader hold the ADDRESS
    /// (`HipThread::memcpy_d2h_addr_blocking`) where it cannot borrow the launcher. The key is the
    /// same string on both ranks: plugin, op and shape, never an address.
    #[must_use]
    pub fn ar_seq_regions(&self) -> Vec<(String, DeviceAddr, u64)> {
        self.ipc
            .iter()
            .filter_map(|(k, r)| match r {
                IpcRegion::Mine { ptr, .. } if k.name == "ar_seq" => Some((
                    format!("{}/{} m={} n={} k={}", k.plugin, k.op, k.m, k.n, k.k),
                    ptr.addr(),
                    ptr.len(),
                )),
                _ => None,
            })
            .collect()
    }

    /// Re-zero every Persistent region, and return how many were zeroed.
    ///
    /// `zero_once` zeroes a region when it is allocated and nothing else ever does, which is
    /// right for one bucket and not enough for a launcher reused across several: a region whose
    /// protocol left it dirty has to start the next bucket at zero.
    ///
    /// **Zero is the only byte this writes, deliberately.** A Persistent region here is a
    /// *protocol* variable, not a scratch buffer. The fused split-K reduce elects the last block
    /// to arrive with `atomicAdd(cnt) == DKS - 1`, which merely fails to elect anyone if the
    /// counter is dirty — but the GDN fused update's `gdn_fused_barrier` is a cross-workgroup
    /// **barrier**, and a barrier counter that does not start at zero is waited on forever. A
    /// non-zero fill was tried as a negative control on 2026-09-19 and wedged the card for fifty
    /// minutes. The property that control was after is now checked where it belongs, before any
    /// node runs: see [`crate::block_capture::check_persistent_ready`].
    pub fn zero_persistent(&self, ctx: &LaunchCtx<'_>, stream: &Stream) -> Result<usize> {
        for p in self.persistent.values() {
            ctx.hip.memset(p, 0, stream)?;
        }
        ctx.hip.sync_stream(stream)?;
        Ok(self.persistent.len())
    }

    /// Does this launcher already own the Persistent region `(desc, args.shape(), name)` names?
    ///
    /// The question a capture has to ask before it opens: a Persistent workspace first seen
    /// *inside* a capture would allocate and `hipMemset` while the stream is capturing, which
    /// makes the zeroing a NODE — re-run on every replay, over a protocol variable whose whole
    /// contract is that it survives one. The key is built exactly as [`Self::persistent_region`]
    /// builds it, because a second opinion about the key is a second bug.
    #[must_use]
    pub fn has_persistent(&self, desc: &LaunchDescriptor, args: &ArgBindings, name: &str) -> bool {
        let shape = args.shape();
        self.persistent.contains_key(&PersistentKey {
            plugin: desc.plugin.clone(),
            op: desc.op.clone(),
            m: shape.m,
            n: shape.n,
            k: shape.k,
            name: name.to_owned(),
        })
    }

    fn scratch_slab(
        &mut self,
        ctx: &LaunchCtx<'_>,
        index: usize,
        w: &WorkspaceReq,
        report: &mut LaunchReport,
    ) -> Result<DeviceAddr> {
        if w.zero_once {
            return Err(Error::launch(
                &w.name,
                "a Scratch workspace cannot ask to be zeroed once: Scratch is aliasable, so its \
                 contents are whatever the previous op on this stream left behind",
            ));
        }
        while self.scratch.len() <= index {
            let ptr = ctx.hip.malloc(w.bytes.max(1))?;
            report.scratch_allocated += ptr.len();
            self.counters.hit_scratch_new();
            self.scratch.push(ptr);
        }
        if self.scratch[index].len() < w.bytes {
            // Grow in place: dropping the old slab posts `hipFree` to the HIP thread *before* the
            // new `hipMalloc` job, because both cross the same FIFO channel.
            let ptr = ctx.hip.malloc(w.bytes)?;
            report.scratch_allocated += ptr.len();
            self.counters.hit_scratch_grown();
            self.scratch[index] = ptr;
        } else {
            self.counters.hit_scratch_reused();
        }
        Ok(self.scratch[index].addr())
    }

    fn persistent_region(
        &mut self,
        ctx: &LaunchCtx<'_>,
        desc: &LaunchDescriptor,
        args: &ArgBindings,
        w: &WorkspaceReq,
        stream: &Stream,
        report: &mut LaunchReport,
    ) -> Result<DeviceAddr> {
        let shape = args.shape();
        let key = PersistentKey {
            plugin: desc.plugin.clone(),
            op: desc.op.clone(),
            m: shape.m,
            n: shape.n,
            k: shape.k,
            name: w.name.clone(),
        };
        if let Some(p) = self.persistent.get(&key) {
            if p.len() >= w.bytes {
                self.counters.hit_persistent_reused();
                return Ok(p.addr());
            }
            // A Persistent region that has to change size at the same (plugin, op, shape) means the
            // size law disagrees with itself. Refuse: silently reallocating it would break any
            // captured graph holding the old address.
            return Err(Error::launch(
                &desc.kernel,
                format!(
                    "persistent workspace `{}` at ({}, {}, m={} n={} k={}) was allocated with {} B \
                     and is now asked for {} B",
                    w.name,
                    desc.plugin,
                    desc.op,
                    shape.m,
                    shape.n,
                    shape.k,
                    p.len(),
                    w.bytes
                ),
            ));
        }
        let ptr = ctx.hip.malloc(w.bytes.max(1))?;
        if w.zero_once {
            // Zeroed exactly once, here, and synchronised so that the very first launch on this
            // region cannot race the memset.
            ctx.hip.memset(&ptr, 0, stream)?;
            ctx.hip.sync_stream(stream)?;
            self.zero_once.insert(key.clone());
        }
        report.persistent_allocated += ptr.len();
        self.counters.hit_persistent_new();
        let addr = ptr.addr();
        self.persistent.insert(key, ptr);
        Ok(addr)
    }
}

impl std::fmt::Debug for Launcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (s, p) = self.arena_bytes();
        write!(
            f,
            "Launcher({}, scratch {} slab(s)/{s} B, persistent {}/{p} B)",
            self.stream_label,
            self.scratch.len(),
            self.persistent.len()
        )
    }
}

#[cfg(test)]
mod protocol_family_tests {
    use super::protocol_family;

    /// CMP27S: the wave-parallel wht6 body and its twin key onto the served objects' regions (one
    /// sequence, one scratch), like P27-1d's split form; the exact family keys as itself.
    #[test]
    fn the_wht_bodies_share_one_protocol_region_set() {
        assert_eq!(protocol_family("saber_ar_wht"), "saber_ar_wht");
        assert_eq!(protocol_family("saber_ar_wht_split"), "saber_ar_wht");
        assert_eq!(protocol_family("saber_ar_wht_wpar"), "saber_ar_wht");
        assert_eq!(protocol_family("saber_ar_wht_wpar_chk"), "saber_ar_wht_chk");
        assert_eq!(protocol_family("saber_ar_wht_chk"), "saber_ar_wht_chk");
        assert_eq!(protocol_family("saber_ar_exact"), "saber_ar_exact");
    }

    /// CMP27AE: the GEMM + wht6 fold keys onto the rotated one-shot's regions (family AND op), so
    /// it allocates no second scratch; every other descriptor keys as its own op.
    #[test]
    fn the_gemm_wht_fold_keys_onto_the_rotated_reduces_regions() {
        use super::protocol_op;
        assert_eq!(protocol_family("mxfp4_gemm_decode_arw"), "saber_ar_wht");
        assert_eq!(
            protocol_op("mxfp4_gemm_decode_arw", "gemm_decode_arw"),
            "allreduce_wht"
        );
        assert_eq!(
            protocol_op("saber_ar_wht", "allreduce_wht"),
            "allreduce_wht"
        );
        assert_eq!(
            protocol_op("mxfp4_gemm_decode_arnq", "gemm_decode_ar_nq"),
            "gemm_decode_ar_nq"
        );
    }
}
