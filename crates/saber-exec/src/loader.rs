//! Loading a plugin: three validations, then `hipModuleLoadData` and one
//! `hipModuleGetFunction` per symbol.
//!
//! > **Kernarg ABI descriptor**, versioned and hash-checked against the code object's own
//! > metadata — this is the code-object equivalent of the partition-mismatched tuple that silently
//! > hangs the GPU: a kernarg layout the host and the kernel disagree about is memory corruption,
//! > not an error.
//! >
//! > — ARCHITECTURE.md, *Kernel plane*
//!
//! So nothing is launched until three *independent* derivations agree, and each one has its own
//! which-path counter so that "which check refused this plugin" is a number rather than a log
//! grep:
//!
//! | # | Check | Compares | Counter on refusal |
//! |---|---|---|---|
//! | (b) | [`check_kernarg_layout`] | the manifest's own argument table against `saber_abi::check_kernarg` and the structural rules (ascending, non-overlapping, inside the segment) | `refused_kernarg` |
//! | (c) | [`check_wave_size`] | the wave size the manifest records against the code object's, and against itself across the plugin's kernels | `refused_wave` |
//! | (a) | [`saber_kernels::Manifest::validate_against_hsaco`] | every declared number — blob sha256, target, kernarg size/align, LDS, VGPR/SGPR, symbol, argument offsets — against the `.hsaco`'s msgpack metadata | `refused_metadata` |
//!
//! The order is deliberate and is the opposite of the order the table above reads in the
//! architecture: (b) needs nothing but the manifest, (c) needs the code object's metadata, and (a)
//! subsumes parts of both. Running the narrow checks first is what makes each counter reachable —
//! if `validate_against_hsaco` ran first it would absorb every drifted offset and a *kernarg*
//! refusal would never be observed, which is how a diagnostic that arms and never runs happens.
//!
//! **Wave size, specifically.** `-mwavefrontsize64` is translation-unit wide and the per-function
//! attribute is silently ignored (`rdna4-wave64-wmma`), so a plugin that mixes wave sizes across
//! its kernels is two code objects wearing one name. That is checked here rather than trusted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use saber_abi::saber_kernarg;
use saber_hip::{Function, HipThread, Module};
use saber_kernels::codeobj::CodeObject;
use saber_kernels::manifest::KernelDecl;
use saber_kernels::Manifest;

use crate::counters::LoadCounters;
use crate::error::{Error, Result};

/// What validation established about a plugin, beyond "it did not refuse".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginFacts {
    /// Kernel-set stem, e.g. `mxfp4_gemm_decode`.
    pub plugin: String,
    /// Offload target the code object was built for.
    pub target: String,
    /// Kernel symbols in the plugin.
    pub kernels: usize,
    /// The translation unit's wave size — one value per plugin, checked to be uniform.
    pub wave_size: u32,
    /// The largest static LDS any kernel in the plugin declares, for the occupancy budget.
    pub max_lds_bytes: u32,
    /// The largest explicit kernarg segment, for the launcher's staging buffer.
    pub max_kernarg_bytes: u32,
}

/// (b) The manifest's kernarg table against the ABI's own rules.
///
/// `saber_abi::check_kernarg` owns the per-argument part (size and alignment match the kind, the
/// offset is naturally aligned). The structural part — ascending, non-overlapping, inside the
/// declared segment — is here because it is a property of the *table*, not of one row.
///
/// The code object reports `by_value` for every scalar, so a 4-byte `by_value` is mapped to
/// `ArgKind::I32` and an 8-byte one to `ArgKind::I64`. That is not a guess about the kernel's C
/// type: `check_kernarg` only asks about size and alignment, and `f32`/`i32` (and `f64`/`i64`)
/// are indistinguishable to it by construction. Naming the operands is
/// `saber_kernels::abi::signature_for`'s job, and it is done when the manifest is *built*.
///
/// # By-value structs
///
/// A `by_value` argument of any other size is a **struct passed by value**
/// ([`saber_abi::ArgKind::Struct`], ABI 0.3). The reference's two paged-attention kernels take
/// `const SABERArgs a`, which the code object reports as one 136-byte `by_value` argument.
///
/// Its alignment is not in the blob — the per-argument metadata carries `.offset`, `.size` and
/// `.value_kind` and nothing else — so it is **derived**, not invented, from two facts the AMDGPU
/// C ABI guarantees: `sizeof(T)` is a multiple of `alignof(T)`, and no argument is more aligned
/// than the segment. [`implied_struct_align`] therefore takes the largest power of two dividing
/// the declared size, capped at `kernarg_segment_align`. That is the *weakest* alignment
/// consistent with the blob's own numbers being a legal C layout, so it refuses exactly the
/// layouts that cannot be legal and accepts every one that can — the opposite choice would reject
/// a legitimately 4-aligned struct sitting at offset 4.
///
/// The exact alignment is pinned elsewhere and earlier: `saber_kernels::abi::signature_for` names
/// `ArgKind::Struct { bytes: 136, align: 8 }`, and `Manifest::from_code_object` refuses at *build*
/// time if the blob does not pack that way.
pub fn check_kernarg_layout(m: &Manifest) -> Result<()> {
    for k in &m.kernel {
        let mut end = 0u32;
        for (i, a) in k.arg.iter().enumerate() {
            let kind = match (a.value_kind.as_str(), a.size) {
                ("global_buffer", 8) => saber_abi::ArgKind::Ptr,
                ("by_value", 4) => saber_abi::ArgKind::I32,
                ("by_value", 8) => saber_abi::ArgKind::I64,
                ("by_value", _) => saber_abi::ArgKind::Struct,
                (vk, sz) => {
                    return Err(Error::validate(
                        &m.plugin,
                        &k.name,
                        &format!("arg[{i}].value_kind"),
                        format!(
                            "`{vk}` at size {sz} is not an argument kind this ABI packs \
                             (global_buffer/8, or by_value of any size for a scalar or a \
                             by-value struct)"
                        ),
                    ))
                }
            };
            let align = match kind.align() {
                Some(natural) => natural,
                None => implied_struct_align(a.size, k.kernarg_segment_align),
            };
            let raw = saber_kernarg {
                name: std::ptr::null(),
                kind: kind.to_raw(),
                offset: a.offset,
                size: a.size,
                align,
            };
            if let Err(e) = saber_abi::check_kernarg(&raw) {
                return Err(Error::validate(
                    &m.plugin,
                    &k.name,
                    &format!("arg[{i}]"),
                    format!("offset {} size {} kind {:?}: {e}", a.offset, a.size, kind),
                ));
            }
            if a.offset < end {
                return Err(Error::validate(
                    &m.plugin,
                    &k.name,
                    &format!("arg[{i}].offset"),
                    format!(
                        "offset {} overlaps the previous argument, which ends at {end}",
                        a.offset
                    ),
                ));
            }
            end = a.offset + a.size;
        }
        if end > k.kernarg_segment_size {
            return Err(Error::validate(
                &m.plugin,
                &k.name,
                "kernarg_segment_size",
                format!(
                    "the explicit arguments need {end} B but the segment is {} B",
                    k.kernarg_segment_size
                ),
            ));
        }
        if k.kernarg_segment_align == 0 || !k.kernarg_segment_align.is_power_of_two() {
            return Err(Error::validate(
                &m.plugin,
                &k.name,
                "kernarg_segment_align",
                format!("{} is not a power of two", k.kernarg_segment_align),
            ));
        }
    }
    Ok(())
}

/// The alignment a by-value struct of this size can have, derived from the blob alone.
///
/// The AMDGPU C ABI makes `sizeof(T)` a multiple of `alignof(T)` and places the argument at a
/// multiple of `alignof(T)`, and no argument is more aligned than `.kernarg_segment_align`. So the
/// true alignment divides the size and is at most the segment's, and the largest such power of two
/// is the strongest claim the metadata supports. `SABERArgs` at 136 bytes in a segment aligned to 8
/// gives 8, which is its real alignment.
///
/// Never zero: `size` is checked non-zero by [`saber_abi::check_kernarg`] immediately afterwards,
/// and a zero `segment_align` is refused by the segment check below.
#[must_use]
pub fn implied_struct_align(size: u32, segment_align: u32) -> u32 {
    let from_size = if size == 0 {
        1
    } else {
        1 << size.trailing_zeros()
    };
    let cap = if segment_align == 0 { 1 } else { segment_align };
    from_size.min(cap).max(1)
}

/// (c) Wave size: the manifest against the code object, and the plugin against itself.
///
/// Returns the plugin's single wave size.
pub fn check_wave_size(m: &Manifest, co: &CodeObject) -> Result<u32> {
    let mut plugin_wave: Option<(u32, &str)> = None;
    for d in &m.kernel {
        if d.wavefront_size != 32 && d.wavefront_size != 64 {
            return Err(Error::validate(
                &m.plugin,
                &d.name,
                "wavefront_size",
                format!("{} is neither 32 nor 64", d.wavefront_size),
            ));
        }
        let Some(k) = co.kernel(&d.name) else {
            // Left to (a), which reports missing kernels with the whole list at once.
            continue;
        };
        if d.wavefront_size != k.wavefront_size {
            return Err(Error::validate(
                &m.plugin,
                &d.name,
                "wavefront_size",
                format!(
                    "manifest says {}, the code object says {}",
                    d.wavefront_size, k.wavefront_size
                ),
            ));
        }
        match plugin_wave {
            None => plugin_wave = Some((d.wavefront_size, &d.name)),
            Some((w, first)) if w != d.wavefront_size => {
                return Err(Error::validate(
                    &m.plugin,
                    &d.name,
                    "wavefront_size",
                    format!(
                        "wave{} here but wave{w} in `{first}`; -mwavefrontsize64 is \
                         translation-unit wide, so one plugin cannot hold both",
                        d.wavefront_size
                    ),
                ));
            }
            Some(_) => {}
        }
    }
    plugin_wave.map(|(w, _)| w).ok_or_else(|| {
        Error::validate(
            &m.plugin,
            "*",
            "kernel",
            "the manifest declares no kernels at all",
        )
    })
}

/// Run all three validations, in the order that keeps each counter reachable.
///
/// Pure: no HIP, no device, no allocation beyond parsing. That is what lets the negative control
/// run in `cargo test` on a host with no ROCm.
pub fn validate_plugin(m: &Manifest, hsaco: &[u8], counters: &LoadCounters) -> Result<PluginFacts> {
    if let Err(e) = check_kernarg_layout(m) {
        counters.hit_refused_kernarg();
        return Err(e);
    }
    let co = match CodeObject::parse(hsaco) {
        Ok(co) => co,
        Err(e) => {
            counters.hit_refused_metadata();
            return Err(e.into());
        }
    };
    let wave = match check_wave_size(m, &co) {
        Ok(w) => w,
        Err(e) => {
            counters.hit_refused_wave();
            return Err(e);
        }
    };
    if let Err(e) = m.validate_against_hsaco(hsaco) {
        counters.hit_refused_metadata();
        return Err(e.into());
    }
    Ok(PluginFacts {
        plugin: m.plugin.clone(),
        target: co.target.clone(),
        kernels: m.kernel.len(),
        wave_size: wave,
        max_lds_bytes: m
            .kernel
            .iter()
            .map(|k| k.group_segment_fixed_size)
            .max()
            .unwrap_or(0),
        max_kernarg_bytes: m
            .kernel
            .iter()
            .map(KernelDecl::explicit_kernarg_bytes)
            .max()
            .unwrap_or(0),
    })
}

/// One loaded, validated plugin: its manifest, what validation established, and the HIP module.
pub struct LoadedPlugin {
    /// The manifest as committed next to the blob.
    pub manifest: Manifest,
    /// What the three validations established.
    pub facts: PluginFacts,
    module: Module,
}

impl std::fmt::Debug for LoadedPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LoadedPlugin({})", self.facts.plugin)
    }
}

/// Every plugin the engine has loaded, and the one place a `.hsaco` reaches HIP.
pub struct PluginSet {
    plugins: Vec<LoadedPlugin>,
    counters: Arc<LoadCounters>,
}

impl PluginSet {
    /// Load every `*.manifest.toml` in `dir`, validating each before it touches HIP.
    ///
    /// Sorted by file name so that the load order — and therefore the counters and any log — is
    /// the same on every machine.
    pub fn load_dir(hip: &HipThread, dir: &Path, counters: &Arc<LoadCounters>) -> Result<Self> {
        let mut manifests: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| Error::io(dir, e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().ends_with(".manifest.toml"))
            .collect();
        manifests.sort();
        if manifests.is_empty() {
            return Err(Error::NotFound(format!(
                "{}: no *.manifest.toml (is this the prebuilt directory?)",
                dir.display()
            )));
        }
        Self::load_manifests(hip, dir, &manifests, counters)
    }

    /// Load the named plugins only (`mxfp4_quant`, ...), by manifest stem.
    pub fn load_named(
        hip: &HipThread,
        dir: &Path,
        names: &[&str],
        counters: &Arc<LoadCounters>,
    ) -> Result<Self> {
        let paths: Vec<PathBuf> = names
            .iter()
            .filter(|n| **n != saber_kernels::saber::RCCL_PLUGIN)
            .map(|n| dir.join(format!("{n}.manifest.toml")))
            .collect();
        Self::load_manifests(hip, dir, &paths, counters)
    }

    fn load_manifests(
        hip: &HipThread,
        dir: &Path,
        paths: &[PathBuf],
        counters: &Arc<LoadCounters>,
    ) -> Result<Self> {
        let mut plugins = Vec::with_capacity(paths.len());
        for path in paths {
            let manifest = Manifest::load(path)?;
            let blob = manifest.hsaco_path(dir);
            let bytes = std::fs::read(&blob).map_err(|e| Error::io(&blob, e))?;
            let facts = validate_plugin(&manifest, &bytes, counters)?;
            // Only now does a byte of this blob reach the runtime.
            let module = hip.load_module(&bytes)?;
            // One `hipModuleGetFunction` per declared symbol, at load: a symbol that is in the
            // manifest but not in the loaded module is a refusal here rather than a `hipErrorNotFound`
            // in the middle of a captured bucket.
            for k in &manifest.kernel {
                hip.get_function(&module, &k.name).map_err(|e| {
                    Error::validate(
                        &manifest.plugin,
                        &k.name,
                        "symbol",
                        format!("hipModuleGetFunction refused: {e}"),
                    )
                })?;
            }
            counters.hit_loaded();
            tracing::debug!(
                plugin = %facts.plugin,
                kernels = facts.kernels,
                wave = facts.wave_size,
                lds = facts.max_lds_bytes,
                "plugin loaded"
            );
            plugins.push(LoadedPlugin {
                manifest,
                facts,
                module,
            });
        }
        Ok(PluginSet {
            plugins,
            counters: Arc::clone(counters),
        })
    }

    /// Every loaded plugin, in load order.
    #[must_use]
    pub fn plugins(&self) -> &[LoadedPlugin] {
        &self.plugins
    }

    /// The plugin of that name.
    #[must_use]
    pub fn plugin(&self, name: &str) -> Option<&LoadedPlugin> {
        self.plugins.iter().find(|p| p.facts.plugin == name)
    }

    /// The load counters this set was built with.
    #[must_use]
    pub fn counters(&self) -> &Arc<LoadCounters> {
        &self.counters
    }

    /// Resolve every declared symbol to a `hipFunction_t`, keyed by `(plugin, kernel)`.
    ///
    /// Borrowed from the module, so a function handle cannot outlive the module it came from —
    /// which is the same reason [`saber_hip::Function`] carries a lifetime at all.
    pub fn functions(&self, hip: &HipThread) -> Result<FunctionTable<'_>> {
        let mut fns = BTreeMap::new();
        for p in &self.plugins {
            for k in &p.manifest.kernel {
                let f = hip.get_function(&p.module, &k.name)?;
                fns.insert((p.facts.plugin.clone(), k.name.clone()), (f, k));
            }
        }
        Ok(FunctionTable { fns })
    }
}

/// Every resolved entry point, with the manifest row that describes its kernarg layout.
pub struct FunctionTable<'m> {
    fns: BTreeMap<(String, String), (Function<'m>, &'m KernelDecl)>,
}

impl<'m> FunctionTable<'m> {
    /// The entry point and its declaration, or `None`.
    #[must_use]
    pub fn get(&self, plugin: &str, kernel: &str) -> Option<(&Function<'m>, &'m KernelDecl)> {
        self.fns
            .get(&(plugin.to_owned(), kernel.to_owned()))
            .map(|(f, d)| (f, *d))
    }

    /// How many symbols are resolved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fns.len()
    }

    /// Whether nothing is resolved.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fns.is_empty()
    }
}

impl std::fmt::Debug for FunctionTable<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FunctionTable({} symbols)", self.fns.len())
    }
}
