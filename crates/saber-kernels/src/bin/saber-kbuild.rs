//! Build the committed code objects and their manifests.
//!
//! One plugin per kernel-set stem: resolve the source through the `<k>.gfx1201.hip ->
//! <k>.gfx12.hip -> <k>.hip` ladder, preprocess it to get a build hash that is invariant to
//! comments and include paths, compile it with `tools/hipcc.sh`, then read the code object's own
//! metadata back and write `prebuilt/<arch>/<k>.manifest.toml` next to the blob.
//!
//! Needs the container (it shells `tools/hipcc.sh`); nothing else in this crate does.
//!
//!   cargo run -p saber-kernels --bin saber-kbuild -- [--out DIR] [--arch gfx1201] [stem ...]

use std::path::{Path, PathBuf};
use std::process::Command;

use saber_kernels::manifest::{sha256_hex, BuildInfo, Manifest};
use saber_kernels::{ladder, prebuilt};

const FLAGS: &str = "--genco --offload-arch=gfx1201 -O3";

/// Extra flags a plugin's REFERENCE was built with, which its code object has to inherit or the
/// port is not a port. The reference's `build.sh:35` puts `-ffp-contract=off` on every translation unit
/// and its own header says why: the quantised all-reduce requires it, because the two ranks see
/// the two products in opposite order and contracting either into an FMA makes them disagree by
/// ~1 ULP. The GDN recurrent update needs it for the same reason against the reference -- with
/// contraction on, the state cache diverged from the reference's at 4% of its bytes while attention and
/// the gated norm still matched, which is exactly how a flag difference presents.
///
/// A table, not a habit: the flags land in the manifest and therefore in the build hash.
///
/// `attn_glue` needs it for a different reference and the same reason: prod computes those ops in
/// Triton, which emits a separate `mulf` and `subf` for `a*c - b*s` and truncates once. With
/// contraction on, hipcc fuses them into an FMA that keeps the product at full width, and the
/// rope output disagrees with the reference server's on about 1% of the rotary elements.
const EXTRA_FLAGS: &[(&str, &str)] = &[
    // The attention, GDN and all-reduce family (`saber_attn*`, `saber_gdn_*`, `saber_ar_*`). Not one
    // `saber_` prefix: `saber_argmax_pv`, `saber_device_row_*` and `saber_head_w8_m64` build at the
    // default contraction.
    ("saber_attn", "-ffp-contract=off"),
    ("saber_gdn_", "-ffp-contract=off"),
    ("saber_ar_", "-ffp-contract=off"),
    ("flashnext_ar_host", "-ffp-contract=off"),
    // P27-15: `p27_tree_attn` is the reference's ragged decode translation unit (the `saber_attn` row's
    // reason); `p27_tree_gdn` replaces rx7's recurrent update and is built like it.
    ("p27_tree_", "-ffp-contract=off"),
    // P27-W2-5: the replay update reproduces rx7's recurrent update byte for byte (same flag).
    ("p27w2_", "-ffp-contract=off"),
    ("attn_glue", "-ffp-contract=off"),
    // The vision tower's rope and pos-embed round every ATen bf16 op separately; a contraction
    // would fuse a product into the next add across one of those roundings.
    ("vision", "-ffp-contract=off"),
    // Flash-Next's MoE routing, combine and A16 control (FN4): no upstream reference to inherit a
    // flag from -- the arithmetic is Saber's, DEFINED so a numpy float32 program reproduces it bit
    // for bit (`tools/capture/fn4/moe_ref.py`), which needs every product and sum rounded where the
    // source writes it. `moe_grouped` is deliberately absent: it is the decode GEMM's body and
    // `silu_mul_quant`'s, and must contract exactly where they do.
    ("moe_route", "-ffp-contract=off"),
    // FN148: the NVFP4 oracle is reproduced bit for bit by a numpy f32 program
    // (`tools/capture/fn148/fn148_ref.py`); the W4A8 forms write their one FMA explicitly.
    ("moe_fn148", "-ffp-contract=off"),
    // FN151: byte-equal twins of moe_fn148's form (b) cells (the same flag).
    ("moe_fn151", "-ffp-contract=off"),
    // FN152 (`moe_fn152`) is deliberately absent: its text is moe_grouped_far's (helpers and silu row at
    // the default contraction, the shipped `#pragma clang fp contract(off)` line after them) with FN151's
    // impl placed below that pragma -- every line contracts exactly as in the object it was taken from.
    // FN153: FN148 form (b) in the grouped prefill tile (byte-equal to the ks1 cells; the same flag).
    ("moe_fn153", "-ffp-contract=off"),
    // FN5: the HC kernels' host reference (tools/capture/fn5/hc_ref.py) reproduces every fp32
    // multiply and add as its own IEEE op; a contraction would fuse one across that. The one
    // function that must contract like `mxfp4_quant` (the bf16 gdn-norm twin) says so by pragma.
    ("flashnext_hc", "-ffp-contract=off"),
    // FN6: the attention object is the reference's translation unit at other template arguments and
    // must inherit the reference's flag (the `saber_attn` row's reason); the glue's host spec
    // (tools/capture/fn6/fn6_ref.py) reproduces every fp32 op on its own. `flashnext_moe` is
    // deliberately ABSENT: its W8A8 half is `dflash_gemm`'s body, built with the default
    // contraction, and its bf16 half has no fp32 multiply-add outside the WMMA.
    ("flashnext_attn", "-ffp-contract=off"),
    ("flashnext_glue", "-ffp-contract=off"),
    // FN9b: the exact-gate chunk scan is the reference's scan translation unit with its gate weights
    // re-derived (the retired source generator's SCAN_XG_SUBS): it inherits the `saber_gdn_` row's flag.
    // FN24: `flashnext_gdn_fc` (a prefix of this row) is rx7's decode updates + the f16 carry;
    // it inherits the same flag.
    ("flashnext_gdn", "-ffp-contract=off"),
    // FN19: the QSA selection (score / exact top-512 / union) is Saber's own arithmetic, DEFINED so
    // the numpy spec (tools/capture/fn19/qsa_spec.py) reproduces it bit for bit; the gather
    // attention (`flashnext_qsa_attn`, a prefix of this row) is the reference's TU with the key axis
    // re-pointed and inherits the `saber_attn` row's flag.
    ("flashnext_qsa", "-ffp-contract=off"),
    // FN22: the drafter's device-side chain (argmax -> embedding row) is an integer law over bf16
    // bit patterns; nothing contracts, the flag only keeps the row's one line.
    ("flashnext_draft", "-ffp-contract=off"),
    // FN57: the step's seam pick on the card -- the same integer key law over bf16 bit patterns.
    ("flashnext_pick", "-ffp-contract=off"),
    // FN119: the device sampler twin reproduces the host's f32 / f64 ops one by one (the FMAs
    // glibc's expf contracts are written as `__builtin_fma`); nothing else may contract.
    ("flashnext_dstep", "-ffp-contract=off"),
    // LR-N: coupled drafting's chain tail -- drafting is free (the walk keeps the target's draw);
    // the flag keeps the drafter family's line uniform.
    ("flashnext_couple", "-ffp-contract=off"),
    // FN119 round 2: includes flashnext_dstep's source verbatim -- the same flag, the same ops.
    ("flashnext_ahead", "-ffp-contract=off"),
    // FN114b: the flag-in-data exact reduce carries the reference's `vadd` verbatim and must inherit the
    // `saber_ar_` row's flag so its fp32 add lowers as the one-shot's does.
    ("flashnext_ar", "-ffp-contract=off"),
    // P27-W2-2: the 27B drafter's device candidate seam -- comparisons only, built like every
    // exact law.
    ("p27_dstep", "-ffp-contract=off"),
];

// The OPPOSITE case, twice over, because the instinct is to reach for the same flag everywhere.
//
// `mxfp4_quant` (the producers) is absent from the table, and that is a DECISION: prod builds its
// reference with contraction on. `serve-mxfp4.sh:1041` -- the line that compiles
// `ref_mxfp4_fp8.hip` at serve time, and `Dockerfile.ggz14:345` which bakes the same one --
// is `hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201`, with no `-ffp-contract`
// anywhere in the launcher, so hipcc's HIP default (`fast-honor-pragmas`) applies. Measured on the
// ISA rather than argued (2026-09-20, `-save-temps` on both sources, fp32 mix per kernel):
//
//   kernel                 ours (shipped flags)         prod's reference, same flags
//   add_rms_quant_t512_1   fmac 12 + dual 2, mul 12     fmac 13 + dual 1, mul 12   -- 14 each
//   gdn_norm_quant_t256    fma 35 fmac 16 mul 37        fma 35 fmac 16 mul 37      -- identical
//   silu_mul_quant_t512_1  fma 34 fmac 15 mul 24        fma 34 fmac 15 mul 24      -- identical
//
// and with `-ffp-contract=off` BOTH sides change: `add_rms_quant` drops to fmac 5-6 with mul/add
// 18/20, `gdn_norm_quant` loses four. So contraction is live in two of the three producers -- the
// clearest site is the sum of squares, `ssq += v * v`, one `v_fmac_f32` per element -- and the
// 72/72 producer gate passes because the two builds contract in the same places, not by luck of
// the expressions. Adding the flag here would BREAK it.
//
// The reference's own comment ("every intermediate is rounded exactly where the traced path
// rounds") is sliced verbatim into `mxfp4_quant.hip` and is a claim about the explicit
// `(float)(__bf16)` casts, which contraction cannot cross; the fp32 arithmetic between them
// rounds where the FLAG says, and the flag is only right because it is prod's. Read it as a
// statement about the casts plus the build line, not about the expressions.
//
// And the drafter's int2 head must be built WITH contraction, the default. Triton contracts --
// `_draft_head_int2.amdgcn` carries `v_fma_f32` for exactly the accumulator updates `acc += dot * s`
// and `acc -= xs * zs` -- so `-ffp-contract=off` there produced a separate multiply and add, and
// the head's severe cancellation (each group is a small difference of two large numbers) turned
// that single lost rounding into 1-4 ULP on 79% of the block maxima. The flag is not a habit in
// either direction; it is a property of the reference, and the reference's disassembly is where it
// is read.

fn extra_flags(stem: &str) -> &'static str {
    EXTRA_FLAGS
        .iter()
        .find(|(prefix, _)| stem.starts_with(prefix))
        .map(|(_, f)| *f)
        .unwrap_or("")
}

/// The repo root, canonical: every path handed to `tools/hipcc.sh` is relative to it, because the
/// container sees the repo mounted at /work and nothing above it.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root")
}

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or_else(|_| panic!("{} is not under {}", p.display(), root.display()))
        .to_string_lossy()
        .into_owned()
}

fn hipcc(root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let sh = root.join("tools").join("hipcc.sh");
    let out = Command::new(&sh)
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("{}: {e}", sh.display()))?;
    if !out.status.success() {
        return Err(format!(
            "hipcc.sh {args:?} failed ({}):\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out.stdout)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut arch = prebuilt::ARCH.to_owned();
    let mut out_dir: Option<PathBuf> = None;
    let mut stems: Vec<String> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--arch" => arch = args.next().expect("--arch needs a value"),
            "--out" => out_dir = Some(PathBuf::from(args.next().expect("--out needs a value"))),
            other => stems.push(other.to_owned()),
        }
    }
    if stems.is_empty() {
        stems = prebuilt::ALL
            .iter()
            .map(|(b, _)| b.trim_end_matches(".hsaco").to_owned())
            .collect();
    }
    let root = repo_root();
    let hip = root.join("crates").join("saber-kernels").join("hip");
    let out = out_dir.unwrap_or_else(|| root.join("prebuilt").join(&arch));
    std::fs::create_dir_all(&out).expect("create the output directory");

    let mut failed = false;
    for stem in &stems {
        match build_one(&root, &hip, &out, stem, &arch) {
            Ok(msg) => println!("{msg}"),
            Err(e) => {
                eprintln!("FAILED {stem}: {e}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}

fn build_one(
    root: &Path,
    hip: &Path,
    out: &Path,
    stem: &str,
    arch: &str,
) -> Result<String, String> {
    let src = ladder::resolve_in(hip, stem, arch)
        .ok_or_else(|| format!("no source for `{stem}` under {}", hip.display()))?;
    let src_rel = rel(root, &src);
    let src_bytes = std::fs::read(&src).map_err(|e| e.to_string())?;

    // Build hash = sha256(preprocessed source ++ flags). Preprocessed, so a comment edit does not
    // invalidate a blob and an include that moved does.
    let pp_path = root
        .join(".cache")
        .join("kbuild")
        .join(format!("{stem}.{arch}.i"));
    std::fs::create_dir_all(pp_path.parent().expect("parent")).map_err(|e| e.to_string())?;
    let pp_rel = rel(root, &pp_path);
    hipcc(root, &[&src_rel, &pp_rel, "-E"])?;
    let extra = extra_flags(stem);
    let flags = if extra.is_empty() {
        FLAGS.to_owned()
    } else {
        format!("{FLAGS} {extra}")
    };
    let mut pp = std::fs::read(&pp_path).map_err(|e| e.to_string())?;
    pp.extend_from_slice(flags.as_bytes());
    let build_hash = sha256_hex(&pp);

    let hsaco_name = format!("{stem}.hsaco");
    let hsaco = out.join(&hsaco_name);
    let hsaco_rel = rel(root, &hsaco);
    let mut args = vec![src_rel.as_str(), hsaco_rel.as_str()];
    if !extra.is_empty() {
        args.push(extra);
    }
    hipcc(root, &args)?;
    let blob = std::fs::read(&hsaco).map_err(|e| e.to_string())?;

    let m = Manifest::from_code_object(
        &BuildInfo {
            plugin: stem,
            arch,
            source: &src_rel,
            source_sha256: &sha256_hex(&src_bytes),
            flags: &flags,
            build_hash: &build_hash,
            hsaco_name: &hsaco_name,
        },
        &blob,
    )
    .map_err(|e| e.to_string())?;
    m.validate_against_hsaco(&blob).map_err(|e| e.to_string())?;
    let mpath = out.join(format!("{stem}.manifest.toml"));
    m.save(&mpath).map_err(|e| e.to_string())?;
    Ok(format!(
        "{stem}: {} kernels, {} KiB blob, build_hash {}",
        m.kernel.len(),
        blob.len() / 1024,
        &m.build_hash[..12]
    ))
}
