//! The Flash-Next drivers (FN12): gate A (TP=1, layer-major, byte-for-byte against the FN9/FN9b
//! Python harness's hashes) and the engine (TP=2 / EP=2, both cards, one process, the expert tier
//! and the PLE tier, prefill only).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context};
use saber_exec::loader::{FunctionTable, PluginSet};
use saber_exec::LoadCounters;
use saber_hip::{DevicePtr, Event, EventFlags, HipThread, Job, MemcpyKind, PinnedBuf, Stream};
use saber_kernels::flashnext::FlashNextRules;
use saber_kernels::manifest::sha256_hex;
use saber_mem::device_mover::{DeviceMover, HipCopyQueue};
use saber_mem::dio::OpenMode;
use saber_mem::expert_mover::parallel_reads;
use saber_mem::expert_tier::{ExpertTier, PathCounts, Placement, RouteProfile};
use saber_mem::flashnext_budget::{
    geometry, weights_from_plan, Ep2Budget, HostBudget, RuntimeParams,
};
use saber_mem::ledger::Ledger;
use saber_mem::ple_tier::{PleGate, PleIndex, PleSourceKind, PleTier};
use saber_mem::xtier::XtierFile;
use saber_model::qwen4_exp::layer_graph::Group;
use saber_model::qwen4_exp::ple_hash::{PleContext, PleHash};
use saber_model::qwen4_exp::Qwen4ExpConfig;
use serde_json::{json, Value};

use super::dev::{ArPair, ArRules, RankDev};
use super::exec::{
    emit_layer, emit_post, ChunkConsts, Ctl, Dims, Emitter, HcW, LayerIn, LayerOut, LayerState,
    LayerW, Pending, Reduce, Resident, XptrHook, H, PLE_HIST, RING, VOCAB, W,
};
use super::kern::{Arena, ArenaSizes, Bump, Decls, ALIGN, PLUGINS};
use super::model::{self, Owned, Src, P};
use super::plan::{self, LAYERS, PLE_LAYER};
use super::store::{embed_rows, Ckpt, PackRecs};
use crate::xtier_flags::XtierFlags;

/// The NARROW prefill rung (FN19's cut: QSA == dense up to 2,048 tokens; plan §8 F3): the chunk a
/// prompt takes beside decode, and the default widest chunk. FN38: a boot's widest chunk is
/// [`FnFlags::max_chunk`] (`--fn-max-chunk`); every buffer a chunk fills is sized to THAT.
pub const MAX_ROWS: u32 = 2048;
/// FN38: the widest chunk `--fn-max-chunk` may name.
pub const WIDE_ROWS_MAX: u32 = 8192;
/// FN38: `--fn-max-chunk`'s default. FN44 (the user's word, 2026-09-30 ~20:30 CDT, on FN43b's
/// numbers): 8,192 rows -- with the arena borrow, `always` and the 128-slot ring above 2,048 rows
/// (8k prompt 2,429 -> 1,169 ms; served 8k load +40 %, ITL p50 +5 %). The control arm is the old
/// default by explicit flags: `--fn-max-chunk 2048 --no-fn-arena-borrow --fn-wide-when idle
/// --no-xtier-stage`.
pub const DEFAULT_FN_MAX_CHUNK: u32 = WIDE_ROWS_MAX;
/// FN45c2: the prompt cache's ceiling a rank (`--fn-kv-pcache-gib`).
pub const PCACHE_GIB_MAX: f64 = 128.0;
/// FN45c3: the prompt / conversation cache's budget a rank when it is on by default (both ranks:
/// 2 x 16 GiB on the drive, LRU, in `--fn-kv-dir`).
pub const DEFAULT_FN_KV_PCACHE_GIB: f64 = 16.0;
/// FN45c3: whether a `--fn-serve` boot under the tiered KV opens the cache with no flag. NOT YET
/// (LEDGER FN45c3): byte laws, BetterBench, the 4k/8k load and the API ids passed on 4 + 4 boots,
/// but the returning-conversation load stalled on 2 of 4 cache boots (TTFT p90 1.1 -> 1.8 s, wall
/// +21..+50 %) -- the flip waits for that fix. `--fn-kv-pcache-gib 16` = the arm;
/// `--no-fn-kv-pcache` = the control once this is true.
/// FN68b: the QSA K/V page dtype a plain boot runs (`--fn-kv-dtype`).
pub const DEFAULT_FN_KV_DTYPE: &str = "fp8"; // FN68b: fp8 + the pack's scale law, the user's word 2026-10-08 ("yes to 2 and 3"); bf16 = the control
pub const PCACHE_SERVED_DEFAULT: bool = false;
/// FN68d: the MTP layer's (48's) K/V page dtype under fp8 pages (`--fn-kv-mtp-dtype`): `bf16` =
/// its own one-layer bf16 tier (its ~750-nat logits cannot survive K at e4m3, LEDGER FN68c);
/// `fp8` = one dtype for all 13 QSA layers (FN68b's boot, the control).
pub const DEFAULT_FN_KV_MTP_DTYPE: &str = "bf16"; // FN68d (2026-10-08): single +3.5 %, conc-8 held, target bytes 25/25; fp8 = the control
/// FN113 / FN113b: `--fn-prefix-tree` at a plain boot -- `on` IS THE DEFAULT (the user's word
/// 2026-10-13 08:15 on FN113b h9: every class == the verify arm, update p99 and single inside the
/// `off` boots' range, conc-8 0.8 % low on a no-prefix workload -- accepted; TTFT on a cached
/// prefix: turn 102 -> 61 ms, sys 163 -> 56, an 8.7k chunk 1,553 -> 220); `off` = the control.
/// A served boot WITHOUT the flag stands the tree down where it cannot run (the flat KV control,
/// `--no-fn-kv-pcache`, FN66's `--warm-prompts`); an explicit `on` there is refused.
pub const DEFAULT_FN_PREFIX_TREE: &str = "on";
/// FN113b: the tree's drive tier a rank (`kv.pcache`) when `--fn-kv-pcache-gib` is absent (the
/// held setting: 4 GiB a rank).
pub const DEFAULT_FN_PREFIX_TREE_DRIVE_GIB: f64 = 4.0;
/// FN113: the tree's pinned RAM tier a rank (1.5 GiB = 3.0 GiB a box, inside the 4.0 GiB X1 freed).
pub const DEFAULT_FN_PREFIX_TREE_RAM_GIB: f64 = 1.5;
/// FN113b: where the tree stores a prompt's END node (`--fn-prefix-tree-end`): `live` = `side`
/// WITHOUT the m - 1 row drafter pass (the node keeps the drafter rows as the prompt's own m-row
/// pass leaves them -- drafter-only bytes; h1 attributed 3.2 of the 6.7 ms END block to that
/// pass); `side` = its
/// copies OFF the step (the mover's stream into the node's pinned pages, the victim's demotion on
/// the host under the drafter's m - 1 row pass, one fence before the next command); `step` =
/// FN113's store inside `Cmd::First` on the decode step (the control); `off` = a DIAGNOSTIC (END
/// nodes never written: the attribution arm of the update-p99 term).
pub const DEFAULT_FN_PREFIX_TREE_END: &str = "live";

/// FN45c3: the shortest prefix a conversation cut stores (`--fn-kv-pcache-min-tokens`).
pub const DEFAULT_FN_KV_PCACHE_MIN_TOKENS: u32 = 512;
/// FN66: the conversation cut's floor when `--warm-prompts` turns the cache on (FN45c4's tied
/// setting: no served chat prompt reaches it, so a turn pays no cut; the warm prompts' own cut
/// is the warm law, `pcache::warm_cut`).
pub const WARM_FN_KV_PCACHE_MIN_TOKENS: u32 = 1_000_000;
/// FN45c4: the conversation cut's alignment (`--fn-kv-pcache-align`). 1 = FN45c3's cut: the
/// 64-aligned cut did NOT shrink the byte change vs no cache under the f16 GDN state (text 10/50
/// vs 9/50) and its KL was WORSE (46/50 under 0.0227, max 0.092, vs 50/50, max 0.019).
pub const DEFAULT_FN_KV_PCACHE_ALIGN: u32 = 1;
/// FN45c4: the drive tier (`--fn-kv-ssd-gib`) is REFUSED at boot until FN45c3's fault is
/// understood (its 64k boot faulted on BOTH cards in `fn_draft_wgather` on its first 16k request).
pub const SSD_TIER_REFUSED: bool = true;
/// FN38: `--fn-wide-when`'s default. FN44: `always` (a wide chunk runs beside decode too).
pub const DEFAULT_FN_WIDE_WHEN: WideWhen = WideWhen::Always;

/// FN38: `--fn-max-chunk` is one of the rungs 2,048 / 4,096 / 8,192.
/// LR-I: `--fn-prefix-tree-cuts` (normalized to the rule's spelling).
fn parse_tree_cuts(s: &str) -> Result<String, String> {
    saber_mem::prefix_tree::CutRule::parse(s).map(saber_mem::prefix_tree::CutRule::name)
}

fn parse_max_chunk(s: &str) -> Result<u32, String> {
    match s.trim().parse::<u32>() {
        Ok(n @ (2048 | 4096 | 8192)) => Ok(n),
        _ => Err(format!("--fn-max-chunk 2048 | 4096 | 8192, not {s}")),
    }
}

/// FN42b: `--fn-arena-borrow`'s default. FN44: `on` (the default stands down where the borrow
/// cannot run -- the one-process form, the poison arms -- instead of refusing; an explicit
/// `--fn-arena-borrow on` there is still refused by name).
pub const DEFAULT_FN_ARENA_BORROW: OnOff = OnOff::On;

/// FN42b: an `on | off` flag value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnOff {
    On,
    Off,
}

impl std::str::FromStr for OnOff {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "on" => Ok(OnOff::On),
            "off" => Ok(OnOff::Off),
            t => Err(format!("on | off, not {t}")),
        }
    }
}

impl std::fmt::Display for OnOff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if *self == OnOff::On { "on" } else { "off" })
    }
}

/// FN38: when the serve plane takes the wide rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideWhen {
    /// Only while no sequence decodes (beside decode a chunk keeps [`MAX_ROWS`]).
    Idle,
    /// Every chunk.
    Always,
}

impl std::str::FromStr for WideWhen {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "idle" => Ok(WideWhen::Idle),
            "always" => Ok(WideWhen::Always),
            t => Err(format!("--fn-wide-when idle | always, not {t}")),
        }
    }
}

impl std::fmt::Display for WideWhen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            WideWhen::Idle => "idle",
            WideWhen::Always => "always",
        })
    }
}

impl FnFlags {
    /// FN45b2: the tiered KV at this boot (`--fn-kv-tier`; absent = `kvwire::DEFAULT_FN_KV_TIER`
    /// where the tier can run -- the worker form at TP=2 with sparse QSA -- else off). An explicit
    /// `on` where it cannot run is refused by [`FnFlags::kv_tier_check`].
    #[must_use]
    pub fn kv_tier(&self) -> bool {
        match self.fn_kv_tier {
            Some(v) => v == OnOff::On,
            None => super::kvwire::DEFAULT_FN_KV_TIER && self.kv_tier_ok(),
        }
    }

    /// FN51f-b: ELASTIC VRAM at this boot (`--fn-vram-elastic`, else the default) -- the tiered
    /// KV only (a flat-pool boot has no hot pool to lend).
    #[must_use]
    pub fn vram_elastic(&self) -> bool {
        self.kv_tier()
            && match self.fn_vram_elastic {
                Some(v) => v == OnOff::On,
                None => DEFAULT_FN_VRAM_ELASTIC,
            }
    }

    fn kv_tier_ok(&self) -> bool {
        !self.fn_one_process && self.fn_qsa == saber_kernels::fn19::QsaMode::Sparse
    }

    /// FN45b2: refuse `--fn-kv-tier on` where the tier cannot run.
    pub fn kv_tier_check(&self, tp: u32) -> anyhow::Result<()> {
        if let Some(w) = &self.warm_prompts {
            ensure!(
                self.fn_serve.is_some(),
                "--warm-prompts {} runs in the serve plane (--fn-serve PORT)",
                w.display()
            );
            ensure!(
                self.pcache_gib().is_some(),
                "--warm-prompts {} fills the PROMPT CACHE: not with --no-fn-kv-pcache or the \
                 flat pools (--fn-kv-tier off)",
                w.display()
            );
        }
        if self.fn_kv_tier == Some(OnOff::On) {
            ensure!(
                tp == 2 && self.kv_tier_ok(),
                "--fn-kv-tier on runs in the worker form at TP=2 with --fn-qsa sparse only"
            );
        }
        if self.kv_dtype() == saber_mem::kv_tier::KvDtype::Bf16 {
            ensure!(
                self.fn_kv_scales.is_none(),
                "--fn-kv-scales is the fp8 pages' scale law: not with --fn-kv-dtype bf16"
            );
            ensure!(
                self.fn_kv_mtp_dtype.as_deref() != Some("fp8"),
                "--fn-kv-mtp-dtype fp8 under --fn-kv-dtype bf16: the MTP layer at fp8 beside bf16 \
                 target pages is not a mode"
            );
        }
        ensure!(
            !(self.fn_one_process && self.fn_kv_mtp_dtype.as_deref() == Some("bf16")),
            "--fn-kv-mtp-dtype bf16 runs in the worker form (the one-process control keeps one \
             page dtype)"
        );
        // (the drive tier is refused above all; `KvDev::new` refuses a twin beside one by name)
        if let Some(g) = self.fn_kv_ssd_gib {
            ensure!(
                !SSD_TIER_REFUSED,
                "--fn-kv-ssd-gib {g}: the drive tier is REFUSED until its fault is understood \
                 (LEDGER FN45c3: the 64k boot faulted on both cards in fn_draft_wgather on its \
                 first 16k request; FN45c4 keeps it unreachable). Serve without it (two tiers)."
            );
            ensure!(
                g > 0.0 && g <= super::kvwire::SSD_GIB_MAX,
                "--fn-kv-ssd-gib {g}: 0 < GiB <= {} a rank (the drive is shared by both ranks)",
                super::kvwire::SSD_GIB_MAX
            );
        }
        if let Some(g) = self.fn_kv_pcache_gib {
            ensure!(
                g > 0.0 && g <= PCACHE_GIB_MAX,
                "--fn-kv-pcache-gib {g}: 0 < GiB <= {PCACHE_GIB_MAX} a rank"
            );
        }
        if self.fn_prefix_tree.as_deref() == Some("on") {
            ensure!(
                self.fn_serve.is_some() && self.kv_tier(),
                "--fn-prefix-tree on needs --fn-serve and the tiered KV"
            );
            ensure!(
                !self.no_fn_kv_pcache,
                "--fn-prefix-tree on keeps its nodes on the drive: not with --no-fn-kv-pcache"
            );
            let g = self.fn_prefix_tree_ram_gib;
            ensure!(
                (0.0625..=8.0).contains(&g),
                "--fn-prefix-tree-ram-gib {g}: 0.0625..8 GiB a rank"
            );
        }
        ensure!(
            self.fn_prefix_tree_cuts == "today" || self.prefix_tree(),
            "--fn-prefix-tree-cuts {}: the tree is off (--fn-prefix-tree on --fn-serve PORT)",
            self.fn_prefix_tree_cuts
        );
        ensure!(
            self.fn_prefix_tree_end == DEFAULT_FN_PREFIX_TREE_END || self.prefix_tree(),
            "--fn-prefix-tree-end {}: the tree is off (--fn-prefix-tree on --fn-serve PORT)",
            self.fn_prefix_tree_end
        );
        ensure!(
            self.prefix_tree() || !(self.kv_admit(self.max_ctx()) && self.pcache_gib().is_some()),
            "--fn-kv-pcache-gib stores a slot's identity index pages: not under admission by \
             reservation (--fn-max-ctx past {}; --no-fn-kv-admit is the control)",
            super::kvwire::DEFAULT_FN_KV_MAX_CTX
        );
        if let Some(t) = self.fn_kv_admit_slab_tokens {
            let ctx = self.max_ctx();
            ensure!(
                self.kv_admit(ctx),
                "--fn-kv-admit-slab-tokens {t}: the admission slab exists under admission by \
                 reservation only (the tiered KV, --fn-max-ctx past {}, no --no-fn-kv-admit / \
                 --fn-kv-ram-overcommit / --fn-kv-ssd-gib / --warm-prompts)",
                super::kvwire::DEFAULT_FN_KV_MAX_CTX
            );
            let floor = super::kvwire::admit_slab_tokens(ctx);
            ensure!(
                t >= floor,
                "--fn-kv-admit-slab-tokens {t}: under the default slab ({floor} tokens a rank at \
                 --fn-max-ctx {ctx}: one max-context request + one {}-token request + the \
                 extras); the arm only grows it",
                super::kvwire::DEFAULT_FN_KV_MAX_CTX
            );
        }
        if !self.kv_tier() {
            ensure!(
                self.fn_kv_ssd_gib.is_none() && self.fn_kv_pcache_gib.is_none(),
                "--fn-kv-ssd-gib / --fn-kv-pcache-gib need the tiered KV (--fn-kv-tier on)"
            );
            ensure!(
                self.fn_kv_hot_gib.is_none()
                    && self.fn_kv_ram_gib.is_none()
                    && !self.fn_kv_drop_flush
                    && !self.no_fn_kv_gather
                    && !self.fn_kv_ram_overcommit,
                "--fn-kv-hot-gib / --fn-kv-ram-gib / --fn-kv-drop-flush / --no-fn-kv-gather / \
                 --fn-kv-ram-overcommit need the tiered KV (--fn-kv-tier on)"
            );
            ensure!(
                self.fn_vram_elastic != Some(OnOff::On),
                "--fn-vram-elastic on lends the tiered KV's hot pool (--fn-kv-tier on)"
            );
        }
        ensure!(
            self.fn_vram_elastic_kv_max_gib >= 0.0 && self.fn_vram_elastic_kv_max_gib <= 16.0,
            "--fn-vram-elastic-kv-max-gib {}: 0 <= GiB <= 16 a rank",
            self.fn_vram_elastic_kv_max_gib
        );
        Ok(())
    }

    /// FN45b4: the served max context -- `--fn-max-ctx`, else under the tiered KV
    /// `kvwire::DEFAULT_FN_SERVED_MAX_CTX` (262,144 by admission by reservation, FN51b) on a
    /// `--fn-serve` boot and the pinned-RAM ceiling (`kvwire::DEFAULT_FN_KV_MAX_CTX`) for the
    /// harness / scripted runs, else today's 8,192.
    #[must_use]
    pub fn max_ctx(&self) -> u32 {
        self.fn_max_ctx.unwrap_or(if !self.kv_tier() {
            8192
        } else if self.fn_serve.is_some() && self.warm_prompts.is_none() {
            super::kvwire::DEFAULT_FN_SERVED_MAX_CTX
        } else {
            super::kvwire::DEFAULT_FN_KV_MAX_CTX
        })
    }

    /// FN51: ADMISSION BY RESERVATION -- the tiered KV past the pinned-RAM ceiling
    /// (`kvwire::DEFAULT_FN_KV_MAX_CTX`) on the two tiers, unless `--no-fn-kv-admit` (the FN45b4
    /// law: 8 x the max context pinned, refused past the host) or a gate arm's own slab law
    /// (`--fn-kv-ram-overcommit`, `--fn-kv-ssd-gib`).
    #[must_use]
    pub fn kv_admit(&self, max_ctx: u32) -> bool {
        self.kv_tier()
            && !self.no_fn_kv_admit
            && !self.fn_kv_ram_overcommit
            && self.fn_kv_ssd_gib.is_none()
            && self.warm_prompts.is_none()
            && max_ctx > super::kvwire::DEFAULT_FN_KV_MAX_CTX
    }

    /// FN45b2: the tier's sizes at this boot (None = the flat pools).
    #[must_use]
    pub fn kv_geom(&self, max_ctx: u32) -> Option<super::kvwire::KvGeom> {
        self.kv_geom_for(max_ctx, max_ctx)
    }

    /// [`FnFlags::kv_geom`] with a harness sequence's admission `first` (FN51: its prompt +
    /// decode under admission by reservation; the sizes do not depend on it).
    #[must_use]
    pub fn kv_geom_for(&self, max_ctx: u32, first: u32) -> Option<super::kvwire::KvGeom> {
        self.kv_tier().then(|| {
            let mut g = super::kvwire::KvGeom::new_dt(
                max_ctx,
                self.max_chunk(),
                self.fn_kv_hot_gib,
                self.fn_kv_ram_gib,
                self.kv_dtype(),
            );
            let mut lent = 0;
            if self.vram_elastic() {
                // FN51f-b: the region = the boot window + the borrow (the exact ledger carves it
                // out of the expert static slots; idle, it is lent straight back)
                let bb = g.bb();
                let extra = (self.fn_vram_elastic_kv_max_gib.max(0.0) * (1u64 << 30) as f64) as u64;
                let h = (g.hot_bytes + extra) / bb * bb;
                lent = h - g.hot_bytes;
                g.hot_bytes = h;
            }
            g.count_ram = self.fn_kv_pages;
            g.drop_flush = self.fn_kv_drop_flush;
            g.gather = !self.no_fn_kv_gather;
            let g = if self.kv_admit(max_ctx) {
                let mut g = g.with_admission(self.fn_kv_ram_gib, first);
                // LR-K: the slab in tokens (absent: `admit_slab_tokens`, byte-identical)
                if let Some(t) = self.fn_kv_admit_slab_tokens {
                    g = g.with_admit_slab_tokens(t);
                }
                // LR-G: lazy growth rides admission by reservation only
                g.lazy = self.kv_lazy() > 0;
                g
            } else {
                match self.fn_kv_ssd_gib {
                    Some(s) => g.with_ssd(s, self.fn_kv_ram_gib).unwrap_or(g),
                    None => g,
                }
            };
            // FN68d: the MTP layer's own tier at its dtype (the elastic borrow stays the main's)
            g.split_mtp(self.kv_mtp_dtype(), lent)
        })
    }

    /// LR-G: the lazy headroom in tokens (`--fn-kv-lazy`; 0 = off).
    #[must_use]
    pub fn kv_lazy(&self) -> u32 {
        self.fn_kv_lazy.unwrap_or(0)
    }

    /// FN45c2: rank `rank`'s drive slab file (`--fn-kv-ssd-gib`), None without the drive tier.
    #[must_use]
    pub fn kv_ssd_path(&self, rank: u32) -> Option<PathBuf> {
        self.fn_kv_ssd_gib
            .map(|_| self.fn_kv_dir.join(format!("kv-rank{rank}.ssd")))
    }

    /// FN45c3: the prompt cache's budget a rank in GiB at this boot -- `--fn-kv-pcache-gib`, else
    /// [`DEFAULT_FN_KV_PCACHE_GIB`] on a `--fn-serve` boot under the tiered KV, unless
    /// `--no-fn-kv-pcache` (None = off).
    #[must_use]
    pub fn pcache_gib(&self) -> Option<f64> {
        if self.no_fn_kv_pcache {
            return None;
        }
        if self.prefix_tree() {
            return Some(
                self.fn_kv_pcache_gib
                    .unwrap_or(DEFAULT_FN_PREFIX_TREE_DRIVE_GIB),
            );
        }
        self.fn_kv_pcache_gib.or_else(|| {
            ((PCACHE_SERVED_DEFAULT || self.warm_prompts.is_some())
                && self.fn_serve.is_some()
                && self.kv_tier())
            .then_some(DEFAULT_FN_KV_PCACHE_GIB)
        })
    }

    /// FN113: the tiered prefix tree is on (`--fn-prefix-tree on` on a `--fn-serve` boot; FN113b:
    /// the DEFAULT, standing down without the tiered KV, the drive cache or under
    /// `--warm-prompts`).
    #[must_use]
    pub fn prefix_tree(&self) -> bool {
        self.fn_serve.is_some()
            && match self.fn_prefix_tree.as_deref() {
                Some(v) => v == "on",
                None => {
                    DEFAULT_FN_PREFIX_TREE == "on"
                        && self.kv_tier()
                        && !self.no_fn_kv_pcache
                        && self.warm_prompts.is_none()
                }
            }
    }

    /// FN113: the tree's switches (None = off).
    #[must_use]
    pub fn prefix_tree_opts(&self) -> Option<super::worker::TreeOpts> {
        self.prefix_tree().then_some(super::worker::TreeOpts {
            ram_bytes: (self.fn_prefix_tree_ram_gib * (1u64 << 30) as f64) as u64,
            min_gap: self.fn_prefix_tree_min_gap,
            sys_min: self.fn_prefix_tree_sys_min,
            cuts: saber_mem::prefix_tree::CutRule::parse(&self.fn_prefix_tree_cuts)
                .unwrap_or_default(),
            lambda: crate::xtier_flags::DEFAULT_XTIER_LRFU_LAMBDA,
        })
    }

    /// FN113: the tree's pinned RAM bytes a rank (0 = off).
    #[must_use]
    pub fn prefix_tree_ram_bytes(&self) -> u64 {
        self.prefix_tree_opts().map_or(0, |o| {
            o.ram_bytes / super::worker::TREE_PAGE * super::worker::TREE_PAGE
        })
    }

    /// FN45c3 / FN66: the conversation cut's floor -- `--fn-kv-pcache-min-tokens`, else
    /// [`WARM_FN_KV_PCACHE_MIN_TOKENS`] under `--warm-prompts`, else
    /// [`DEFAULT_FN_KV_PCACHE_MIN_TOKENS`].
    #[must_use]
    pub fn pcache_min_tokens(&self) -> u32 {
        self.fn_kv_pcache_min_tokens
            .unwrap_or(if self.warm_prompts.is_some() {
                WARM_FN_KV_PCACHE_MIN_TOKENS
            } else {
                DEFAULT_FN_KV_PCACHE_MIN_TOKENS
            })
    }

    /// FN45c2: rank `rank`'s prompt-cache directory and budget in bytes (`--fn-kv-pcache-gib`).
    #[must_use]
    pub fn kv_pcache(&self, rank: u32) -> Option<(PathBuf, u64)> {
        self.pcache_gib().map(|g| {
            (
                self.fn_kv_dir.join(format!("pcache-rank{rank}")),
                (g * (1u64 << 30) as f64) as u64,
            )
        })
    }

    /// FN38: this boot's widest prefill chunk (`--fn-max-chunk`).
    #[must_use]
    pub fn max_chunk(&self) -> u32 {
        self.fn_max_chunk
            .unwrap_or(DEFAULT_FN_MAX_CHUNK)
            .clamp(MAX_ROWS, WIDE_ROWS_MAX)
    }

    /// FN38: `--fn-wide-when` (FN44: [`DEFAULT_FN_WIDE_WHEN`] when absent).
    #[must_use]
    pub fn wide_when(&self) -> WideWhen {
        self.fn_wide_when.unwrap_or(DEFAULT_FN_WIDE_WHEN)
    }

    /// FN42b: the wide arena is BORROWED from the expert tier (`--fn-arena-borrow on` with a
    /// wide rung; the narrow rung has nothing to borrow). FN44: absent = [`DEFAULT_FN_ARENA_BORROW`]
    /// where the borrow can run (the worker form without the poison arms), else the permanent cut;
    /// `--no-fn-arena-borrow` / `--fn-arena-borrow off` = the control.
    #[must_use]
    pub fn arena_borrow(&self) -> bool {
        let on = if self.no_fn_arena_borrow {
            false
        } else {
            match self.fn_arena_borrow {
                Some(v) => v == OnOff::On,
                None => {
                    DEFAULT_FN_ARENA_BORROW == OnOff::On
                        && !self.fn_one_process
                        && self.fn_poison.is_none()
                        && self.fn_poison_step.is_none()
                }
            }
        };
        on && self.max_chunk() > MAX_ROWS
    }

    /// FN38: the chunk cap while some sequence decodes (0 = no cap): [`MAX_ROWS`] under
    /// `--fn-wide-when idle` with a wide rung, else none.
    #[must_use]
    pub fn busy_chunk(&self) -> u32 {
        if self.max_chunk() > MAX_ROWS && self.wide_when() == WideWhen::Idle {
            MAX_ROWS
        } else {
            0
        }
    }

    /// FN55c: the vision form of this boot. An explicit `--fn-vision on|lazy` is taken as given
    /// (the boot refuses where it cannot run); an ABSENT flag is [`DEFAULT_FN_VISION`] where it
    /// can stand -- the worker form, without the poison arms, on FN8's tier law (a lent slot),
    /// with the pack's `vision/` member -- else `off`.
    #[must_use]
    pub fn vision_mode(&self, pack: &std::path::Path) -> VisionMode {
        let parse = |v: &str| match v {
            "on" => VisionMode::On,
            "lazy" => VisionMode::Lazy,
            _ => VisionMode::Off,
        };
        match self.fn_vision.as_deref() {
            Some(v) => parse(v),
            None => {
                let m = parse(DEFAULT_FN_VISION);
                let can = self.ring_default_ok()
                    && !self.xtier.xtier_defer
                    && self.xtier.xtier_demote == "ram"
                    && pack.join(super::vision::DIR).join("rank0.pack").exists();
                if m == VisionMode::Off || can {
                    m
                } else {
                    VisionMode::Off
                }
            }
        }
    }

    /// FN55c: the tower's arena in patches (`--fn-vision-max-patches`, else the default).
    #[must_use]
    pub fn vision_patches(&self) -> u32 {
        self.fn_vision_max_patches
            .unwrap_or(DEFAULT_FN_VISION_MAX_PATCHES)
    }

    /// FN43c: whether the DEFAULT staging ring may stand at this boot (TP=2 plans only): the
    /// worker form without the poison arms and the FN30b alias (they write / re-point the ring's
    /// lent slots). Where it may not, the default stands down to UVA in place; an explicit
    /// `--xtier-stage` is refused by the boot instead (`plan_boot`).
    #[must_use]
    pub fn ring_default_ok(&self) -> bool {
        !self.fn_one_process
            && self.fn_poison.is_none()
            && self.fn_poison_step.is_none()
            && !self.fn_diag_alias_ram
    }
}
/// Positions the cos/sin table covers at least (FN19: the long-context contexts; 8 MiB of bf16 a
/// rank). FN51: a boot whose pools' context is longer covers its context ([`rope_pos`]); a row of
/// the table is a function of its position only, so a longer table is byte-identical below 65,536.
pub const ROPE_POS: u32 = 65_536;
/// FN51: the Flash-Next plane's pinned bytes outside the declared classes (`HostBudget::
/// serve_pinned`; the 27B plane's is 4 GiB of park staging, which this plane never allocates):
/// both ranks' warm / verify stage (64 expert slots = 168 MiB each), the xptr and page-table
/// mirrors (< 16 MiB), rounded up to 0.5 GiB.
pub const FN_PLANE_PINNED: u64 = 1 << 29;
/// FN51: the longest context any boot may plan (the checkpoint's `max_position_embeddings`).
pub const ROPE_POS_MAX: u32 = 262_144;
/// FN51: the rope table's positions at a boot of pools context `ctx` (`max(ctx, ROPE_POS)`).
#[must_use]
pub fn rope_pos(ctx: u32) -> u32 {
    ctx.max(ROPE_POS)
}
/// FN51: bytes of [`rope_table`]`(n)` (128 a position: 64 bf16 cos/sin pairs), for the ledger.
#[must_use]
pub fn rope_bytes(n: u32) -> u64 {
    u64::from(n) * 128
}
/// One xptr table (gate_up or down): 3 x (256 + 1) u64 at EP=2.
pub(super) const XN: usize = 3 * 257;
pub(super) const XBYTES: u64 = (XN * 8) as u64;

/// FN110b: the MTP drafter's routed + shared experts by default -- the 4-bit SLAB (`slab4`: the
/// pack's additional `rank<r>/mtp_x4.*` MXFP4 records on the trunk's W4A8 train; the user's word
/// 2026-10-12, re-gated on the day's defaults, LEDGER FN110b); `--fn-mtp-experts fp8` (FN50's
/// first-party fp8) is the control, `bf16` FN50's own control. FN134: `--fn-experts tp2` runs
/// the slab too (the MTP layer at `Dims::mtp_form`, EP's tables; `auto` resolves to it).
pub const DEFAULT_FN_MTP_EXPERTS: &str = "slab4";
/// The MTP experts `--fn-mtp-experts auto` resolves to under `--fn-experts tp2`: FN134 (hold h2,
/// 2026-10-15) = the slab, as under EP (FN80d's `fp8` here was the confound: +0.46 ms drafter
/// round, ~2x the slab's VRAM); `--fn-mtp-experts fp8` = FN80d's arm ([`FnFlags::mtp_experts`]).
pub const DEFAULT_FN_MTP_EXPERTS_TP2: &str = "slab4";

/// Flash-Next flags of `saber engine` (inert on a 27B pack, which refuses them).
#[derive(clap::Args, Debug, Clone, PartialEq)]
pub struct FnFlags {
    /// Flash-Next: the result directory (JSON + optional captures).
    #[arg(long)]
    pub fn_out: Option<PathBuf>,
    /// Flash-Next: the FN3 prompt ids to run (default: every prompt of `--prompt-ids`' run).
    #[arg(long, value_delimiter = ',')]
    pub fn_prompts: Vec<String>,
    /// Flash-Next: synthetic prompts of these lengths (cyclic concatenation of the prompt set's ids).
    #[arg(long, value_delimiter = ',')]
    pub fn_long: Vec<u32>,
    /// Flash-Next: write `flashnext-capture/1` per request (per layer resid / attn_out / mlp_out /
    /// router ids + logits, ple_out, final_hidden, logits_last). OFF by default. (`--fn-capture`
    /// until FN14, which gave the word to the decode graphs.)
    #[arg(long)]
    pub fn_dump: bool,
    /// Offline TP=2 workers: save the full next-token logits after each prompt as
    /// little-endian FP32 (`<prompt-id>.f32`) for comparison with a reference model.
    #[arg(long)]
    pub fn_prefill_logits_dir: Option<PathBuf>,
    /// Flash-Next (FN14): the pairing CONTROL -- run every decode step EAGERLY (FN13's form). The
    /// default replays one hipGraph per (decode bucket 1/2/4/8, PLE parity), captured on first use.
    #[arg(long)]
    pub no_fn_capture: bool,
    /// Flash-Next (FN14, diagnostic): re-emit every replayed step on the host (no launch) and
    /// refuse a step whose launch records differ from the captured graph's (the by-construction
    /// law, checked).
    #[arg(long)]
    pub fn_capture_verify: bool,
    /// Flash-Next (FN14): how a captured decode step is cut -- `layer` (the default: one graph per
    /// decoder layer + one for the final mixer and head; each layer's graph waits only for its own
    /// layer's expert copies, as the eager step does) or `step` (ONE graph, every copy waited for
    /// before it: hold 1's form, measured slower at b2+ because the LRU's promotions stop
    /// overlapping the step); FN64: `lazy` = one graph per RUN of layers with no copy to wait
    /// for, cut only at a layer whose gate waits (quantized down to `--fn-graph-lazy-q`) + the
    /// PLE layer + the head. Absent = [`DEFAULT_FN_GRAPH_SPLIT`].
    #[arg(long)]
    pub fn_graph_split: Option<String>,
    /// FN64 (`--fn-graph-split lazy`): a gated layer cuts the step at the largest multiple of this
    /// many layers at or below it (or at the PLE layer), so a rank captures at most one graph per
    /// (bucket form, run of such blocks). Absent = [`DEFAULT_FN_GRAPH_LAZY_Q`].
    #[arg(long)]
    pub fn_graph_lazy_q: Option<u32>,
    /// FN64 (`--fn-graph-split lazy`): the captured runs a rank keeps (least recently replayed
    /// evicted past it). Absent = [`DEFAULT_FN_GRAPH_LAZY_CACHE`].
    #[arg(long)]
    pub fn_graph_lazy_cache: Option<usize>,
    /// FN64b (`--fn-graph-split lazy`): `on` = before the first step, capture every run the step
    /// forms of the boot's FIRST context band can use (short + long rows, every bucket and parity:
    /// no capture pass on a served step below that band); `all` = every band's forms up to the
    /// cache; `off` = FN64's capture on first use. A form past the cache is left to the on-demand
    /// path either way. Absent = [`DEFAULT_FN_GRAPH_LAZY_PRE`].
    #[arg(long)]
    pub fn_graph_lazy_pre: Option<String>,
    /// FN64c (`--fn-graph-split lazy`): write the runs this boot's SERVED steps used, per step
    /// form with their use counts, to `DIR/lazy_runs.rank<r>.json` (every 4,096 replays and at
    /// exit) -- the set `--fn-graph-lazy-pre recorded` pre-captures.
    #[arg(long)]
    pub fn_graph_lazy_record: Option<PathBuf>,
    /// FN64c (`--fn-graph-lazy-pre recorded`): the directory holding `lazy_runs.rank<r>.json`.
    /// Absent = the pack's `profile/` ([`LAZY_SET_IN_PACK`]).
    #[arg(long)]
    pub fn_graph_lazy_set: Option<PathBuf>,
    /// FN64d (`--fn-graph-split lazy`): the device bytes the lazy runs' graph execs may hold a
    /// rank (MiB; every hipGraphExec holds one 2 MiB kernarg pool on ROCm 10.1, FN64c), BILLED in
    /// the ledger: the plan budget grows by the 1,408 MiB today's unbilled ~704 lazy execs take and the
    /// class `fn.graph.exec` takes this many, so the difference becomes expert slots. The boot
    /// pre-captures the recorded set's MOST USED runs (across forms) within it; the rest are
    /// captured on demand and the least recently replayed evicted past it. Absent =
    /// [`DEFAULT_FN_GRAPH_EXEC_BILL`]; `0` = FN64c's unbilled cache of 1,024 runs.
    #[arg(long)]
    pub fn_graph_exec_budget_mib: Option<u64>,
    /// FN64e (an ARM, `--fn-graph-split lazy`): CHARGE the graph execs in the ledger
    /// ([`ExecBill::Charge`]: the class `fn.graph.exec` = [`FN64E_CHARGE_MIB`], the plan budget
    /// grows by the same bytes so the slots do not move; the boot pre-captures the recorded set's
    /// most used 704 runs, the LRU holds <= 736).
    #[arg(long)]
    pub fn_graph_exec_charge: bool,
    /// FN98 (`--fn-graph-split lazy`): `one` = ONE graph exec per decode / verify step form (two
    /// with the PLE overlap's layer-1 copy point), the tier's layer gates captured INSIDE it as
    /// wait nodes on the mover's landed-mark words (each set before its launch to the block's
    /// latest mark), the graph execs BILLED in the ledger (`fn.graph.exec`) and the bytes today's
    /// unbilled lazy execs took made STATIC expert slots ([`ExecBill::One`]); the lazy run table
    /// is not consulted. `lazy` = FN64's run split (the control). Absent =
    /// [`DEFAULT_FN_GRAPH_FORM`].
    #[arg(long)]
    pub fn_graph_form: Option<String>,
    /// Flash-Next (FN14, the seed-law poison arm): fill the WHOLE arena (scratch, seeded regions,
    /// carries) with this byte before every decode step, outside the graph.
    #[arg(long)]
    pub fn_poison_step: Option<u8>,
    /// Flash-Next (FN14): the decode schedule -- `groups` (FN13's: arrivals in groups of
    /// `--fn-conc`, decoded together) or `grow` (the bucket-transition gate: 8 sequences join one
    /// by one and leave at staggered lengths, the batch walks 1->2->4->8->4->2->1).
    #[arg(long, default_value = "groups")]
    pub fn_sched: String,
    /// Flash-Next (FN17, the pad-law gate's CONTROL): the pre-FN17 pad law -- a padded decode row
    /// `i` reads and writes pool slot `i` (which may hold a prompt still prefilling between its
    /// chunks). The default puts every pad row at the pools' SPARE slot.
    #[arg(long)]
    pub fn_pad_control: bool,
    /// Flash-Next (FN17, an ARM, measured not winning): the scheduler's packed mixed form -- a decode
    /// step carries the queued prompts' chunks, run as one engine job in the serial form (bit-exact
    /// per row). The default is FN14's schedule: every prompt chunk a scheduler step of its own.
    #[arg(long, conflicts_with = "no_fn_mixed")]
    pub fn_mixed: bool,
    /// Flash-Next (FN17): FN14's unmixed schedule (the default; this flag names it).
    #[arg(long)]
    pub no_fn_mixed: bool,
    /// Flash-Next (FN18): the EXACT runtime ledger -- the per-rank runtime classes are what the boot
    /// allocates (`fn.pools` = the pools byte for byte at `--fn-max-ctx` x 9 slots + the drafter's,
    /// the prefill arena, `fn.buffers` = the fixed buffers + the reduce scratch), not FN7's planned
    /// maximum (131,072 KV tokens, 8 x 5 GDN slots, a 0.5 GiB decode arena the engine never
    /// allocates: decode runs in the prefill arena). The freed bytes become expert slots.
    #[arg(long, conflicts_with = "no_fn_ledger_exact")]
    pub fn_ledger_exact: bool,
    /// Flash-Next (FN18): FN7's planned-maximum runtime ledger (the control).
    #[arg(long)]
    pub no_fn_ledger_exact: bool,
    /// Flash-Next (FN18, the placement byte law): at the end of the run (the harness's last prompt,
    /// the serve plane's stop) every VRAM slot and every RAM slot of each rank is read back and
    /// compared with its record's sha256 (`fnslots rank<r>: checked N, differ 0`).
    #[arg(long)]
    pub fn_verify_slots: bool,
    /// Flash-Next (FN18, a RECORDER, byte-neutral): append every engine step's routed expert ids
    /// (target layers + the MTP drafter's passes) and every PLE gather's row ids to this file
    /// (`fn18-route-trace/1`, rank 0; `saber flashnext-profile` reads it: the real-traffic profile,
    /// the host replay of the tier, the owner-permutation estimate, the PLE cache sizing).
    #[arg(long)]
    pub fn_route_trace: Option<PathBuf>,
    /// Flash-Next serve plane (FN75, a RECORDER, byte-neutral): write every served step's joined
    /// record (the scheduler's clock + shape and rank 0's job: pick-up, device spans by part, the
    /// target step's host phases, the reply, the post-reply drafter round) to this file, one JSON
    /// line a step (`fn75-step/1`); the aggregate is on `/metrics` (`saber_fn_step_*`) and the
    /// shutdown line `fnstep75` with or without it. `tools/capture/fn75/ledger.py` reads it.
    #[arg(long)]
    pub fn_step_trace: Option<PathBuf>,
    /// Flash-Next (FN65, the served RECORDER, byte-neutral): record the served engine's routes
    /// (target steps + PLE rows; not the MTP passes) into rotated `fn18-route-trace/1` files under
    /// this directory (`routes-<unix>-<pid>-<seq>.bin`), encoded and written by a writer thread
    /// off the rank thread (a full queue drops a record, never waits). `saber flashnext-profile
    /// rebuild --trace DIR` turns them into a warm-start profile (`--fn-profile user`).
    /// `tools/serve_flashnext.sh --record-routes` = `/mnt/nvme/saber-kv/routes/<date>/`.
    #[arg(long, conflicts_with = "fn_route_trace")]
    pub fn_route_record: Option<PathBuf>,
    /// FN65: target steps a recorded file holds before the recorder rotates.
    #[arg(long, default_value_t = DEFAULT_FN_ROUTE_RECORD_STEPS)]
    pub fn_route_record_steps: u64,
    /// FN65: the most GiB of its own files the recorder keeps (a ring: the oldest goes first).
    #[arg(long, default_value_t = DEFAULT_FN_ROUTE_RECORD_GIB)]
    pub fn_route_record_gib: f64,
    /// Flash-Next (FN14): serve the OpenAI routes on this port instead of running the prompt set.
    #[arg(long)]
    pub fn_serve: Option<u16>,
    /// Flash-Next serve: the chat template (default: the checkpoint's own `chat_template.jinja`).
    #[arg(long)]
    pub fn_chat_template: Option<PathBuf>,
    /// Flash-Next serve (FN66): WARM PROMPTS -- after the server listens, every conversation head
    /// in FILE (`saber_serve::warm`: a JSON array / JSON lines / plain text = one system prompt)
    /// is rendered through the chat path and prefilled once into the PROMPT CACHE, so a session's
    /// first turn that begins with it reloads it. Turns the cache on at the tied settings when
    /// no cache flag says otherwise (`--fn-kv-pcache-gib 16`, min tokens 1,000,000 = no
    /// conversation cut) and serves without admission by reservation (the cache's law), so an
    /// absent `--fn-max-ctx` boots at 10,240; the boot log says so.
    #[arg(long)]
    pub warm_prompts: Option<PathBuf>,
    /// Flash-Next serve: the served model name.
    #[arg(long, default_value = "Qwen3.8-Flash-Next")]
    pub fn_served_name: String,
    /// Flash-Next serve: the default of `enable_thinking` for a request that does not say
    /// (`on` = the template's own default, reasoning effort xhigh; `off` renders
    /// `enable_thinking=false` and parses no reasoning).
    #[arg(long, default_value = "on")]
    pub fn_thinking: String,
    /// Flash-Next serve: the longest context a sequence may reach (the pools' size: 8 slots of it;
    /// the ledger's KV class follows, so a context past 16,384 -- 131,072 tokens a rank, the
    /// budget's default KV -- is paid for in VRAM expert slots). FN45b4: absent =
    /// `kvwire::DEFAULT_FN_KV_MAX_CTX` (10,240: the pinned-RAM ceiling) under the tiered KV,
    /// 8,192 on the flat pools.
    #[arg(long)]
    pub fn_max_ctx: Option<u32>,
    /// Flash-Next (FN45b2): the TIERED KV -- `on`: the QSA layers' K/V pages live in a VRAM hot
    /// pool (`--fn-kv-hot-gib`, default today's flat pools at 8,192 = 0.916 GiB) <- a pinned RAM
    /// slab holding every block of 8 x `--fn-max-ctx` (`--fn-kv-ram-gib`), read through
    /// per-layer page tables (no kernel change; `flashnext::kvwire`); `off` = the flat pools (the
    /// CONTROL). Worker form at TP=2 with `--fn-qsa sparse` only.
    #[arg(long)]
    pub fn_kv_tier: Option<OnOff>,
    /// Flash-Next (FN45b2): the tier's VRAM hot pool in GiB (absent: today's flat pools' bytes).
    #[arg(long)]
    pub fn_kv_hot_gib: Option<f64>,
    /// Flash-Next (FN45b2): the tier's pinned RAM slab in GiB a rank (absent: every block of 8 x
    /// the max context; a smaller slab is refused at boot -- the admission law).
    #[arg(long)]
    pub fn_kv_ram_gib: Option<f64>,
    /// Flash-Next (FN45b2, the page byte law): at the end of a harness run every (slot, QSA layer)
    /// prints the sha256 of its K/V pages in logical order (`fnkvpages` lines), tier or flat.
    #[arg(long)]
    pub fn_kv_pages: bool,
    /// Flash-Next (FN68): the QSA layers' main K/V pages -- `bf16` (THE DEFAULT, upstream QSA's
    /// only mode, the control) or `fp8` (the ARM: e4m3 bytes at one per-tensor scale 1.0, the
    /// 27B's append law; the attention / gather bodies' fp8kv instantiations upconvert exactly to
    /// f16 in LDS). Half the bytes a token: the pinned slab, the hot pool and every page halve at
    /// the same tokens. The indexer's compressed keys and raw-key ring stay bf16.
    /// FN68b: the default is [`DEFAULT_FN_KV_DTYPE`]; fp8 pages take the scale law
    /// `--fn-kv-scales` (the pack's calibrated `kv_scales.json` by default) and the prompt cache.
    #[arg(long, default_value = DEFAULT_FN_KV_DTYPE, value_parser = ["bf16", "fp8"])]
    pub fn_kv_dtype: String,
    /// Flash-Next (FN68d): the MTP layer's (48's) K/V page dtype under `--fn-kv-dtype fp8` --
    /// `bf16` (its own one-layer bf16 tier beside the 12 target layers' fp8 tier: the drafter's
    /// ~750-nat attention logits pick the wrong key with K at e4m3, LEDGER FN68c; ~+0.13 GiB
    /// pinned a rank at 262k) or `fp8` (FN68b's one dtype, the control). Absent =
    /// [`DEFAULT_FN_KV_MTP_DTYPE`] in the worker form; the one-process form keeps one dtype.
    /// `bf16` pages need no flag (every layer is bf16).
    #[arg(long, value_parser = ["bf16", "fp8"])]
    pub fn_kv_mtp_dtype: Option<String>,
    /// Flash-Next (FN68b): the fp8 K/V pages' scale law -- `pack` (THE DEFAULT under fp8: the
    /// pack's calibrated `kv_scales.json`, a `fn68b-kv-scales/1` document, refused when absent),
    /// `unit` (FN68's one scale 1.0: the control), or a scales FILE (gate arms). bf16 refuses it.
    #[arg(long)]
    pub fn_kv_scales: Option<String>,
    /// Flash-Next (FN45b3, the eviction cell's NEGATIVE CONTROL): the tier skips every D2H flush
    /// (an evicted page then reads stale RAM: the bytes must move). Gate arm only.
    #[arg(long)]
    pub fn_kv_drop_flush: bool,
    /// Flash-Next (FN45b4): the tier's prefill gather OFF (FN45b3's in-place RAM reads of a
    /// prompt's evicted pages; the gather's control arm). The protection order stays.
    #[arg(long)]
    pub no_fn_kv_gather: bool,
    /// Flash-Next (FN45b4, gate arms only): boot although the tier's pinned RAM (`kv.ram`, 8 x
    /// the max context a rank) does not fit the host's pinnable bytes beside the expert RAM cache
    /// -- the slab then eats into the 16 GiB free floor. Without it such a boot is refused.
    #[arg(long)]
    pub fn_kv_ram_overcommit: bool,
    /// Flash-Next (FN45c2): the tier's DRIVE slab in GiB a rank (absent = two tiers; bounded by
    /// `kvwire::SSD_GIB_MAX`). With it the tier runs FN45c's resident law: the pinned RAM slab
    /// (default today's served slab, 8 x 10,240 tokens) holds the ACTIVE sequences' blocks,
    /// admitted by REQUEST LENGTH (the scheduler's KV budget = the slab's tokens: a request that
    /// does not fit waits), every complete block written through to the drive -- so
    /// `--fn-max-ctx` may pass the RAM ceiling (one 64k request beside short ones).
    #[arg(long)]
    pub fn_kv_ssd_gib: Option<f64>,
    /// Flash-Next (FN51): the CONTROL of admission by reservation -- a `--fn-max-ctx` past the
    /// pinned-RAM ceiling then pins 8 x it (FN45b4's law: refused past the host's pinnable bytes).
    #[arg(long)]
    pub no_fn_kv_admit: bool,
    /// Flash-Next (FN45c2): the PROMPT / CONVERSATION CACHE in GiB a rank on the drive (absent =
    /// off): every non-last prompt chunk's state (its KV pages + the sequence's GDN / conv /
    /// indexer ring + store / PLE history + the drafter's stash) is stored under the chained key
    /// of the chunks so far, and a later prompt whose chunks repeat them RELOADS instead of
    /// prefilling (the last chunk always runs). Byte law: reload == prefill.
    #[arg(long)]
    pub fn_kv_pcache_gib: Option<f64>,
    /// Flash-Next (FN45c3): the prompt cache OFF on a `--fn-serve` boot (the control once
    /// `PCACHE_SERVED_DEFAULT` is true; today a no-op).
    #[arg(long, conflicts_with = "fn_kv_pcache_gib")]
    pub no_fn_kv_pcache: bool,
    /// Flash-Next (FN45c3, the async store's CONTROL): the cache's drive write runs inside the
    /// step (FN45c2's form).
    #[arg(long)]
    pub fn_kv_pcache_sync: bool,
    /// Flash-Next (FN45c3, the byte law's REFERENCE arm): a cache hit is COMPUTED at the very
    /// same cut instead of reloaded (the reload == prefill law at the conversation cut).
    #[arg(long)]
    pub fn_kv_pcache_verify: bool,
    /// Flash-Next (FN45c3): the shortest prompt prefix the conversation cut stores (a chat
    /// prompt's last chunk is cut before its last `<|im_start|>` and the part before it stored).
    /// Absent: [`DEFAULT_FN_KV_PCACHE_MIN_TOKENS`], or [`WARM_FN_KV_PCACHE_MIN_TOKENS`] under
    /// `--warm-prompts` (FN66).
    #[arg(long)]
    pub fn_kv_pcache_min_tokens: Option<u32>,
    /// Flash-Next (FN45c4): the conversation cut is rounded down to a multiple of this many
    /// tokens (1 = FN45c3's cut, the default; 64 = FN13's aligned boundary, measured worse in FN45c4).
    #[arg(long, default_value_t = DEFAULT_FN_KV_PCACHE_ALIGN)]
    pub fn_kv_pcache_align: u32,
    /// Flash-Next (FN113): the TIERED PREFIX TREE (`off` = the default and the control; `on`):
    /// the prompt is cut by the tree's law (prefix-stable: after every earlier generation prompt,
    /// after a long system message, at the chunk ends), every piece but the final one is reloaded
    /// from the lowest tier holding its node (pinned RAM, then the drive's `kv.pcache`) or prefilled
    /// and stored, and a prompt ending in a generation prompt stores its END node in `Cmd::First`
    /// (FLASHNEXT-PLAN §16). Needs the tiered KV and `--fn-serve`; turns the drive cache on at
    /// `--fn-kv-pcache-gib` (16 a rank) and replaces FN45c3's conversation cut.
    #[arg(long, value_parser = ["off", "on"])]
    pub fn_prefix_tree: Option<String>,
    /// Flash-Next (FN113): the tree's pinned RAM tier in GiB a rank (2 MiB THP pages; counted in
    /// `fnhost` beside `kv.ram`).
    #[arg(long, default_value_t = DEFAULT_FN_PREFIX_TREE_RAM_GIB)]
    pub fn_prefix_tree_ram_gib: f64,
    /// Flash-Next (FN113b): where the END node is stored -- `live` (the default: its copies off
    /// the decode step on the mover's stream, the drafter rows as the prompt's own pass leaves
    /// them), `side` (the same copies after the cut's m - 1 row drafter pass), `step` (FN113: inside `Cmd::First`, the control),
    /// `off` (a DIAGNOSTIC: no END node is ever written; the turn class then never hits).
    #[arg(long, default_value = DEFAULT_FN_PREFIX_TREE_END, value_parser = ["live", "side", "step", "off"])]
    pub fn_prefix_tree_end: String,
    /// Flash-Next (FN113b, a DIAGNOSTIC): `off` = the tree's evicted nodes are dropped, never
    /// demoted to the drive (the demotion's attribution arm); `on` = the lazy demotion.
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    pub fn_prefix_tree_demote: String,
    /// Flash-Next (FN113): the shortest piece a turn cut leaves behind it (tokens since the last cut).
    #[arg(long, default_value_t = saber_mem::prefix_tree::DEFAULT_MIN_GAP)]
    pub fn_prefix_tree_min_gap: u32,
    /// Flash-Next (FN113): the shortest system message that is cut as a shared node (0 = never).
    #[arg(long, default_value_t = saber_mem::prefix_tree::DEFAULT_SYS_MIN)]
    pub fn_prefix_tree_sys_min: u32,
    /// Flash-Next (LR-I, an ARM; `today` = the default and the control): the tree's cut rule --
    /// `asst` (the turn cut right after `<|im_start|> assistant \n`, before `<think>`: a client
    /// that sends its history without the reasoning resumes 2 tokens below its previous prompt's
    /// end instead of at the cut before it), `grid<N>` (a checkpoint cut every N tokens: the next
    /// turn resumes within N of where it diverges), `asst+grid<N>`. Any rule but `today` moves a
    /// computed piece's boundary: a cold prompt's bytes differ from today's (G4, the user's call);
    /// within a rule a hit is byte-identical to its cold prefill.
    #[arg(long, default_value = "today", value_parser = parse_tree_cuts)]
    pub fn_prefix_tree_cuts: String,
    /// Flash-Next (FN45c2): the drive directory of the KV slab and the prompt cache.
    #[arg(long, default_value = "/mnt/nvme/saber-kv")]
    pub fn_kv_dir: PathBuf,
    /// Flash-Next (FN45b3, harness only): raise the pools' context to at least this (rounded to
    /// 256) -- a harness run at a served boot's `--fn-max-ctx` (the decode split law's input).
    #[arg(long)]
    pub fn_pools_ctx: Option<u32>,
    /// Flash-Next (FN96): the per-sequence state pools' LAYOUT -- `layer` (one buffer a layer
    /// holding every sequence slot: FN13's pools, the control) or `seq` (sequence-major: one
    /// region a sequence slot holding its conv + GDN live + GDN snapshot rows of every GDN layer,
    /// so an idle sequence is one contiguous run; the kernels keep their per-layer base + slot
    /// index, the tables carry the region-strided index). [`DEFAULT_FN_POOLS_LAYOUT`] when absent.
    #[arg(long, default_value = DEFAULT_FN_POOLS_LAYOUT, value_parser = ["layer", "seq"])]
    pub fn_pools_layout: String,
    /// Flash-Next (FN100): the row-scaled buffers' LAYOUT -- `one` (ONE allocation, each buffer's
    /// head of `--xtier-lend-arena-top-rows` rows at its base, the tails after it: what
    /// `--xtier-lend-buffers-top` lends) or `separate` (one allocation a buffer: the pre-FN100 boot,
    /// the control; refuses the lend). [`DEFAULT_FN_BUFFERS_LAYOUT`] when absent.
    #[arg(long, default_value = DEFAULT_FN_BUFFERS_LAYOUT, value_parser = ["one", "separate"])]
    pub fn_buffers_layout: String,
    /// Flash-Next (FN99): where the rank's half of the token embedding (`embed_tokens`, 0.592 GiB
    /// a rank, read only by the drafter's device chain: `fn_draft_pack` copies one row a drafted
    /// token) lives -- `vram` (FN22's resident half, the control) or `ram` (one pinned host
    /// allocation read IN PLACE over UVA by the same kernel: the VRAM class leaves the ledger and
    /// becomes static expert slots, the pinned bytes are billed in `fnhost`). Absent =
    /// [`DEFAULT_FN_EMBED_PLACEMENT`] (`ram` since FN107) with FN99's STAND-DOWN: a plain boot whose
    /// RAM L2 would stop covering the complement falls back to `vram` (a `fnembed99` line says
    /// so); only an EXPLICIT `ram` is refused there.
    #[arg(long, value_parser = ["vram", "ram"])]
    pub fn_embed_placement: Option<String>,
    /// Flash-Next (FN19): the QSA attention above 2,048 tokens -- `sparse` (THE MODEL'S LAW, the
    /// default: the indexer's top-512 groups per row, `saber_kernels::fn19`) or `dense` (the
    /// control arm: every prior token, FN12-FN17's attention).
    #[arg(long, default_value = "sparse")]
    pub fn_qsa: saber_kernels::fn19::QsaMode,
    /// Flash-Next: cut each prompt into chunks of this many rows (0 = one chunk, the default).
    #[arg(long, default_value_t = 0)]
    pub fn_chunk: u32,
    /// Flash-Next (FN38): the widest prefill chunk of this boot -- 2,048 (FN19's cut), 4,096 or
    /// 8,192 rows (FN44: the default). A wider chunk pays the link's RAM-expert set once per chunk
    /// instead of once per 2,048 rows; the prefill arena, the staging buffers and the MTP hidden
    /// buffer are sized to it through the ledger (the expert slots pay for it).
    #[arg(long, value_parser = parse_max_chunk)]
    pub fn_max_chunk: Option<u32>,
    /// Flash-Next (FN38): when the serve plane takes the wide rung (only with `--fn-max-chunk` >
    /// 2,048): `idle` = only while no sequence is decoding (a chunk beside decode keeps 2,048 rows,
    /// so a decode step never waits behind a wide chunk), `always` = every chunk (FN44: the
    /// default).
    #[arg(long)]
    pub fn_wide_when: Option<WideWhen>,
    /// Flash-Next (LR-B, an ARM; absent / 0 = today's prefill-first schedule, the control): decode
    /// steps the serve plane runs BETWEEN two prefill chunks while some sequence is ready to
    /// decode, so a decoder waits one chunk instead of the whole prompt. The chunk cut and every
    /// step's form are unchanged (ordinary unmixed steps, disjoint sequences).
    #[arg(long)]
    pub fn_prefill_yield: Option<u32>,
    /// Flash-Next (LR-G, an ARM; absent / 0 = today's whole reservation, the control): LAZY KV
    /// GROWTH under admission by reservation. A plain request reserves `prompt + min(max_tokens,
    /// N)` (+ the speculative overshoot) instead of `prompt + max_tokens`; a decoding sequence
    /// takes its next block(s) in the scheduler's arena before the step that writes them (the
    /// engine's tier grows to match on demand); a growth the arena refuses preempts the youngest
    /// lazy sequence for recompute (back to the queue's front, its stream continues). A lazy
    /// admission leaves N tokens free for the running sequences' growth. The elastic want counts
    /// written blocks. KV bytes of a sequence that is never preempted are unchanged.
    #[arg(long)]
    pub fn_kv_lazy: Option<u32>,
    /// Flash-Next (LR-K, an ARM; absent = [`super::kvwire::admit_slab_tokens`] = one max-context
    /// request + one 10,240-token request + the extras, 273,104 tokens a rank at 262,144, the
    /// control): THE ADMISSION SLAB in TOKENS a rank under admission by reservation (every QSA
    /// layer's blocks at the boot's page dtypes, the MTP twin included; the scheduler's budget =
    /// this less the 9 slots' extras). Never under the default (refused by name); past the host's
    /// pinnable bytes the boot is refused by the host law, which names the largest that pins.
    /// `--fn-kv-lazy` keeps working with it (the margin and growth run against the bigger
    /// arena). Exclusive with `--fn-kv-ram-gib` (the byte form).
    #[arg(long, conflicts_with = "fn_kv_ram_gib")]
    pub fn_kv_admit_slab_tokens: Option<u64>,
    /// Flash-Next (FN42b): the wide prefill arena as a TRANSIENT slot borrow -- `on` (FN44: the
    /// default): the wide arena's scratch past the narrow (2,048-row) rung is lent to the expert
    /// tier as INCLUSIVE slots while no wide chunk runs and reclaimed (a placement flip, no copy)
    /// before one; `off` = FN38's permanent cut (the control). Only with `--fn-max-chunk` > 2,048.
    #[arg(long, conflicts_with = "no_fn_arena_borrow")]
    pub fn_arena_borrow: Option<OnOff>,
    /// Flash-Next (FN44): the borrow's control by name (= `--fn-arena-borrow off`).
    #[arg(long)]
    pub no_fn_arena_borrow: bool,
    /// Flash-Next (FN44b, the FN44 arm): the borrow's FN42b layout -- ONE layout, zero / ones /
    /// carry permanent at the wide rung, only the wide scratch past the narrow scratch lent. The
    /// default (FN44b) is the TWO-LAYOUT borrow: narrow passes run in the c2048 boot's arena
    /// carved at the base, every byte past it is lent, a wide pass reclaims it all.
    #[arg(long)]
    pub fn_borrow_cut: bool,
    /// Flash-Next (FN44b, the FN44 arm): leave lent slots EMPTY until decode misses fill them.
    /// The default (FN44b) FILLS every empty lent slot (the borrow's and the ring's) at a prefill
    /// pass's start, behind a fence, one H2D a slot, each layer's copies gated as the inserts'.
    #[arg(long)]
    pub no_fn_lend_fill: bool,
    /// Flash-Next: request concurrency levels to time (arrivals in groups of N).
    #[arg(long, value_delimiter = ',', default_value = "1")]
    pub fn_conc: Vec<u32>,
    /// Flash-Next: passes over the request set per concurrency level (the first is IC-cold).
    #[arg(long, default_value_t = 1)]
    pub fn_repeat: u32,
    /// Flash-Next diagnostic: fill the scratch arena with this byte before every layer (poison arm).
    #[arg(long)]
    pub fn_poison: Option<u8>,
    /// Flash-Next gate control: `swap-arg` (the GDN gated norm's z stride off by one head).
    #[arg(long)]
    pub fn_control: Option<String>,
    /// Flash-Next (FN90a; absent = [`DEFAULT_FN_PLE_DRIVE_KEEPALIVE_MS`] = `0` (off) since 2026-10-14, the
    /// user's power directive; `40` = FN90a's keep-awake arm): keep the PLE drive out of its autonomous power states -- one thread of rank 0's process reads one 4 KiB `O_DIRECT` block of the first PLE
    /// file every N ms (below the kernel's 100 ms APST idle timer), so a decode step's late PLE
    /// rows never pay a power-state exit. Nothing the engine computes reads it (byte-neutral);
    /// the root form is `nvme_core.default_ps_max_latency_us=0` or the device's
    /// `power/pm_qos_latency_tolerance_us`.
    #[arg(long)]
    pub fn_ple_drive_keepalive_ms: Option<u64>,
    /// Flash-Next (FN121, bit-exact: placement + timing only; absent =
    /// [`super::ple121::DEFAULT_FN_PLE_PREFETCH`] = `off` = today, the control): the cold PLE
    /// drive-read tail's fixes, `all` or a comma set of `cand` (THE PLE PREFETCH LAW: after every
    /// drafter round the next verify's candidate tokens' n-gram rows -- the committed token's and
    /// every draft's -- are submitted to the drive in one batch without waiting and land in the
    /// row cache before the step's PLE gate; a row not resident there is the counted stall
    /// `fn121 ple-miss`, never a wrong byte), `keep` (the keep-awake at `--fn-ple-keep-ms`, one
    /// reader per block device the PLE / embedding files sit on), `embed` (the host embedding
    /// gather from a host-RAM copy of the table, read once at boot). End lines `fnple121 rank<r>:`.
    #[arg(long)]
    pub fn_ple_prefetch: Option<String>,
    /// Flash-Next (FN121, `--fn-ple-prefetch keep`): the keep-awake's period in ms.
    #[arg(long, default_value_t = super::ple121::DEFAULT_FN_PLE_KEEP_MS)]
    pub fn_ple_keep_ms: u64,
    /// Flash-Next DIAGNOSTIC (FN121, byte-neutral): every decode step's and drafter round's PLE /
    /// embedding / gate record and every drive read batch's per-read timings, one JSON line each,
    /// in `DIR/rank<r>.jsonl` (both ranks).
    #[arg(long)]
    pub fn_ple_trace: Option<PathBuf>,
    /// Flash-Next (FN106d; absent = [`DEFAULT_FN_PINNED_PAGES`] = `huge` since FN106d's hold, `shmem` =
    /// the control): the page
    /// form of EVERY pinned host allocation of >= 2 MiB both ranks make (the RAM L2 slab, the
    /// embedding half, the KV RAM slab, the stage / staging buffers, the valve's segments) --
    /// `shmem` = `hipHostMalloc` (hsakmt's shmem, 4 KiB pages on this box), `huge` = a private
    /// anonymous 2 MiB-aligned mapping + `MADV_HUGEPAGE` + `hipHostRegister` (no root), falling
    /// back BY NAME to `hipHostMalloc` where THP or the register refuses. Byte-neutral (a page size
    /// never changes a byte). Boot / end lines `fnpinned106d rank<r>:`.
    #[arg(long)]
    pub fn_pinned_pages: Option<String>,
    /// Flash-Next: the per-rank device budget the ledger plans against (MiB).
    #[arg(long, default_value_t = 30_500)]
    pub fn_budget_mib: u64,
    /// Flash-Next: the forced-miss control: evict every expert placement before the run.
    #[arg(long)]
    pub fn_evict_all: bool,
    /// Flash-Next: per-op events (the timing arm; not the timed pass).
    #[arg(long)]
    pub fn_op_timing: bool,
    /// FN54b (a DIAGNOSTIC, off): time the compute stream's idle at every per-layer decode gate
    /// that waits on in-flight tier copies (two timing events around the wait; `fndecode` line +
    /// the steps' `gate_ms`).
    #[arg(long)]
    pub fn_gate_time: bool,
    /// Flash-Next: the HOST pre-flight (no card): prompts, the PLE hash vs FN3's ids, the embedding
    /// rows vs FN3's, the dry-run launch law, the ledger at this budget.
    #[arg(long)]
    pub fn_preflight: bool,
    /// Flash-Next: store the GDN recurrent state in f32 (the chunk-law control; f16 is the default).
    #[arg(long)]
    pub fn_gdn_state_f32: bool,
    /// Flash-Next gate A: layers to run (48 = the whole forward + the head).
    #[arg(long, default_value_t = 48)]
    pub fn_depth: u32,
    /// Flash-Next: the warm-start route profile when `--xtier-profile` is not given: `pack` (the
    /// DEFAULT, FN18c: the PACK's `profile/route_profile.json` = [`DEFAULT_PROFILE_IN_PACK`],
    /// FN18's real-traffic profile), `user` (FN65: the pack's ADDED member
    /// `profile/route_profile.user.json` = [`USER_PROFILE_IN_PACK`], built from the user's own
    /// recorded traffic by `saber flashnext-profile rebuild`), or a path. FN7's FN3-prompt
    /// profile (`~/saber-cache/ws-fn7/profile/route_profile.json`) is the control.
    #[arg(long)]
    pub fn_profile: Option<PathBuf>,
    /// Flash-Next (FN13): decode this many greedy tokens per request at SPEC 0 (0 = prefill only;
    /// the first token is the prefill's pick, so N tokens = N - 1 decode steps).
    #[arg(long, default_value_t = 0)]
    pub fn_decode: u32,
    /// Flash-Next (FN13): two-turn conversations (prefill, decode, a user turn as a continuation
    /// prefill, decode) beside the same conversation as one prompt.
    #[arg(long, default_value_t = 1)]
    pub fn_turns: u32,
    /// Flash-Next (FN13, the chunk-law arm): every all-reduce on the EXACT kernel.
    #[arg(long)]
    pub fn_ar_exact: bool,
    /// Flash-Next (FN13, the chunk-law diagnostic): dump this layer's GDN intermediates per chunk.
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub fn_tap_layer: i32,
    /// Flash-Next (FN13): TP=2 as FN12's ONE process with two HipThreads (the byte control of the
    /// worker-process form, which is the default).
    #[arg(long)]
    pub fn_one_process: bool,
    /// Internal (the worker form): this process's rank.
    #[arg(long, default_value_t = 0, hide = true)]
    pub fn_rank: u32,
    /// Internal (the worker form): rank 0's exchange port.
    #[arg(long, default_value_t = 0, hide = true)]
    pub fn_peer_port: u16,
    /// FN73 (DEBUG, the teardown law's injection): `R:N` -- rank R's serve loop fails after N
    /// decode steps (rank 0: N served steps; rank 1: N decode commands applied).
    #[arg(long = "fail-rank-after-steps", hide = true, value_name = "R:N")]
    pub fail_rank_after_steps: Option<String>,
    /// Flash-Next (FN15b / FN16 / FN16f): MTP through the target's VERIFY step. `auto` = WIDTH BY
    /// BATCH (FN16: the measured table `verify::AUTO_TABLE`, bucket -> width; `--fn-mtp-table`
    /// overrides it) -- THE SERVED DEFAULT since FN16f (the user's word, 2026-09-29); `0` = SPEC 0
    /// (the served control arm, and the prompt-set harness's default: its byte gates are SPEC-0
    /// gates); `k` (1..4) = the drafter's chain of k passes after every step and the verify at
    /// width k + 1 at every batch. The walk commits the matched prefix + one (greedy: the argmax;
    /// sampled, in the serve plane: F40's drawn rows). `--fn-mtp-shadow` = FN15's shadow form (the
    /// chain after every SPEC-0 step, drafts recorded, the target untouched). Absent, the law is
    /// [`FnFlags::default_mtp`]; the resolved value is `fn_mtp`.
    #[arg(long = "fn-mtp", value_name = "auto|0|k")]
    pub fn_mtp_arg: Option<FnMtp>,
    /// The resolved `--fn-mtp` ([`FnFlags::resolved`]; not a flag).
    #[arg(skip)]
    pub fn_mtp: FnMtp,
    /// Flash-Next (FN16): the `--fn-mtp auto` table, `bucket:width,...` (width = rows = k + 1),
    /// e.g. `1:4,2:4,4:3,8:1`. Default: `verify::AUTO_TABLE` (the measured law).
    #[arg(long)]
    pub fn_mtp_table: Option<String>,
    /// Flash-Next (FN16, the control of a bit-exact drafter lever): the drafter's decode-form passes
    /// EAGER (the default captures each (rows, width, input) pass on first use and replays it).
    #[arg(long)]
    pub no_fn_mtp_capture: bool,
    /// Flash-Next (FN16, the control of a bit-exact drafter lever): the drafts from the 64-entry
    /// seam's top-1 (the default reads each row's argmax: 8 bytes a row across the pair).
    #[arg(long)]
    pub no_fn_mtp_argmax: bool,
    /// Flash-Next (FN22, the control of a bit-exact drafter lever): the drafter's chain through
    /// the HOST between passes (FN16's form: each pass's logits half to the host, the argmax, the
    /// pair's 8-byte exchange, the embedding row from the checkpoint file, the upload). The default
    /// is the DEVICE-SIDE chain: the argmax, the cross-rank exchange (the certified exact reduce)
    /// and the next token's embedding row from the rank-resident half, on the card; one D2H of the
    /// chain's ids at its end.
    #[arg(long)]
    pub no_fn_mtp_device_chain: bool,
    /// Flash-Next (FN22, the gate arm): after every device-chain pass, read the pass's logits and
    /// the embedding rows it wrote back and compare them with the HOST chain's law on the same
    /// logits (ids per row, sha256 of the embedding bytes); records under `--fn-out`
    /// (`chaincheck_rank<r>.jsonl`), the summary line `fnchaincheck`. Syncs every pass.
    #[arg(long)]
    pub fn_mtp_chain_check: bool,
    /// Flash-Next (FN154, a census arm; off = the default path, untouched): every MTP draft's
    /// confidence -- the int2 tail's exact candidate scores, both ranks' 2 x K merged: the draft's
    /// candidate-softmax prob, the top-1 / top-2 logit margin, the candidates' entropy -- with the
    /// verify step's outcome (accepted drafts a sequence), its width, rows and wall, as one JSON
    /// line a decode step (rank 0) at this path. Syncs every drafter pass, runs the passes eager and
    /// the FN119 round un-ahead (like the check arm): its walls are not the served walls.
    #[arg(long)]
    pub fn_mtp_conf_log: Option<PathBuf>,
    /// Flash-Next (FN33): the drafter's head on the device chain's passes -- `bf16` (FN22's pass:
    /// the shared bf16 head over the whole vocabulary half, the control) or `int2` (the 27B's int2
    /// scorer picks the top `--fn-draft-head-k` candidates of the rank's half, the target head's
    /// own GEMM re-scores them exactly, FN22's pick over the candidates). The drafts are the only
    /// thing it can move (the verify step decides every token; gate D's byte law holds).
    #[arg(long, default_value_t = DEFAULT_FN_DRAFT_HEAD)]
    pub fn_draft_head: saber_kernels::fn33::DraftHead,
    /// Flash-Next (FN33): the int2 head's candidates per rank (1..=80; 1 = no rerank, the
    /// ceiling's control).
    #[arg(long, default_value_t = saber_kernels::fn33::DEFAULT_K)]
    pub fn_draft_head_k: u32,
    /// Flash-Next (FN22, the tie law's gate): at boot, run the device chain's tail on constructed
    /// logits (ties inside a rank, across the pair, a flat row, -0 vs +0, -inf rows, a pad row) and
    /// compare with the host law, then with the tie-high CONTROL (which must differ on the ties).
    #[arg(long)]
    pub fn_mtp_chain_selftest: bool,
    /// Flash-Next (FN22, an arm): the device chain's passes 2..k as ONE captured graph per (bucket,
    /// live rows, passes, sparse flags) instead of one replayed graph per pass.
    #[arg(long)]
    pub fn_mtp_chain_graph: bool,
    /// Flash-Next (FN15, the control form): the MTP chain as a SHADOW of the SPEC-0 decode.
    #[arg(long)]
    pub fn_mtp_shadow: bool,
    /// Flash-Next (FN15b): the verify width per step from the 27B's acceptance-EMA cap
    /// (`ceil(ema) + 2`, at least 2, at most k + 1; the batch takes its widest sequence's).
    #[arg(long)]
    pub fn_mtp_dynamic: bool,
    /// Flash-Next (FN92): the b1 verify DEPTH law under `--fn-mtp auto` -- `table` (the DEFAULT:
    /// the table's b1 width, `verify::AUTO_TABLE`) or `adaptive` (an ARM: at bucket 1 ONLY the
    /// depth maximises the expected committed tokens over the measured iteration cost, per
    /// `flashnext::depth`; ceiling width 5 = the verify cap; b >= 2 keeps the table). Bit-exact by
    /// FN24's width-1 class: the depth moves how many tokens a step commits, never which.
    #[arg(long, default_value = DEFAULT_FN_MTP_DEPTH)]
    pub fn_mtp_depth: String,
    /// Flash-Next (FN15b, diagnostic): with `--fn-mtp-ref`, dump the MTP pass's per-op taps (block
    /// input, projection, q / k after norm + rope, K/V pages, core, gated core, o_proj partial).
    #[arg(long)]
    pub fn_tap_mtp: bool,
    /// Flash-Next (FN15b, gate D's control): the verify step's rollback dropped -- the next step
    /// starts from the LAST verify row's GDN snapshot / conv window and the PLE history keeps the
    /// rejected rows, as if every draft had been accepted (the ids must diverge).
    #[arg(long)]
    pub fn_mtp_no_rollback: bool,
    /// Flash-Next (FN24): the kernel CLASS of an MTP verify step's rows -- `width1` (THE DEFAULT:
    /// every M-keyed cell at SPEC 0's rows for the same batch + the f16 GDN carry across the
    /// window, so a verify row's bytes == SPEC 0's and MTP-k's greedy ids ARE SPEC 0's; the
    /// all-reduce band is kept) or `wide` (FN15b's routing, each cell at b x w rows and the fp32
    /// carry: the control).
    #[arg(long, default_value = "width1")]
    pub fn_verify_class: super::exec::VerifyClass,
    /// Flash-Next (FN78, the decode-form reduce band; bit-exact for the target): `exact` (THE
    /// DEFAULT) -- every TP = 2 reduce of a DECODE form (a target decode / verify step, a drafter
    /// pass) of <= 64 rows runs on the EXACT kernel whatever its payload, so a b8 verify at width 4
    /// or more (32+ rows, 128+ KiB) reduces its rows as SPEC 0's b8 step does and MTP depth 3 / 4
    /// at b8 keeps the width-1 byte law; `band` = the payload band (exact below 128 KiB, wht6
    /// above: the pre-FN78 law, the control). Every pre-FN78 served cell (<= 24 decode rows) is
    /// exact under both.
    #[arg(long, default_value = DEFAULT_FN_DECODE_AR, value_parser = ["exact", "band"])]
    pub fn_decode_ar: String,
    /// Flash-Next (FN24, the byte law's record): sha256 of every picked logits row (both vocabulary
    /// halves) in the harness results (`row_sha` per sequence per generated token) and the serve
    /// plane's nothing (harness only). Costs one hash of the row the pick already reads.
    #[arg(long)]
    pub fn_row_sha: bool,
    /// Flash-Next (FN24, the control of FN21's glue switch): the pre-switch glue symbols
    /// (`qk_norm_rope_wg8_h256_rot64_mi1_red5`, `fn_idx_q_compress_h128`); byte-identical by FN21.
    #[arg(long)]
    pub fn_glue_control: bool,
    /// Flash-Next (FN30): the target's MoE train -- `fused` (THE DEFAULT since FN30d: at a decode /
    /// verify step of <= 64 rows FN31's front half + FN26's back half, 2 launches a block at T <= 8,
    /// router + LDS route + back half, 3, above; prefill chunks and the drafter keep the seven) or
    /// `legacy` (the CONTROL: FN4's seven launches at every form). Byte-identical (LEDGER FN30,
    /// FN30d); [`DEFAULT_FN_MOE_TRAIN`].
    #[arg(long, default_value = DEFAULT_FN_MOE_TRAIN)]
    pub fn_moe_train: super::exec::MoeTrain,
    /// Flash-Next (FN84): the fused back half's work ORDER -- `id` (FN26's symbol, the default),
    /// `first` (the RAM-tier tiles' gate_up and down workgroups take the first tickets), `capped`
    /// (one RAM workgroup per k resident ones, k by the rules' `[[moe_miss]]` row), `cappre`
    /// (`capped` + a RAM down tile streams its second W slab before it waits). Bit-exact by
    /// construction (LEDGER FN84); the `fnmissorder` line is the which-path counter.
    #[arg(long, default_value = DEFAULT_FN_MOE_MISS_ORDER, value_parser = ["id", "first", "capped", "cappre"])]
    pub fn_moe_miss_order: String,
    /// Flash-Next (FN85): where the TP = 2 exact reduce of a block output runs at a target decode /
    /// verify form -- `off` (its own launch, `saber_ar_oneshot_2rank_exact_bf16`: THE DEFAULT and the
    /// control) or `norm` (inside the consumer's prologue: the decode-band combine down's `_rp`
    /// twin pushes this rank's partial, waits on the peer's flag words and forms the block output
    /// with the one-shot's own vadd; the reduce's launch, graph node and edge go -- 94 of a verify
    /// step's 96; the PLE layer's and the final mixer's keep theirs). Bit-exact by construction
    /// (LEDGER FN85); the `fnarpro` line is the which-path counter.
    #[arg(long, default_value = DEFAULT_FN_AR_PROLOGUE, value_parser = ["off", "norm"])]
    pub fn_ar_prologue: String,
    /// Flash-Next (FN114b): the protocol of the TP = 2 exact reduce (every standalone
    /// `saber_ar_oneshot_2rank_exact_bf16` site) -- `oneshot` (the reference's push / drain / release
    /// flag / acquire spin: the control since FN114d) or `ll` (`fn_ar_ll_exact_bf16`, the flag
    /// carried in the data words: each 16-byte word stored inverted, an all-zero dword = not yet
    /// landed, relaxed system-scope polls of the data themselves, the reader zeroes what it
    /// consumed; no drain, no flag store, no acquire on the common path). Bit-exact by
    /// construction and by its two-rank oracle (LEDGER FN114b); the `fnar114b` line is the
    /// which-path counter. The prologue (FN85) and producer-rr (FN37) channels keep their
    /// protocol. FN114d adds `gp`: every standalone exact AND wht6 reduce runs its global-parity
    /// twin (`saber_ar_oneshot_2rank_{exact,wht6}_bf16_gp`: the shipped bodies byte for byte, the
    /// workspace slot chosen by the CALL's parity -- counters that all advance once a call
    /// whatever the grid -- instead of each block's own counter, which aliases a slot the peer is
    /// still reading when the grid changes between back-to-back calls; LEDGER FN114b / FN114d):
    /// THE DEFAULT since FN114d.
    /// `host` selects an exact system-scope reduce over registered host pages on two
    /// workers without GPU peer access. SPEC=0 is the default; an explicit fixed MTP
    /// depth uses the host draft chain without peer producer fusions or issue-ahead.
    #[arg(long, default_value = DEFAULT_FN_AR, value_parser = ["oneshot", "ll", "gp", "host"])]
    pub fn_ar: String,
    /// Flash-Next (FN137 Phase B): the TP = 2 MoE-out all-reduce of a fused decode / verify back
    /// half (`--fn-experts tp2`) -- `off` (its own launch after the back half: THE DEFAULT and the
    /// control) or `on` (FUSED into the back half's combine: `fn137_moe_back_h2_far_*` in
    /// `moe_grouped_far` pushes each 128-column combine slice to the peer's private window, releases
    /// the slice's flag, spins on the peer's and writes the slice of the SUM with the exact
    /// one-shot's vadd; the slot by the channel's CALL parity, FN114d's law). Bit-exact by
    /// construction and by the two-card oracle (LEDGER FN137); -48 launches a step. Stands down
    /// (the standalone reduce, counted) where the back half is not FN80b-1's id-order half shape
    /// or the reduce is not the exact decode-form one; the `fnar137` line is the which-path counter.
    #[arg(long, default_value = DEFAULT_FN_MOE_AR_PUSH, value_parser = ["off", "on"])]
    pub fn_moe_ar_push: String,
    /// Flash-Next (FN164): the TP = 2 MIXER-OUT all-reduce (the attention o_proj / GDN out_proj
    /// partial, 48 a decode / verify step) -- `off` (its own launch after the W8 GEMV: the
    /// control), `on` (FUSED into the GEMV's epilogue at every target decode / verify
    /// form: `fn164_far_<parent>` in `flashnext_t8far` pushes each workgroup's 64 columns to the
    /// peer's private window, releases the workgroup's flag, spins on the peer's and writes the
    /// SUM with the exact one-shot's vadd; the slot by the workgroup's CALL parity, FN114d's law;
    /// THE DEFAULT since FN164's hold h1) or `b1` (the one-sequence forms only). Bit-exact by construction and by the two-card oracle
    /// (LEDGER FN164); -48 launches a fused step. Stands down (the standalone reduce, counted)
    /// where the cell has no fused twin, under FN85's prologue / FN37's producer-rr, or where the
    /// reduce is not the exact decode-form one; the `fnar164` line is the which-path counter.
    #[arg(long, default_value = DEFAULT_FN_MIXER_AR_PUSH, value_parser = ["off", "on", "b1"])]
    pub fn_mixer_ar_push: String,
    /// Flash-Next (FN116): the drafter's FIRST round after a prompt -- `pass` (the prompt's m-row
    /// prefill-form MTP pass + the first draft read on the host + the chain: THE DEFAULT) or `kv`
    /// (rows 0..m-2 run only the MTP layer's K-side prefix -- the K / V / indexer state a later
    /// pass reads -- and the last row is a steady drafter round: the decode-form pass with the
    /// device tail, then the chain). The target's bytes are unchanged by construction (the
    /// width-1 verify class, FN24); `kv` moves drafter bytes only (the last row's class, its head).
    /// Single-stream first rounds only (a ready set of one, the chain run); the `fnfirst116` end
    /// lines are the which-path counters. FN119e: `kvnext` = `kv` with the round's ids left
    /// pending (`--fn-device-step next`'s commit law): the first decode is built and queued under
    /// the first round (its row 0 from the sampled token written to the walk's result block).
    /// FN119e: `chunk` = `kvnext` + the K-side front staged under the last prompt chunk (its
    /// inputs uploaded async behind the chunk) and issued right after the chunk's pick, before the
    /// host samples the first token (the chunk's last logits row backed up for the dense fallback).
    /// FN119f: `dev` = `chunk` + the first token walked ON THE CARD behind the chunk (round 2's
    /// walk at width 1, the host's draw checked against it) and the first round queued behind the
    /// front -- the END node's placement + drafter part on the stream, its stash read back async,
    /// the chunk's tier end deferred onto its queued route log.
    /// FN119g: `under` = `dev` + the K-side front issued on a second stream as soon as the chunk's
    /// final mixer wrote the stream rows -- under the chunk's head, its pick reads and the walk --
    /// the round joining it (the same kernels, the same bytes; only the issue point moves).
    #[arg(long, default_value = DEFAULT_FN_FIRST_STEP, value_parser = ["pass", "kv", "kvnext", "chunk", "dev", "under"])]
    pub fn_first_step: String,
    /// Flash-Next (FN86b): the launch-fusion families whose ADDED fused symbol runs at their sites
    /// -- `all` (every built family: THE DEFAULT since FN86b's hold), `none` (every shipped chain:
    /// the control) or a list (`f1,f4`). Built: f1 (the GDN gated norm / the QSA attention gate + the W8 GEMM's
    /// activation quant in one launch) and f4 (a merged projection's bf16 suffix GEMM as extra
    /// workgroups of the W8 prefix's quant launch); each family's fused output is byte-equal to
    /// its chain (card-1 oracle, LEDGER FN86b); a family not built is refused by name. The
    /// `fnfuse86` line is the which-path counter. FN86c adds f5 / f2; FN86d adds f10 (the drafter
    /// pass's int2-tail glue under the W8 head, 8 launches -> 4 a pass; `f1,f2,f4,f5` = FN86c's
    /// set, FN86d's control arm). ATTN-OPT-3 adds f6 (a target QSA layer's norm/rope + indexer +
    /// KV/ring append, 3 -> 1, at <= 5 rows a sequence) and f7 (the split-KV combine + f1's gate +
    /// q8, 2 -> 1; needs f1), both in `all` since the f6/f7 flip (target decode / verify only;
    /// prefill and the drafter keep the chains); `f1,f2,f4,f5,f10` = the flip's control arm.
    #[arg(long, default_value = DEFAULT_FN_FUSE)]
    pub fn_fuse: String,
    /// Flash-Next (FN111): the roofline census's re-tiled cells whose ADDED symbol runs at their
    /// sites -- `all` (the tiles that passed the card-1 bar: t1 + t3; THE DEFAULT since FN111's
    /// hold), `default` (every shipped cell: the control) or a list (`t1,t3` / the fn111_ symbol names). t1 = the W8
    /// out / o projection at <= 16 rows on 64-column tiles, t2 = family 2's merged projection on
    /// 64-column tiles (missed the bar), t3 = the MoE route at 17..64 tokens at 16 waves; each
    /// byte-equal to the shipped cell (LEDGER FN111). The `fntile111` line is the which-path count.
    #[arg(long, default_value = DEFAULT_FN_TILES)]
    pub fn_tiles: String,
    /// Flash-Next (FN128): the QSA selection launches of a decode / verify step (CTAs of <= 8
    /// rows) whose ADDED `flashnext_qsa128` symbol runs -- `all` (top-k + union: THE DEFAULT since
    /// FN128's holds h1 + h4, 2026-10-14), `default` (every parent: the control) or a list
    /// (`topk`, `union`, `score`); each byte-equal to its `flashnext_qsa` parent (LEDGER FN128).
    /// The `fnsel128` line is the which-path count.
    #[arg(long, default_value = DEFAULT_FN_SEL128)]
    pub fn_sel128: String,
    /// Flash-Next (FN131): the fp8-mixer HC DOWN at the decode forms (cfgs `m1w3t1d` / `m1w7t1`)
    /// -- `none` / `default` (the shipped `flashnext_hc8` downs: the control) or `split` / `all` (THE DEFAULT)
    /// (the ADDED `flashnext_hc131` downs re-tiled by phase: tile waves stream the weight ring
    /// while phase waves run the norm; byte-equal to the parent, LEDGER FN131). The `fnhcd131`
    /// line is the which-path count.
    #[arg(long, default_value = DEFAULT_FN_HC_DOWN)]
    pub fn_hc_down: String,
    /// Flash-Next (FN141): the drained-ring twins whose ADDED symbols run -- `none` (THE DEFAULT:
    /// every shipped symbol, the control) or `all` / a list of `up` (the fp8-mixer decode up s1d:
    /// loads at entry + one-round q8 tail + select-free RNE, `flashnext_hc141`), `t8` (the out / o
    /// W8 GEMM as a 4-deep register ring at <= 32 rows, `flashnext_t8141`), `router` (the b >= 2
    /// router GEMM as a 16-step ring); each byte-equal to its parent (LEDGER FN141). The `fnk141`
    /// line is the which-path count.
    #[arg(long, default_value = DEFAULT_FN_K141)]
    pub fn_k141: String,
    /// Flash-Next (FN163): the MoE route twins whose ADDED symbols run -- `none` (THE DEFAULT: every
    /// shipped symbol, the control) or `all` / a list of `b8` (the three-launch train's route at
    /// 9..64 rows on `fn163_route_nt1024`: the shipped selection at 32 waves + one packed scan) and
    /// `b1` (the split-4 fused front on `fn163_front_e512_nw8_s4`: a register-rank top-10); each
    /// byte-equal to its parent (`moe_route163`, LEDGER FN163). The `fnroute163` line is the
    /// which-path count.
    #[arg(long, default_value = DEFAULT_FN_ROUTE163)]
    pub fn_route163: String,
    /// Flash-Next (FN122b / FN122c): the PREFILL tiles whose ADDED symbols run at their sites --
    /// `hc,dq` / `all` (THE DEFAULT since FN122c: at >= 1,024 prefill rows the fp8 mix's matrices
    /// dequantized once a mix and run on the bf16 bodies -- the down on `flashnext_hc8p`'s
    /// `m8w21t1k64qb`, the up on the shipped `p4s`; `hc` = FN122b's fp8 `m8w21t1k64q` down, which
    /// a producer-rr mix keeps), `hc` (FN122b's form) or `default` (every shipped prefill cell: the
    /// control). Every form byte-equal (LEDGER FN122b / FN122c). The `fnpfx122` line is the
    /// which-path count.
    #[arg(long, default_value = DEFAULT_FN_PREFILL_TILES)]
    pub fn_prefill_tiles: String,
    /// Flash-Next (FN120b): THE GLUE FOLD of a captured decode / verify run -- `fold` (THE
    /// DEFAULT: every seeded buffer of the run from one region outside the arena at a distinct
    /// address per layer, seeded at boot and restored by two fills at the run's end, + the route
    /// ids at the route-id log's own offsets: one copy a run; FN120e: the region sized at boot,
    /// a served run past it falls back to the per-layer fills), `seeds` (the seeds only) or `off`
    /// (the control: the per-layer seed fills + the per-layer route-id-log copy). Bytes identical
    /// by construction (only the seeded buffers' addresses move). The `fnglue120` line names the
    /// form.
    #[arg(long, default_value = DEFAULT_FN_GLUE)]
    pub fn_glue: String,
    /// Flash-Next (FN120e, a GATE CONTROL, never a serving mode): force the glue region's ZERO part
    /// to this many MiB instead of the boot census's derived size (`6` = FN120b's region: the
    /// served fallback's card gate replays the 10-14 crash form through it). Absent = the census.
    #[arg(long, hide = true)]
    pub fn_glue_region_mib: Option<u64>,
    /// Flash-Next (FN120c, the megakernel program's Phase B): THE DATAFLOW FORM of a captured
    /// target verify -- `off` (THE DEFAULT, the control: the launch train) or `gdn` (every b1
    /// verify GDN layer's tail -- merged projection, fused update + norm + quant, out projection
    /// -- as ONE persistent dataflow launch `fn120_df_gdn_b1` over a host-built table; the bodies
    /// are the shipped sources included verbatim, so the bytes are the train's by construction).
    /// The `fndf120` lines name the form and count the sites.
    #[arg(long, default_value = DEFAULT_FN_DATAFLOW)]
    pub fn_dataflow: String,
    /// Flash-Next (FN50): the MTP drafter's ROUTED experts -- `fp8` (THE DEFAULT since FN50's
    /// hold 2: acceptance 2.963 vs 2.956, target bytes 1,600/1,600, step faster) or `bf16` (the
    /// CONTROL: Quark's bf16 on FN6's bf16 band). fp8 = Qwen's first-party e4m3 block-128 experts from
    /// the pack's additional `rank<r>/mtp_fp8.*` file on FN6's W8A8 body, the bf16 shared expert
    /// on its bf16 band; half the resident bytes, the freed VRAM becomes expert slots). Only the
    /// drafts move: the target's verify keeps its bytes.
    /// FN110 / FN110b: `slab4` (THE DEFAULT since FN110b, the user's word 2026-10-12; numerics: the
    /// drafts move, the target's bytes do not; `fp8` = the control) = the routed + shared MTP experts as MXFP4
    /// records in the expert tier's native form from the pack's additional `rank<r>/mtp_x4.*`
    /// (`tools/pack/build_mtp_x4.py`), one fixed VRAM slab (0.62 GiB a rank vs fp8's 1.17: the
    /// freed bytes become expert slots), on the trunk's seven-launch W4A8 train. EP only.
    /// FN80d: `auto` (the flag's default) = slab4 under `--fn-experts ep`, and under `tp2` since FN134
    /// ([`FnFlags::mtp_experts`]: read the form through it, never this field).
    #[arg(long, default_value = "auto", value_parser = ["auto", "bf16", "fp8", "slab4"])]
    pub fn_mtp_experts: String,
    /// Flash-Next (FN138): the split of the MTP layer's slab4 experts -- `ep` (THE DEFAULT, the
    /// control: FN134's `Dims::mtp_form`, rank r's whole records of experts [256r, +256) + the
    /// shared expert on rank 0) or `tp2` (the ARM, numerics: the drafts move): every rank holds
    /// ITS HALF of all 512 MTP experts + the shared expert (FN80a's law, sliced at load from both
    /// ranks' `mtp_x4` files) and the MTP MoE runs at the target's tp2 dims (TpHalf cells, all
    /// 512 experts local). Needs `--fn-experts tp2` and the slab4 form.
    #[arg(long, default_value = "ep", value_parser = ["ep", "tp2"])]
    pub fn_mtp_split: String,
    /// Flash-Next (FN80b-2): the TARGET's routed experts -- `ep` (THE DEFAULT and the rollback:
    /// rank r owns experts [256r, +256) whole) or `tp2` (the ARM: every rank holds half r of every
    /// expert's intermediate dimension from the pack's `rank<r>/experts.tp2.xtier` (FN80a), both
    /// ranks route all 512 experts as local, the shared expert split, ONE mirrored placement; the
    /// tier's slot is one half record, read from the header's split marker; the MTP experts stay
    /// EP). A pack without the tp2 member refuses `tp2` by name; a 27B pack refuses the flag;
    /// The decode / verify forms run FN80b-1's TpHalf cells (DBK-64 down, `moe_back_h2_*`).
    #[arg(long, default_value = super::xtp2::DEFAULT_FN_EXPERTS, value_parser = ["ep", "tp2"])]
    pub fn_experts: String,
    /// Flash-Next (FN150): the expert tier's RECORD FORMAT -- `mxfp4` (THE DEFAULT: the Quark
    /// packs' v1 MXFP4_E8M0 records, every boot byte for byte today's) or `nvfp4` (the FP4-Local
    /// v2 NVFP4_E4M3G16 tier, 676-page records: FN148's form (b) `fn148_w4a8b_*` cells, the f32
    /// globals in the epilogue). The tier header's format tag must match (refused by name either
    /// way); `nvfp4` refuses what has no NVFP4 body yet: prefill tiles (chunks > 512 rows), the
    /// `_p1` / fused back half (the default `--fn-moe-train fused`: `legacy` only), MTP experts
    /// (`--fn-mtp 0` only), the TP
    /// split (`--fn-experts tp2`), the one-process form, and the shared expert's W8 lane.
    #[arg(long, default_value = super::xtp2::DEFAULT_FN_EXPERTS_FMT, value_parser = ["mxfp4", "nvfp4"])]
    pub fn_experts_fmt: String,
    /// Flash-Next (FN114c): where the TARGET's shared expert runs on the fused decode / verify
    /// train -- `rank0` (THE DEFAULT, today: the whole shared expert on rank 0 only) or `split`
    /// (the NUMERICS arm: each EP rank computes ITS half of the shared expert's intermediate
    /// dimension -- FN80a's slicer, the TP2 form's `shared_expert_half` -- inside the fused back
    /// half `moe_back_ss_*`, and the halves meet in the post-MoE all-reduce the layer already
    /// pays; prefill chunks and the MTP drafter keep rank 0's whole shared expert). EP only, the
    /// worker form, the fused train at the id miss order.
    #[arg(long, default_value = super::shx114::DEFAULT_FN_SHARED_EXPERT, value_parser = ["rank0", "split"])]
    pub fn_shared_expert: String,
    /// Flash-Next (FN61c / FN61d): the dense trunk's DECODE-form weights -- `bf16` (THE DEFAULT,
    /// the control) or `w8` (the ARM, the user's form of 2026-10-06: GDN in_proj qkvz / out_proj
    /// and QSA q|k|v / o_proj at fp8 W8A8 -- e4m3 128 x 128 block weights from a `-trunk8` pack
    /// variant's `rank<r>/trunk8.*` (`saber flashnext-pack --trunk w8`), per-128 e4m3 activations;
    /// the HEAD, HC, in_proj_ba, the indexer, router and norms stay bf16) or `w8-head` (FN61c's
    /// form: `w8` + the head at W8A8; host KL 0.0238 > the 0.0227 bar). Decode and verify steps
    /// only: prefill and the drafter's exact rerank keep the bf16 records, which stay resident (the
    /// W8 bytes are ADDED until the bf16 copy is dropped). FN61e: `w8` / `w8-head` now run PREFILL
    /// on the W8 records too and DROP the bf16 trunk records from VRAM (the ledger's weight classes
    /// lose them: the slots rise); `w8-decode` = FN61d's form (decode / verify only, the bf16
    /// records resident), kept as the prefill control. FN108: `w8-hc8` (arm A) = `w8` + the HC
    /// MIXING MATRICES at fp8 (the reference's hc8m: e4m3 + f32 row-group scales, weight-only, from the
    /// pack's `shared/hc8.*`, `saber flashnext-pack --hc8`; the bf16 mixers are not uploaded, the
    /// MTP layer's stay bf16) and `w8-hc8-head` (arm B) = `w8-head` + the same.
    #[arg(long, default_value = "bf16", value_parser = ["bf16", "w8", "w8-head", "w8-decode", "w8-hc8", "w8-hc8-head"])]
    pub fn_trunk: String,
    /// Flash-Next (FN49b, the CONTROL of the DPP route switch): the pre-FN49b route symbols -- FN4's
    /// `moe_route_topk10_sort_e512` in the seven-launch train, the `ds_bpermute` LDS route / front
    /// half (`moe_route_topk10_sort_e512_lds`, `moe_front_e512_nw8_s*`) in the fused one. The
    /// default routes FN49a's DPP symbols (`_lds_dpp` at T 1..16 in every train, `_dpp` fronts);
    /// tables byte-identical (FN31 + FN49a oracles).
    #[arg(long)]
    pub fn_route_control: bool,
    /// Flash-Next (FN30b, a DIAGNOSTIC; every byte unchanged): the probe code object
    /// (`tools/capture/fn30b/fn30b_probe.hip`) launched EAGERLY on each rank's stream at the target
    /// step's end, the drafter round's start and the chain's end: the shader clock + a device
    /// stamp and a 128 MiB VRAM read (the run JSON's `fn30b.probes`).
    #[arg(long)]
    pub fn_diag_probe: Option<PathBuf>,
    /// Flash-Next (FN30b, a DIAGNOSTIC; every byte unchanged): drain the stream and sleep this many
    /// us before every drafter round after a verify step (the device idle: a transient-state test).
    #[arg(long, default_value_t = 0)]
    pub fn_diag_gap_us: u64,
    /// Flash-Next (FN30b, a TIMING CONTROL -- bytes move): every RAM-tier xptr entry points at a
    /// VRAM slot's record (same sub-record offset); the tier's accounting is unchanged.
    #[arg(long)]
    pub fn_diag_alias_ram: bool,
    /// Flash-Next (FN47b, a DIAGNOSTIC; every forward byte unchanged): at every target decode /
    /// verify step of <= 64 rows each layer also runs (a) its router on its attention-side mix
    /// output and (b) the next layer's router on its MoE input, and the host scores both
    /// predictions' top-10/16/24 against the TRUE routes and the tier's placement (the misses a
    /// pre-gated prefetch would have to copy): `fnpregate rank<r>` lines at the end of the run.
    /// Needs the eager decode (`--no-fn-capture`).
    #[arg(long)]
    pub fn_pregate_capture: bool,
    /// Flash-Next (FN30c, the tier mover's HOLD law; bit-exact, a SCHEDULING change): a verify
    /// step's promotion / victim copies are queued on the host and issued only after the MTP
    /// drafter round that follows it (the copy stream fenced behind the round), never overlapping
    /// it; a held batch is issued at the latest before the next step's first layer gate.
    /// Default [`DEFAULT_FN_MOVER_HOLD`]; `--no-fn-mover-hold` = the control (FN8's issue-at-end).
    #[arg(long, conflicts_with = "no_fn_mover_hold")]
    pub fn_mover_hold: bool,
    /// Flash-Next (FN30c): the control -- copies issued at the verify step's end (FN8's law).
    #[arg(long)]
    pub no_fn_mover_hold: bool,
    /// Flash-Next (FN56, the host-exposed step; every device byte and launch order unchanged): the
    /// host levers on, a comma list of `pick` (a pick's per-row seam top-k / lse on parallel host
    /// threads), `ple` (a verify step's PLE rows in ONE gather: one parallel NVMe batch, not one
    /// per row), `defer` (a verify step's tier end -- the route log's read, the placement resolve,
    /// the copy issue -- runs while the drafter round that follows is on the card; the copy fence
    /// stays at the step's end), `begin` (with `defer`: the next step's expert pointer tables are
    /// built and uploaded there too, not at the step's begin), or `none` (the control: FN51g's
    /// host path). Absent =
    /// [`DEFAULT_FN_HOST_LEVERS`].
    #[arg(long)]
    pub fn_host_levers: Option<String>,
    /// Flash-Next (FN112; absent = [`DEFAULT_FN_OVERLAP`] = `on`, the default; `off` = the control): the
    /// overlap scheduler, `off` | `on`.
    /// `on` queues a decode / verify step's token-INDEPENDENT reads behind its graph BEFORE the
    /// host waits: the device seam pick, the rows the requests' sampling needs (chosen by the
    /// request parameters, never by a token), the tier end's route log -- each a D2H into pinned
    /// staging on the step's stream, one wait for all; the host then runs the tier end and the
    /// pick on bytes already landed. The same kernels on the same bytes in the same stream order
    /// (bit-exact by construction); end line `fnoverlap112 rank<r>:`.
    #[arg(long)]
    pub fn_overlap: Option<String>,
    /// Flash-Next (FN119; absent = `next` = THE DEFAULT since FN119d 2026-10-13; `ahead` = its
    /// control (FN119b's: the round behind the verify, the commit reads its ids), `lazy` = round 1's,
    /// `off` = the pre-FN119 host law): the device-resident step's forms -- `next` (`ahead` + the
    /// next verify built and issued under the round, the ids read before the drafts' PLE rows) -- `lazy` (a decode row's log-sum-exps only where the sampler reads them: the T = 1
    /// one for logprobs, `lse_t` filled late from the dense row only when the merged seam's tail
    /// can live; bit-exact by construction), `lazy-late` (every deferred row filled late: the
    /// fill's control), `check` (`lazy` + the device draw / walk kernel `fn119_draw_walk` beside
    /// the host walk every decode step, compared; fails loud). End line `fndstep119 rank0:`.
    #[arg(long)]
    pub fn_device_step: Option<String>,
    /// Flash-Next (LR-N; absent = `off` = THE DEFAULT and the control): COUPLED DRAFTING, an ARM.
    /// `on`: the MTP chain drafts each SAMPLED row with the target's own noise at the position the
    /// draft will be verified at (`flashnext_couple`: argmax over the drafter's top-`top_k`
    /// candidates of s / T - ln E, E the target sampler's key at that position), so a draft equals
    /// the target's draw whenever the drafter's law agrees with the target's. The walk is
    /// unchanged: every committed token is still the target's draw, so the output distribution,
    /// the seed law (same seed = the same tokens as non-speculative decoding) and every byte gate
    /// HOLD with the arm on; greedy rows keep FN22's argmax drafts bit for bit. Only the drafts
    /// (acceptance) move. An unseeded sampled request draws ONE seed at its first step (instead of
    /// a fresh key a step) so its drafts can know the noise -- the same law in distribution. Needs
    /// the device chain with the int2 draft head under FN86d's family 10 (the served boot); any
    /// other drafter refuses the boot by name. End line `fncouple rank<r>:`.
    #[arg(long)]
    pub spec_couple: Option<String>,
    /// Flash-Next (FN117b; absent = [`super::hand::DEFAULT_FN_HAND`] = `all`, THE DEFAULT; `none` =
    /// the pre-FN117b host hand, the control): the bit-exact host-hand slices, `none` or a comma
    /// list of `s1` (the drafter
    /// round's three input uploads as ONE async pinned batch, no drain), `s2` (the next step's
    /// expert-pointer tables built + queued under the drafter round, kept iff the tier's key is
    /// unchanged at the begin), `s6` (the pick exchange through a shared host line, the slot by
    /// the call's global parity, instead of the socket), `all` (= s1,s2,s6), and `check` (each
    /// slice's oracle, fail loud). The same kernels, inputs and bytes; end line `fnhand117b
    /// rank<r>:` (`super::hand`).
    #[arg(long)]
    pub fn_hand: Option<String>,
    /// Flash-Next (FN117b, a race-check DIAGNOSTIC): sleep a pseudo-random 0..N us (per rank and
    /// call) before every pick exchange.
    #[arg(long, default_value_t = 0, hide = true)]
    pub fn_hand_skew_us: u64,
    /// FN57: the host-wait arm (`saber engine --host-wait`, crossed in by value at dispatch;
    /// [`DEFAULT_FN_HOST_WAIT`] when absent).
    #[arg(skip = DEFAULT_FN_HOST_WAIT)]
    pub host_wait: saber_exec::HostWait,
    /// Flash-Next (FN57): where a step's pick runs -- `host` (the logits' D2H + the host seam, the
    /// control) or `device` (`fn_pick_top64_bf16` on the card: each row's seam top-64 in the seam's
    /// total order, D2H only those 512 B a row; a row that asks for its log-sum-exp, a temperature,
    /// probes or the row sha still reads its row). Absent = [`DEFAULT_FN_PICK`].
    #[arg(long, value_parser = ["host", "device"])]
    pub fn_pick: Option<String>,
    /// Flash-Next (FN55b): serve IMAGES -- the pack's `vision/` member (FN55: Flash-Next's own
    /// tower, merger out 2,560) resident on each rank (ledger class `fn.vision`, billed before
    /// the expert slots), the tower's rows spliced into the prompt's embedding rows before the
    /// hyper-connection streams, the 27B's M-RoPE positions per token (the QSA attention, the
    /// indexer's queries and its compressed keys through the ring's 3-D positions), the serve
    /// plane's `image_url` parts accepted. FN55c: `--fn-vision [on|lazy|off]` -- `on` (also the
    /// bare flag) = FN55b's resident tower; `lazy` = the tower's region is LENT to the expert
    /// tier until the first image request loads it, released after `--fn-vision-idle-s` with no
    /// image; `off` = a text-only boot (refuses an image by name). Absent = [`DEFAULT_FN_VISION`]
    /// where it can stand ([`FnFlags::vision_mode`]).
    #[arg(long, num_args = 0..=1, default_missing_value = "on", value_parser = ["on", "lazy", "off"])]
    pub fn_vision: Option<String>,
    /// Flash-Next (FN55c, `--fn-vision lazy`): seconds with no image request after which the
    /// resident tower is released and its region lent back to the expert tier.
    #[arg(long, default_value_t = DEFAULT_FN_VISION_IDLE_S)]
    pub fn_vision_idle_s: u64,
    /// Flash-Next (FN55b): the tower's arena in patches per request (`--fn-vision`); 16,384 =
    /// 4,096 image tokens.
    #[arg(long)]
    /// FN55c: absent = [`DEFAULT_FN_VISION_MAX_PATCHES`], except under `lazy` where that
    /// region does not pin beside the KV slab: then the largest of 12,288 / 8,192 / 4,096 whose
    /// region keeps every expert record in VRAM or RAM (`plan_boot`'s "cap the lend").
    pub fn_vision_max_patches: Option<u32>,
    /// Flash-Next (FN55b, a gate's tap): every tower run's output rows written to
    /// `DIR/rank<r>/enc<epoch>.bin` (`[tokens, 2560]` bf16).
    #[arg(long)]
    pub fn_vision_dump: Option<PathBuf>,
    /// Flash-Next (FN125): image prompts and the prefix tree / prompt cache -- `on` = an image
    /// prompt meets them like text: every piece key folds the content hash of each image run the
    /// piece overlaps (`saber_sched::prefix_key`'s law), and the tower run is DEFERRED to the
    /// first computed chunk over an image run (a prompt whose image runs all reload never runs
    /// it) -- THE DEFAULT since FN125 h1; `off` = FN55b's bypass (an image prompt computed, never
    /// stored: the control). Absent = [`DEFAULT_FN_IMAGE_CACHE`]. Inert on a boot with no cache or
    /// no tower.
    #[arg(long, value_parser = ["on", "off"])]
    pub fn_image_cache: Option<String>,
    /// Flash-Next (FN57): pin the rank's threads -- `auto` (rank r's main thread and HIP thread on
    /// two cores of CCD r, its pick threads on the CCD's other cores) or `off` (the control: the
    /// scheduler places every thread). Absent = [`DEFAULT_FN_PIN`].
    #[arg(long, value_parser = ["auto", "off"])]
    pub fn_pin: Option<String>,
    /// Flash-Next (FN58b, bit-exact): where a decode-form step's PLE rows are gathered -- `off`
    /// (the control: every row gathered and uploaded before layer 0) or `layer1` (the committed
    /// token's rows gathered under the drafter round that precedes the step, the drafts' rows
    /// gathered while layer 0 runs and copied on a side stream; the layer-1 graph -- the PLE
    /// layer, the rows' first reader -- waits on that copy). FN58c's ablation arms:
    /// `layer1-l0only` (no pre-hook: every row gathered under layer 0), `layer1-inline` (the copy
    /// in order on the step's stream between the layer-0 and layer-1 graphs, no side stream),
    /// `layer1-inline-l0only` (both). Absent = [`DEFAULT_FN_PLE_OVERLAP`].
    #[arg(long)]
    pub fn_ple_overlap: Option<String>,
    /// Flash-Next (FN53b, an ARM, placement only; serve plane, worker form): THE MEMORY-PLANE
    /// PRESSURE VALVE. `on` = the top `--fn-mem-valve-max-tail` slots of each rank's pinned RAM
    /// L2 become 256-record segments (one pinned allocation each); rank 0's controller
    /// (`saber_mem::mem_valve`: MemAvailable < 16 + 2 GiB for 2 s -> demote one segment's coldest
    /// records, layer-balanced, to the drive and FREE it; > 16 + 4 GiB for 30 s -> reopen and
    /// refill; <= 1 move/s) moves both ranks between steps. While a segment is closed the tier
    /// is incomplete: steps run the demand path (eager), a prefill pass reads its tail through
    /// a bounce one layer ahead (never into the L2). `off` = the DEFAULT (today's slab).
    #[arg(long, default_value = "off", value_parser = ["on", "off"])]
    pub fn_mem_valve: String,
    /// Flash-Next (FN72, bit-exact: the same copies, kernels and order, only the HARDWARE QUEUE
    /// a stream's packets land on changes): `split` = the expert mover's ordered copy stream on a
    /// hardware queue of its own (a full-CU-mask stream: ROCclr never pools it with another
    /// stream), so the drafter round's packets never sit behind the promotion copies' barriers;
    /// `split-both` = the rank's step stream on its own queue too; `shared` = today's pooled
    /// streams (the control). Default [`DEFAULT_FN_SDMA_LANES`].
    #[arg(long, default_value = DEFAULT_FN_SDMA_LANES, value_parser = ["shared", "split", "split-both"])]
    pub fn_sdma_lanes: String,
    /// FN53b: the most records a rank's valve may demote (a multiple of 256).
    #[arg(long, default_value_t = DEFAULT_FN_MEM_VALVE_MAX_TAIL)]
    pub fn_mem_valve_max_tail: u32,
    /// Flash-Next (FN53c): the valve's trigger law. `boot` (the DEFAULT) = the boot's free floor
    /// minus a margin (`saber_mem::mem_valve::resolve`: trigger `max(16, boot - 2)` GiB of
    /// effective free memory, reopen `max(trigger + 1.5, boot - 0.5)`; the engine's own RSS growth
    /// is not pressure), sampled when the serve loop starts; `fixed` = FN53b's constants (16 + 2 /
    /// 16 + 4 GiB of MemAvailable), the control.
    #[arg(long, default_value = DEFAULT_FN_MEM_VALVE_TRIGGER, value_parser = ["boot", "fixed"])]
    pub fn_mem_valve_trigger: String,
    /// Flash-Next (FN53c): steps under an OPEN valve keep the captured graphs (an ARM; FN53c h1:
    /// slower than the demand path -- the static open set misses every step). `on` = a decode / verify step of an incomplete tier replays its graphs with one
    /// `fn_xmiss` launch a layer between the route and the experts: a routed record that is not
    /// resident raises the layer's miss word in a coherent host mailbox, the HOST (polling while
    /// it waits for the step) reads the missing records drive -> the valve's bounce (the P2PDMA
    /// pool under `--xtier-nvme-p2p on`, else pinned RAM) and answers with their addresses; the
    /// kernel patches the layer's xptr entries and the experts run. Placement: the open set is
    /// static (no miss enters the L2). `off` (the DEFAULT) = FN53b's per-layer demand path (eager).
    /// Prefill passes keep FN53b's demand path + the layer-ahead tail reader either way.
    #[arg(long, default_value = DEFAULT_FN_VALVE_GRAPHS, value_parser = ["on", "off"])]
    pub fn_valve_graphs: String,
    /// Flash-Next (FN51f-b, placement only; the tiered KV): ELASTIC VRAM -- the KV hot pool is
    /// allocated at the boot window + `--fn-vram-elastic-kv-max-gib` (carved out of the expert
    /// static slots by the exact ledger) and its top, cut into segments of 4 expert slots
    /// (`saber_mem::elastic`), is LENT to the expert tier while no live request holds it and
    /// RECLAIMED at once when the live reservations need it (a long request at its admission,
    /// before its first prefill chunk). `off` = today's fixed split (the control). Absent =
    /// [`DEFAULT_FN_VRAM_ELASTIC`].
    #[arg(long)]
    pub fn_vram_elastic: Option<OnOff>,
    /// FN51f-b: the fewest static expert slots a rank may keep after the elastic carve (the boot
    /// refuses below it).
    #[arg(long, default_value_t = DEFAULT_FN_VRAM_ELASTIC_MIN_SLOTS)]
    pub fn_vram_elastic_min_slots: u32,
    /// FN51f-b: GiB a rank a long request may borrow past the boot window (the region's extra).
    #[arg(long, default_value_t = DEFAULT_FN_VRAM_ELASTIC_KV_MAX_GIB)]
    pub fn_vram_elastic_kv_max_gib: f64,
    /// FN51f-c (placement only): how a segment the KV hands back reaches the experts. `eager` =
    /// THE FILL ON LEND: in the same tick its slots take the hottest RAM records (LRFU now), one
    /// H2D a slot on the copy stream after the KV's last use of the pages (bounded: one tick's
    /// segment); `lazy` = FN51f-b's law, the slots arrive EMPTY and refill only through the
    /// insert cap (the control). Default [`DEFAULT_FN_VRAM_ELASTIC_REFILL`].
    #[arg(long, default_value = DEFAULT_FN_VRAM_ELASTIC_REFILL, value_parser = ["eager", "lazy"])]
    pub fn_vram_elastic_refill: String,
    /// FN130 (placement only): THE DEMAND LAW of the elastic KV region. `on` = a KV page-opening
    /// site the step law's tick does not cover (the prompt cache / prefix tree's reload of a
    /// stored piece) and every prefill chunk's exact opens reclaim the lent segments they need AT
    /// ONCE (the request waits inline for the flip, `fnelastic rank<r>: EARLY RECLAIM`, counted)
    /// -- a served boot never dies on `no vram slot for a block that must be resident`; `strict`
    /// = such a site REFUSES loudly (the gates; every non-served boot runs strict); `off` = the
    /// control (the pre-FN130 law: FN128 h1 s4_A's FATAL). Default [`DEFAULT_FN_VRAM_ELASTIC_DEMAND`].
    #[arg(long, default_value = DEFAULT_FN_VRAM_ELASTIC_DEMAND, value_parser = ["on", "strict", "off"])]
    pub fn_vram_elastic_demand: String,
    /// FN51f-b (the residency view): at the end of a run every rank writes
    /// `DIR/plane-rank<r>.json` -- per layer the resident expert ids, routed counts by path and
    /// the H2D bytes moved (the `fnplane` lines carry the same census).
    #[arg(long)]
    pub fn_plane_dump: Option<PathBuf>,
    /// Flash-Next (FN37, the producer-rr law; bit-exact): every TP = 2 EXACT reduce of a block
    /// output (below 128 KiB, <= 64 rows) is `fn_ar_exact_hcrr_bf16`, which also writes the next
    /// mix's combined stream x' and its norm factor rr; that mix's down runs its norm-free twin.
    /// Default [`DEFAULT_FN_HC_RR_PRODUCER`]; `--no-fn-hc-rr-producer` = the control (the down
    /// combines and norms, FN11's path).
    #[arg(long, conflicts_with = "no_fn_hc_rr_producer")]
    pub fn_hc_rr_producer: bool,
    /// Flash-Next (FN37): the control -- the down computes the combine and the norm (FN11).
    #[arg(long)]
    pub no_fn_hc_rr_producer: bool,
    /// Flash-Next (FN15, the MTP host-reference gate): run the MTP chain's first pass on FN3's
    /// `resid[47]` (this run dir) + FN3's first generated token instead of the engine's own stream,
    /// and dump the pass's outputs under `--fn-out`/mtpref.
    #[arg(long)]
    pub fn_mtp_ref: Option<PathBuf>,
    /// FN69 (serve plane, MTP on): PROMPT-LOOKUP speculation -- a per-sequence n-gram index over
    /// the prompt + the committed output; at each drafting point a match of at least
    /// `--spec-lookup-min-match` tokens REPLACES the MTP chain's drafts with what followed the
    /// match (`saber_engine::lookup`). The verify step is unchanged (same width), so greedy ids
    /// are the target's own. `off` = the control.
    #[arg(long, default_value = DEFAULT_SPEC_LOOKUP, value_parser = ["on", "off"])]
    pub spec_lookup: String,
    /// FN69: the n-gram lengths indexed (`3..4`: the longest match wins).
    #[arg(long, default_value = "3..4")]
    pub spec_lookup_ngram: String,
    /// FN69: the merge law's bar -- the match length (extended backwards) the lookup needs.
    #[arg(long, default_value_t = 3)]
    pub spec_lookup_min_match: u32,
    #[command(flatten)]
    pub xtier: XtierFlags,
}

/// FN69: prompt-lookup speculation's default on both engines (`off` until a green gate).
pub const DEFAULT_SPEC_LOOKUP: &str = "off";

impl FnFlags {
    /// FN69: the prompt-lookup law of `--spec-lookup` (`None` = off).
    ///
    /// # Errors
    /// A `--spec-lookup-ngram` / `--spec-lookup-min-match` the parser refuses.
    pub fn lookup(&self) -> anyhow::Result<Option<saber_engine::LookupConfig>> {
        if self.spec_lookup != "on" {
            return Ok(None);
        }
        saber_engine::LookupConfig::parse(&self.spec_lookup_ngram, self.spec_lookup_min_match)
            .map(Some)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

/// `--fn-mtp`: SPEC 0, a fixed chain k at every batch, or width by batch (FN16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FnMtp {
    /// SPEC 0 (the harness default; the served control arm).
    #[default]
    Off,
    /// MTP-k at every batch (1..4).
    K(u32),
    /// Width by batch (`verify::AUTO_TABLE` or `--fn-mtp-table`): the served default (FN16f).
    Auto,
}

impl std::str::FromStr for FnMtp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "auto" => Ok(FnMtp::Auto),
            "0" => Ok(FnMtp::Off),
            t => match t.parse::<u32>() {
                Ok(k) if (1..=4).contains(&k) => Ok(FnMtp::K(k)),
                _ => Err(format!(
                    "--fn-mtp auto | 0 | k (1..4: the GQA-12 verify cap is 5 rows), not {t}"
                )),
            },
        }
    }
}

impl std::fmt::Display for FnMtp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FnMtp::Off => write!(f, "0"),
            FnMtp::K(k) => write!(f, "{k}"),
            FnMtp::Auto => write!(f, "auto"),
        }
    }
}

impl FnMtp {
    /// Whether any MTP form is on.
    #[must_use]
    pub fn on(self) -> bool {
        self != FnMtp::Off
    }
}

/// FN16f: the served default of `--fn-mtp` (the user's word, 2026-09-29, after FN16's table:
/// BetterBench --quick single 78.8 vs 54.0 tok/s, KL max 0.0138 vs 0.0113 under the 0.0227 bar).
pub const DEFAULT_SERVE_MTP: FnMtp = FnMtp::Auto;
/// FN92: `--fn-mtp-depth` absent = the table's b1 cell (`adaptive` = the arm).
pub const DEFAULT_FN_MTP_DEPTH: &str = "table";

/// FN30 / FN30d: `--fn-moe-train` when absent -- `fused` since FN30d (2026-10-07, on today's
/// served defaults: bytes identical (row sha 16/16, kla 25/25, API ids + logprobs 25/25 on 8 boots),
/// harness iteration b1 25.73 -> 25.24 / b8 55.21 -> 54.69 ms (4 + 4), slow drafter boots 3/4 vs
/// legacy's 4/4, BetterBench --quick single 117.4 -> 122.3 tok/s, conc-8 259.9 -> 261.5 median).
/// `legacy` = the control. The engine's `MoeTrain::default()` (the plan's dry runs) stays `legacy`:
/// every fused boot (FN30 .. FN30d, served and harness) ran on the arena that dry run sizes.
pub const DEFAULT_FN_MOE_TRAIN: &str = "fused";

/// FN84: `--fn-moe-miss-order` when absent -- `id` (FN26's ticket order, the control) until the
/// order law's gate passes.
pub const DEFAULT_FN_MOE_MISS_ORDER: &str = "id";
/// FN85: `--fn-ar-prologue` (`off` = the standalone reduce, the default; `norm` = the arm).
pub const DEFAULT_FN_AR_PROLOGUE: &str = "off";

/// FN86b: `--fn-fuse` -- `all` IS THE DEFAULT since FN86b's hold h1 (2026-10-12: every byte gate
/// identical, b1 verify device span -0.21 ms paired, update p99 22.35 -> 22.20 ms, nothing worse);
/// `none` = every shipped chain, the control. `all` = f1,f2,f4,f5,f6,f7,f10 since the f6/f7 flip
/// (branch fn-fuse-f6f7-default, ATTN-OPT-3's attention-neighbour fusions: the QSA front 3 -> 1 at
/// <= 5 rows a sequence, the combine + gate + q8 2 -> 1; target decode / verify only, prefill and
/// the drafter keep the chains); `f1,f2,f4,f5,f10` = the previous default, the flip's control.
pub const DEFAULT_FN_FUSE: &str = "all";

/// FN111: `--fn-tiles` -- `all` (t1 + t3) IS THE DEFAULT since FN111's hold h2 (2026-10-12: every
/// byte gate identical, b8 w3 verify device span -0.80 ms paired, b1 tied, BetterBench conc-8
/// +1.4 %, nothing worse past the floor); `default` = the shipped cells, the control.
pub const DEFAULT_FN_TILES: &str = "all";
/// FN128: `--fn-sel128` -- `all` IS THE DEFAULT since the pooled holds h1 + h4 (2026-10-14: bytes
/// identical, 64k decode +5.4 % / 32k +2.8 % pooled, the rest inside the floor); `default` = the
/// parents, the control.
pub const DEFAULT_FN_SEL128: &str = "all";

/// FN131: `--fn-hc-down split` IS THE DEFAULT (the phase-split decode downs, `flashnext_hc131`;
/// FLIPPED by the FN131 served hold h1, 2026-10-16: conc-8 +4.1 %, bytes identical); `none` = the
/// shipped decode downs (the control).
/// FN141: `--fn-k141 all` IS THE DEFAULT (the drained-ring twins: HC up, t8 ring at <= 16 rows,
/// the router ring; FLIPPED by the FN141 served hold h1, 2026-10-16: single +2.8 %, conc-8 +1.0 %,
/// bytes identical); `none` = the shipped symbols (the control).
pub const DEFAULT_FN_K141: &str = "all";
/// FN163: `--fn-route163 none` IS THE DEFAULT until the served hold flips it; `all` = the b8 route
/// + the b1 front twins.
pub const DEFAULT_FN_ROUTE163: &str = "none";

pub const DEFAULT_FN_HC_DOWN: &str = "split";

/// FN122b: `--fn-prefill-tiles` -- `hc` IS THE DEFAULT since FN122b's holds h3 + h4 (2026-10-14,
/// pooled N A N A on the served line: prefill 4k / 8k / 16k / 34k +6.4 / +7.3 / +6.8 / +5.5 %,
/// API c1 25/25 + seeded 20/20 identical, single / update p99 tied, conc-8 -0.8 % inside the floor);
/// `default` = the shipped prefill cells, the control. FN122c: `hc,dq` IS THE DEFAULT since FN122c's
/// holds h6 + h7 (2026-10-14, both orders, pooled vs `hc`: prefill 4k / 8k / 16k / 34k +5.2 / +6.5 /
/// +5.1 / +4.9 %, API c1 25/25 + seeded 20/20 identical, single tied, conc-8 +1.3 %, update p99
/// +0.3 ms inside the floor); `hc` = FN122b's form.
pub const DEFAULT_FN_PREFILL_TILES: &str = "hc,dq";

/// FN122b: the prefill tiles of `--fn-prefill-tiles` (`saber_kernels::flashnext::prefill122`).
///
/// # Errors
/// An unknown family.
pub fn prefill122(f: &FnFlags) -> anyhow::Result<u32> {
    saber_kernels::flashnext::prefill122::parse(&f.fn_prefill_tiles).map_err(|e| anyhow::anyhow!(e))
}

/// FN120b / FN120e: `--fn-glue` -- `fold` IS THE DEFAULT (FN120b's pooled 4 + 4: update p50 -0.31 /
/// p99 -0.45 ms, 153 -> 15 glue dispatches a b1 cycle); `off` (the per-layer seed fills +
/// route-id-log copies) = the control. FN120e: bit-exact single-stream at every band (h2 law55
/// 55k / 135k) and on every byte-law cell; a conc cell with sparse rows past 2,050 is
/// schedule-dependent in EVERY arm (FN24's union exemption: `off` vs `off` differ across boots),
/// so such a cell never decides a flip.
pub const DEFAULT_FN_GLUE: &str = "fold";

/// FN120b: the glue fold's form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glue120 {
    Off,
    Seeds,
    Fold,
}

/// FN120b: `--fn-glue`'s form.
///
/// # Errors
/// An unknown form.
pub fn glue120(f: &FnFlags) -> anyhow::Result<Glue120> {
    Ok(match f.fn_glue.as_str() {
        "off" => Glue120::Off,
        "seeds" => Glue120::Seeds,
        // FN120e: `fold-strict` = `fold` with FN120b's loud refusal of a run past the region
        "fold" | "fold-strict" => Glue120::Fold,
        o => anyhow::bail!("--fn-glue {o}: off | seeds | fold | fold-strict"),
    })
}

/// FN120e: a run whose seeds would pass the glue region REFUSES (FN120b's law) only under
/// `--fn-glue fold-strict` or on a boot that is not `--fn-serve` (the gates and the harness);
/// a served boot SPILLS that run to the per-layer fills (counted, logged), never a fatal.
#[must_use]
pub fn glue_strict(f: &FnFlags) -> bool {
    f.fn_glue == "fold-strict" || f.fn_serve.is_none()
}

/// FN120b / FN120e: rank `r`'s glue seed region as allocated (`fn.glue120`): the boot's derived
/// region ([`glue_census`]) under `seeds` / `fold`, nothing under `off`.
#[must_use]
pub fn glue_class(f: &FnFlags, region: super::exec::GlueRegion) -> u64 {
    if glue120(f).is_ok_and(|g| g != Glue120::Off) {
        region.total()
    } else {
        0
    }
}

/// FN120e: one served decode / verify form's glue-fold high-water (bytes from each part's base;
/// the ids block included under `fold`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlueFormNeed {
    pub b: u32,
    pub w: u32,
    /// Every row at a position <= 2,050 (the dense decode; FN19's `short_rows`).
    pub short: bool,
    /// The step's context band (`cc.max_ctx`, FN51f).
    pub band: u32,
    pub zero: u64,
    pub ones: u64,
}

impl GlueFormNeed {
    #[must_use]
    pub fn name(&self) -> String {
        format!(
            "b{} w{} {} band {}",
            self.b,
            self.w,
            if self.short { "short" } else { "sparse" },
            self.band
        )
    }
}

/// FN120e: THE GLUE CENSUS -- every decode / verify form this boot can capture as a glue-fold
/// run, emitted on the host, and the region the size law derives from the widest.
#[derive(Debug, Clone)]
pub struct GlueCensus {
    pub forms: Vec<GlueFormNeed>,
    /// The widest form by each part (indices into `forms`).
    pub widest_zero: usize,
    pub widest_ones: usize,
    pub region: super::exec::GlueRegion,
}

impl GlueCensus {
    /// The `fnglue120` boot line's census clause.
    #[must_use]
    pub fn line(&self) -> String {
        let (z, o) = (&self.forms[self.widest_zero], &self.forms[self.widest_ones]);
        format!(
            "FN120e sized at boot from {} forms: widest zero {} = {} B, widest ones {} = {} B; \
             region = zero {} + ones {} B (each part: the widest + {} B, whole 2 MiB pages)",
            self.forms.len(),
            z.name(),
            z.zero,
            o.name(),
            o.ones,
            self.region.zero,
            self.region.ones,
            super::exec::GLUE_MARGIN_BYTES
        )
    }
}

/// FN120e: the (bucket, width, short, band) forms a boot of pools context `max_ctx` and verify
/// width `pool_w` can run as a captured decode / verify step -- the worker's own laws: buckets
/// [`super::worker::BUCKETS`], widths `1..=pool_w` (every width a bucket's steps can run), the
/// context bands of `kvwire::band_of` (FN51f: the bands below the boot's context + its context;
/// one band at or under 10,240 or under `--fn-control band-boot`), the short (dense) decode only
/// in the first band (its rows sit at <= 2,050), the sparse decode in every band (`--fn-qsa
/// dense`: the same forms, all dense).
#[must_use]
pub fn glue_forms(f: &FnFlags, max_ctx: u32, pool_w: u32) -> Vec<(u32, u32, bool, u32)> {
    let mut bands: Vec<u32> = super::kvwire::CTX_BANDS
        .iter()
        .copied()
        .filter(|&b| b < max_ctx)
        .collect();
    bands.push(max_ctx);
    if max_ctx <= super::kvwire::DEFAULT_FN_KV_MAX_CTX
        || f.fn_control.as_deref() == Some("band-boot")
    {
        bands = vec![max_ctx];
    }
    let long = max_ctx > saber_kernels::fn19::DENSE_MAX_POS + 1;
    let mut v = Vec::new();
    for (i, &band) in bands.iter().enumerate() {
        for &b in &super::worker::BUCKETS {
            for w in 1..=pool_w.max(1) {
                if i == 0 {
                    v.push((b, w, true, band));
                }
                if long {
                    v.push((b, w, false, band));
                }
            }
        }
    }
    v
}

/// FN120e: [`GlueCensus`] for this boot: every form of [`glue_forms`] emitted as ONE glue-fold
/// run (`plan::dry_run_glue`) under the boot's rules and emitter flags (the W8 trunk / head, the
/// MoE train, the verify class, the expert form, `--fn-qsa`).
///
/// # Errors
/// A dry run's emission error.
pub fn glue_census(
    decls: &Decls,
    rules: &FlashNextRules,
    f: &FnFlags,
    prompts: &[Prompt],
    xtp2: bool,
) -> anyhow::Result<GlueCensus> {
    let (max_ctx, pool_w) = super::worker::pools_geometry(f, prompts)?;
    let mut forms = Vec::new();
    for (b, w, short, band) in glue_forms(f, max_ctx, pool_w) {
        let cc = super::exec::ChunkConsts {
            m: b * w,
            b,
            w,
            decode: true,
            nb: band.div_ceil(16) + 1,
            max_ctx: band,
            // `--fn-qsa dense`: the dense decode at every row, the short form's emission
            qsa_short: short || f.fn_qsa == saber_kernels::fn19::QsaMode::Dense,
            ..super::exec::ChunkConsts::default()
        };
        let (zero, ones) = plan::dry_run_glue(
            decls,
            rules,
            cc,
            f.fn_verify_class,
            f.fn_moe_train,
            f.trunk8(),
            f.trunk8_head(),
            xtp2,
        )?;
        forms.push(GlueFormNeed {
            b,
            w,
            short,
            band,
            zero,
            ones,
        });
    }
    let widest = |k: fn(&GlueFormNeed) -> u64| {
        forms
            .iter()
            .enumerate()
            .max_by_key(|(i, x)| (k(x), std::cmp::Reverse(*i)))
            .map_or(0, |(i, _)| i)
    };
    let (widest_zero, widest_ones) = (widest(|x| x.zero), widest(|x| x.ones));
    let region = super::exec::GlueRegion::of_need(
        forms.get(widest_zero).map_or(0, |x| x.zero),
        forms.get(widest_ones).map_or(0, |x| x.ones),
    );
    Ok(GlueCensus {
        forms,
        widest_zero,
        widest_ones,
        region,
    })
}

/// FN120c: `--fn-dataflow` -- `off` (the launch train) IS THE DEFAULT; `gdn` = the b1 verify's
/// GDN mixer tail as one persistent dataflow launch (an ARM until its hold flips it).
pub const DEFAULT_FN_DATAFLOW: &str = "off";

/// FN120c: the dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataflow120 {
    Off,
    Gdn,
    /// The gate's CONTROL: the same launch with no stage waiting for its producer (must differ).
    GdnNoDep,
}

/// FN120c: `--fn-dataflow`'s form.
///
/// # Errors
/// An unknown form.
pub fn dataflow120(f: &FnFlags) -> anyhow::Result<Dataflow120> {
    Ok(match f.fn_dataflow.as_str() {
        "off" => Dataflow120::Off,
        "gdn" => Dataflow120::Gdn,
        "gdn-nodep" => Dataflow120::GdnNoDep,
        o => anyhow::bail!("--fn-dataflow {o}: off | gdn | gdn-nodep (the control)"),
    })
}

/// FN120c: rank `r`'s dataflow table region as allocated (`fn.df120`: the site tables + their
/// control words under `gdn`, nothing under `off`).
#[must_use]
pub fn dataflow_class(f: &FnFlags) -> u64 {
    if dataflow120(f).is_ok_and(|d| d != Dataflow120::Off) {
        super::exec::DF120_BYTES
    } else {
        0
    }
}

/// FN124: the in-situ QSA twin's counter region (the ledger class `fn.qsa124`).
pub const QSA124_TWIN_BYTES: u64 = 256;

/// FN124: rank `r`'s twin counter region as allocated (only under `qsatwin`).
#[must_use]
pub fn qsa124_class(f: &FnFlags) -> u64 {
    if prefill122(f).is_ok_and(|m| m & saber_kernels::flashnext::prefill122::QSATWIN != 0) {
        QSA124_TWIN_BYTES
    } else {
        0
    }
}

/// FN111: the re-tiles of `--fn-tiles` (`saber_kernels::flashnext::tiles111`).
///
/// # Errors
/// An unknown tile.
pub fn sel128(f: &FnFlags) -> anyhow::Result<u32> {
    saber_kernels::flashnext::sel128::parse(&f.fn_sel128).map_err(|e| anyhow::anyhow!(e))
}

/// FN131: the decode downs of `--fn-hc-down` (`saber_kernels::flashnext::down131`).
///
/// # Errors
/// An unknown family.
pub fn down131(f: &FnFlags) -> anyhow::Result<u32> {
    saber_kernels::flashnext::down131::parse(&f.fn_hc_down).map_err(|e| anyhow::anyhow!(e))
}

/// FN141: the drained-ring twins of `--fn-k141` (`saber_kernels::flashnext::k141`).
///
/// # Errors
/// An unknown family.
pub fn k141(f: &FnFlags) -> anyhow::Result<u32> {
    saber_kernels::flashnext::k141::parse(&f.fn_k141).map_err(|e| anyhow::anyhow!(e))
}

/// FN163: the route twins of `--fn-route163` (`saber_kernels::flashnext::route163`).
///
/// # Errors
/// An unknown band.
pub fn route163(f: &FnFlags) -> anyhow::Result<u32> {
    saber_kernels::flashnext::route163::parse(&f.fn_route163).map_err(|e| anyhow::anyhow!(e))
}

/// FN111: the re-tiles of `--fn-tiles` (`saber_kernels::flashnext::tiles111`).
pub fn tiles111(f: &FnFlags) -> anyhow::Result<u32> {
    saber_kernels::flashnext::tiles111::parse(&f.fn_tiles).map_err(|e| anyhow::anyhow!(e))
}

/// FN86b: the launch-fusion families of `--fn-fuse` (`saber_kernels::flashnext::fuse86`).
///
/// # Errors
/// An unknown family or one this tree does not build.
pub fn fuse86(f: &FnFlags) -> anyhow::Result<u32> {
    saber_kernels::flashnext::fuse86::parse(&f.fn_fuse).map_err(|e| anyhow::anyhow!(e))
}

/// FN114b / FN114d: `--fn-ar` -- `gp` (the shipped one-shot bodies with the workspace slot chosen
/// by the call's GLOBAL parity: closes LEDGER FN114b's slot-reuse hazard by construction,
/// byte-equal) IS THE DEFAULT since FN114d's pooled holds (6 + 6 boots); `oneshot` (the reference's
/// per-block parity: the hazard) is the control; `ll` (the flag carried in the data words) is an
/// arm.
pub const DEFAULT_FN_AR: &str = "gp";

/// FN114b: whether the exact reduce runs the flag-in-data form.
#[must_use]
pub fn ar_ll(f: &FnFlags) -> bool {
    f.fn_ar == "ll"
}

/// FN114d: whether the exact and wht6 reduces run their global-parity twins.
#[must_use]
pub fn ar_gp(f: &FnFlags) -> bool {
    f.fn_ar == "gp"
}

/// FN137: `--fn-moe-ar-push on` IS THE DEFAULT (the MoE-out reduce fused into the tp2 back half;
/// FLIPPED by the FN145 served holds h1 + h2, 2026-10-16: pooled 4 + 4 vs the served defaults,
/// single +2.1 %, conc-8 +4.6 %, bytes identical; stands down, counted, off tp2's half shape);
/// `off` = the standalone MoE-out reduce (the control).
pub const DEFAULT_FN_MOE_AR_PUSH: &str = "on";

/// FN137: whether the tp2 back half carries the MoE-out all-reduce (`--fn-moe-ar-push on`).
#[must_use]
pub fn moe_ar_push(f: &FnFlags) -> bool {
    f.fn_moe_ar_push == "on"
}

/// FN164: `--fn-mixer-ar-push on` IS THE DEFAULT (the mixer-out reduce fused into the out / o
/// projection at every target decode / verify form; FLIPPED by FN164's served hold h1,
/// 2026-10-21: pooled 4 + 4 vs `off`, single +2.6 %, conc-8 +0.1 %, bytes identical); `off` = the
/// standalone mixer-out reduce (the control), `b1` = the one-sequence forms only (an arm).
pub const DEFAULT_FN_MIXER_AR_PUSH: &str = "on";

/// FN164: where the out / o projection carries the mixer-out all-reduce (`--fn-mixer-ar-push`).
#[must_use]
pub fn mixer_ar_push(f: &FnFlags) -> super::dev::MixPush {
    match f.fn_mixer_ar_push.as_str() {
        "on" => super::dev::MixPush::On,
        "b1" => super::dev::MixPush::B1,
        _ => super::dev::MixPush::Off,
    }
}

/// FN116: `--fn-first-step` -- the drafter's first round after a prompt. `pass` (today's: the
/// prompt's m-row prefill-form pass, the first draft read on the host, then the chain) is the
/// control; `kv` = the rows 0..m-2 run only the MTP layer's K-side prefix (pre, mix, projection,
/// q/k norm, indexer and KV append: everything a later drafter pass reads of them), then the last
/// row is a steady drafter round (the decode-form pass with the device tail and the chain).
/// FLIPPED (FN116 h5 + h6, pooled 4 + 4 boots both orders): `kv` was the default, `pass` the control.
/// FN119f (h17 + h18, pooled 4 + 4 boots both orders): `dev` IS THE DEFAULT (the first token walked
/// on the card behind the chunk, the first round queued behind it; update p99 19.45 -> 18.90 ms,
/// TTFT p50 67.3 -> 64.5, first decode step 19.8 -> 18.5, bytes identical); `kv` = the control.
pub const DEFAULT_FN_FIRST_STEP: &str = "dev";

/// FN116: whether the first drafter round runs the K-side prefix + a steady round.
#[must_use]
pub fn first_step_kv(f: &FnFlags) -> bool {
    matches!(
        f.fn_first_step.as_str(),
        "kv" | "kvnext" | "chunk" | "dev" | "under"
    )
}

/// FN119f: whether the first token is walked on the card behind the chunk (`dev`).
#[must_use]
pub fn first_step_dev(f: &FnFlags) -> bool {
    matches!(f.fn_first_step.as_str(), "dev" | "under")
}

/// FN119g: whether the first round's K-side front runs under the chunk's head (`under`).
#[must_use]
pub fn first_step_under(f: &FnFlags) -> bool {
    f.fn_first_step == "under"
}

/// FN119e: whether the first round's ids are left pending (`kvnext`, `chunk`).
#[must_use]
pub fn first_step_defer(f: &FnFlags) -> bool {
    matches!(
        f.fn_first_step.as_str(),
        "kvnext" | "chunk" | "dev" | "under"
    )
}

/// FN119e: whether the K-side front runs under the last prompt chunk (`chunk`).
#[must_use]
pub fn first_step_front(f: &FnFlags) -> bool {
    matches!(f.fn_first_step.as_str(), "chunk" | "dev" | "under")
}

/// FN85: whether the prologue-reduce arm is on.
#[must_use]
pub fn ar_prologue(f: &FnFlags) -> bool {
    f.fn_ar_prologue == "norm"
}

/// FN84: the back half's order by the flags.
#[must_use]
pub fn miss_order(f: &FnFlags) -> saber_kernels::moe::MissOrder {
    saber_kernels::moe::MissOrder::parse(&f.fn_moe_miss_order).unwrap_or_default()
}

/// FN84: a rank's pinned-RAM expert slab as device addresses `[lo, hi)` (the order law's RAM
/// predicate: the tier's RAM slots incl. the spares; elastic-region and P2P slots are VRAM).
#[must_use]
pub fn miss_ram_range(ram_device_addr: u64, slab_slots: u32, stride: u64) -> (u64, u64) {
    (
        ram_device_addr,
        ram_device_addr + u64::from(slab_slots) * stride,
    )
}

/// FN72: `--fn-sdma-lanes` when absent -- `shared` (ROCclr's pooled hardware queues, the control).
pub const DEFAULT_FN_SDMA_LANES: &str = "shared";

/// FN30c: the tier mover's HOLD law (a verify step's copies issued after the drafter round) when
/// neither `--fn-mover-hold` nor `--no-fn-mover-hold` is given.
pub const DEFAULT_FN_MOVER_HOLD: bool = false;

/// FN56: the host levers on when `--fn-host-levers` is absent. `pick` / `ple` each win ~0.5 ms a
/// b8 iteration in the harness and are byte-identical, but the served 4+4 gate (hold 2,
/// 2026-10-05) found `pick,ple` BIMODAL at conc-8 (162-167 tok/s in 4 passes, 211-215 in 4,
/// vs 181-195 for `none`): the threaded pick is boot-placement sensitive beside the spinning
/// HIP threads. The default stays `none` (the control) until FN57 pins the pick's threads and
/// A/Bs `ple` alone; `defer` / `begin` LOSE (the layer gates wait for copies issued ~1 ms later).
/// FN93 (2026-10-09, LEDGER FN93): **`pick` IS THE DEFAULT** -- the d8 step's host term is the
/// pick's host seam (the sampled rows' log-sum-exps, 3.7 ms of 5.7 after the wait; the tier end is
/// 0.67), threads cut it to 1.2: d8 step 52.7 -> 51.0 ms, BetterBench --quick conc-8 277.0 ->
/// 282.5 tok/s (4 + 4, every P pass above every D pass), single tied, API 25/25 at conc 1 and 8,
/// row sha 16/16, placement lines identical, no bimodality under `--host-wait block`.
/// `--fn-host-levers none` = the control.
pub const DEFAULT_FN_HOST_LEVERS: &str = "pick";
/// FN112: `--fn-overlap` when absent -- `on` IS THE DEFAULT (the orchestrator's flip on h2, 2026-10-12:
/// bytes identical, d8 post 3.09 -> 1.91 ms, conc-8 +2.4 %); `off` = the control (the step's reads
/// issued after its wait).
pub const DEFAULT_FN_OVERLAP: bool = true;

/// LR-N: `--spec-couple` when absent -- OFF (an arm; the user decides a flip). `on` = coupled
/// drafting (the drafts take the target's noise; the committed tokens are unchanged).
pub const DEFAULT_SPEC_COUPLE: bool = false;

/// FN14 / FN64 / FN64c: `--fn-graph-split` when absent -- `lazy` since FN64c (2026-10-06: FN64's
/// runs with the recorded set pre-captured; served 4 + 4 tied, graph launches a step 49 -> 9.3,
/// a rank's CPU 56 -> 29 %, VRAM slots equal, bytes identical; LEDGER FN64c); `layer` (one graph
/// per decoder layer + the head) = the control; `step` = one graph, the FN56 loser.
pub const DEFAULT_FN_GRAPH_SPLIT: &str = "lazy";
/// FN90a / the user's power directive (2026-10-14 20:45 CDT: "remove that SSD drive keep awake ...
/// power efficient within reason"): `--fn-ple-drive-keepalive-ms` when absent -- `0` = NO READER
/// since 2026-10-14 (the drive follows APST and idles in its low-power states); `40` = FN90a's
/// keep-awake, now the ARM: one 4 KiB O_DIRECT read every 40 ms kept the PLE drive out of APST
/// (its exit costs 8.9 ms after >= 110 ms idle; 1.2 % of served b1 steps paid it inside the device
/// span; served 4 + 4: update p99 29.4 -> 27.5 ms, single / conc-8 tied, bytes identical; LEDGER
/// FN90a). Bytes identical either way (timing only). The root alternative to both is
/// `nvme_core.default_ps_max_latency_us` (the user's call).
pub const DEFAULT_FN_PLE_DRIVE_KEEPALIVE_MS: u64 = 0;
/// FN106d: `--fn-pinned-pages` when absent -- `huge` since FN106d (2026-10-10: bytes identical, the
/// b8 drafter slow mode 0 of 7 huge boots vs 2 of 3 shmem, update p99 -0.8 ms, conc-8 +1.8 %);
/// `shmem` (`hipHostMalloc`, 4 KiB shmem pages) = the control.
pub const DEFAULT_FN_PINNED_PAGES: saber_hip::PinnedPages = saber_hip::PinnedPages::Huge;
/// FN64: `--fn-graph-lazy-q` when absent (blocks of 4 layers: 13 cut points, <= 91 runs a form).
pub const DEFAULT_FN_GRAPH_LAZY_Q: u32 = 4;
/// FN64: `--fn-graph-lazy-cache` when absent (captured runs a rank keeps).
pub const DEFAULT_FN_GRAPH_LAZY_CACHE: usize = 1024;
/// FN64b: `--fn-graph-lazy-pre` when absent -- `off` (FN64's capture on first use). Pre-capturing
/// every reachable run measured ~181 MiB of VRAM a step form (2.26 MiB a captured run, ROCm 10.1):
/// the first band's 16 forms would take ~2.9 GiB a card, so `on` / `all` stop at the VRAM budget
/// (`worker::PRE_VRAM_BUDGET`) after ~2 forms and stay diagnostic arms (LEDGER FN64b).
/// FN64c: `recorded` (the pack's recorded served run set; absent file = every run on demand, said
/// by name at boot); `off` = FN64's on-demand capture.
pub const DEFAULT_FN_GRAPH_LAZY_PRE: LazyPre = LazyPre::Recorded;

/// FN64d: the device bytes one hipGraphExec's kernarg pool takes on ROCm 10.1 (FN64c's probe:
/// exactly 2.000 MiB an exec for 1..2,048 nodes; the lever is the exec COUNT).
pub const GRAPH_EXEC_POOL_MIB: u64 = 2;
/// FN64d: the execs today's served lazy boot holds a rank OUTSIDE the ledger (FN64c: 378-395 at
/// the end of a served boot on FN64c's traffic; FN64d MEASURED 702 / 704 on the served
/// launcher's boot (trunk8, 262,144) under BetterBench + GSM8K: the long-row forms are not in the
/// pack's recorded set and are captured on demand) -- what a billed
/// `--fn-graph-exec-budget-mib` returns to the plan budget.
pub const FN64D_UNBILLED_EXECS: u64 = 704;
/// FN64e: the lazy runs the default CHARGE reserves past the pre-captured set for on-demand
/// captures (a long-context band's form the recorded set does not pre-capture; the LRU evicts past
/// pre-captured + reserve).
pub const FN64E_RESERVE_EXECS: u64 = 32;
/// FN64e / FN64f: the drafter's graph execs a lazy boot holds a rank, billed with the lazy runs.
/// FN64d's census counted 12 (the MTP passes + the device chains at the first band); FN64e's
/// record boot held 42 a rank once the long rows reached every band, so the reserve is sized from
/// that census (FN64f).
pub const FN64E_DRAFTER_EXECS: u64 = 42;
/// FN64e: the default charge (MiB a rank) -- the measured working set ([`FN64D_UNBILLED_EXECS`]
/// lazy runs, pre-captured from the recorded set) + [`FN64E_RESERVE_EXECS`] + the drafter's
/// [`FN64E_DRAFTER_EXECS`], at [`GRAPH_EXEC_POOL_MIB`] an exec = 1,556 MiB.
pub const FN64E_CHARGE_MIB: u64 =
    (FN64D_UNBILLED_EXECS + FN64E_RESERVE_EXECS + FN64E_DRAFTER_EXECS) * GRAPH_EXEC_POOL_MIB;

/// FN64d / FN64e: how the lazy runs' graph execs are billed in the ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecBill {
    /// FN64c: outside the ledger (the cache of [`DEFAULT_FN_GRAPH_LAZY_CACHE`] runs, the 256 MiB
    /// free-VRAM floor at pre-capture) -- `--fn-graph-exec-budget-mib 0`, the control; a boot
    /// that is not lazy.
    #[default]
    Unbilled,
    /// FN64d (`--fn-graph-exec-budget-mib N`): the class takes N MiB, the plan grows by today's
    /// unbilled 1,408 MiB (the difference becomes slots); the runs a rank holds <= N / 2.
    Budget(u64),
    /// FN64e (`--fn-graph-exec-charge`, an arm): the class `fn.graph.exec` = [`FN64E_CHARGE_MIB`] and the
    /// plan grows by the same bytes (slots unchanged, the ledger true); the boot pre-captures the
    /// recorded set's most used [`FN64D_UNBILLED_EXECS`] runs, the LRU holds <= that +
    /// [`FN64E_RESERVE_EXECS`].
    Charge,
    /// FN98 (`--fn-graph-form one`): one exec per step form; the class `fn.graph.exec` =
    /// [`FN98_CLASS_MIB`], the plan grows by [`FN98_GROW_MIB`] (the difference = static slots).
    One,
}

impl ExecBill {
    /// The lazy runs a rank may hold (the LRU's cap), `None` = the cache alone.
    #[must_use]
    pub fn run_cap(self) -> Option<usize> {
        match self {
            Self::Unbilled | Self::One => None,
            Self::Budget(n) => Some((n / GRAPH_EXEC_POOL_MIB) as usize),
            Self::Charge => Some((FN64D_UNBILLED_EXECS + FN64E_RESERVE_EXECS) as usize),
        }
    }

    /// The runs the boot pre-captures from the recorded set (most used first), `None` = every
    /// recorded run within the cache and the free-VRAM floor (FN64c).
    #[must_use]
    pub fn pre_cap(self) -> Option<usize> {
        match self {
            Self::Unbilled | Self::One => None,
            Self::Budget(n) => Some((n / GRAPH_EXEC_POOL_MIB) as usize),
            Self::Charge => Some(FN64D_UNBILLED_EXECS as usize),
        }
    }
}

/// FN64e: the exec bill when neither `--fn-graph-exec-budget-mib` nor `--fn-graph-exec-charge` is
/// given -- still [`ExecBill::Unbilled`]: the hold (2026-10-08) measured the charge + the re-recorded
/// set at 4.3-6.7 served capture passes per 1,000 steps vs 2.8-3.2 unbilled (the set's long-row
/// forms crowded the 704-run pre-capture), so neither flipped (LEDGER FN64e).
pub const DEFAULT_FN_GRAPH_EXEC_BILL: ExecBill = ExecBill::Unbilled;

/// FN98: `--fn-graph-form` when absent -- `one` since FN98's hold h3 (2026-10-09: bytes identical,
/// BetterBench --quick single TTFT p50 91.5 -> 81.8 ms, conc-8 301.7 -> 315.2 tok/s, update p99
/// 25.38 -> 24.70 ms, +240 static slots a rank, served capture passes 8-9 -> 0 per 1,000);
/// `lazy` = FN64's run split, the control. A lazy-only arm given without `--fn-graph-form`
/// (`--fn-graph-exec-budget-mib`, `--fn-graph-exec-charge`, `--fn-graph-lazy-record`, a split
/// other than `lazy`) keeps the lazy form.
pub const DEFAULT_FN_GRAPH_FORM: &str = "one";
/// FN98: the decode / verify step-form execs the `one` form's class reserves a rank -- 2 a form
/// (the PLE layer-1 copy point cuts it) x 24 forms (the recorded set's 7-8 forms a rank + the
/// long-row bands on demand) -- and the drafter's (8 at main's served boot, FN96 h2's census).
pub const FN98_FORM_EXECS: u64 = 48;
/// FN98: the drafter's execs in the `one` form's class.
pub const FN98_DRAFTER_EXECS: u64 = 16;
/// FN98: the class `fn.graph.exec` under the `one` form = 128 MiB.
pub const FN98_CLASS_MIB: u64 = (FN98_FORM_EXECS + FN98_DRAFTER_EXECS) * GRAPH_EXEC_POOL_MIB;
/// FN98: the VRAM the `one` form returns to the slots (MiB a rank) -- MEASURED on main's served
/// lazy boot (FN96 h2, 2026-10-09): 904-1,090 MiB free before the lazy pre-capture, 174-268 MiB at
/// the end of served traffic with 400-458 execs held; the `one` form keeps the same end-state
/// margin over its billed class: 904 - 174 - 128 = ~600. (FN95's 1.43 GiB was FN64d's census on
/// an older boot; the first h2 arm grew by 1,492 - 276 MiB and OOMed at its first step.)
pub const FN98_FREED_MIB: u64 = 600;
/// FN98: the plan's growth under the `one` form (MiB a rank): the class + the freed bytes.
pub const FN98_GROW_MIB: u64 = FN98_CLASS_MIB + FN98_FREED_MIB;

/// FN64c: the recorded lazy run set's directory inside the pack (`lazy_runs.rank<r>.json`) --
/// `--fn-graph-lazy-pre recorded` reads it when `--fn-graph-lazy-set` is absent.
pub const LAZY_SET_IN_PACK: &str = "profile";

impl FnFlags {
    /// FN64c: the directory `--fn-graph-lazy-pre recorded` reads (`--fn-graph-lazy-set`, else
    /// the pack's [`LAZY_SET_IN_PACK`]).
    #[must_use]
    pub fn lazy_set_dir(&self, pack: &Path) -> PathBuf {
        self.fn_graph_lazy_set
            .clone()
            .unwrap_or_else(|| pack.join(LAZY_SET_IN_PACK))
    }
}

/// FN64c: rank `r`'s recorded lazy run file in `dir`.
#[must_use]
pub fn lazy_set_file(dir: &Path, rank: u32) -> PathBuf {
    dir.join(format!("lazy_runs.rank{rank}.json"))
}

/// FN64b: which step forms `--fn-graph-split lazy` captures at boot (`--fn-graph-lazy-pre`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LazyPre {
    /// `off`: none (FN64: every run captured on first use).
    Off,
    /// `recorded` (FN64c): the runs a served boot used (`--fn-graph-lazy-record`), loaded from
    /// `--fn-graph-lazy-set` (the pack's `profile/` by default), every band's forms.
    Recorded,
    /// `on`: the first context band's forms (short + long rows).
    FirstBand,
    /// `all`: every band's forms, up to the cache.
    All,
}

/// FN14 / FN64: how a captured decode step is cut into graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphSplit {
    /// One graph per decoder layer + one for the head (each layer's gate before its graph).
    Layer,
    /// One graph for the step (every gate hoisted before it).
    Step,
    /// FN64: one graph per run of layers with no pending copy; `q` = the cut quantum (layers),
    /// `cache` = the runs a rank keeps; FN64b: `pre` = the forms whose runs are captured at boot.
    Lazy { q: u32, cache: usize, pre: LazyPre },
}

/// FN58b / FN58c: `--fn-ple-overlap` when absent -- `layer1-inline` since FN58c (bit-exact; the
/// gather hidden under layer 0, the copy on the step's own stream; `off` = the control).
pub const DEFAULT_FN_PLE_OVERLAP: &str = "layer1-inline";
/// FN53b: the valve's ceiling a rank (FN53a: 1,024 records = 4.99 GiB freed on two ranks).
pub const DEFAULT_FN_MEM_VALVE_MAX_TAIL: u32 = 1024;
/// FN53c: the valve's trigger law (`--fn-mem-valve-trigger`): `boot` = the boot's free floor
/// minus a margin; `fixed` = FN53b's constants, the control.
pub const DEFAULT_FN_MEM_VALVE_TRIGGER: &str = "boot";
/// FN53c: an open valve's decode / verify steps keep their captured graphs (`--fn-valve-graphs`):
/// `on` = the device miss flag; `off` = FN53b's demand path, the control.
pub const DEFAULT_FN_VALVE_GRAPHS: &str = "off"; // FN53c h1: the static open set loses to the demand path (b8 261 vs 386 tok/s); an ARM
/// FN51f-b: ELASTIC VRAM's default (`--fn-vram-elastic`): ON since 2026-10-08 on the user's word
/// ("elastic vram should be enabled as that is a clear win") -- bit-exact, conc-8 258.5 -> 273.9,
/// single +2 %, a 262k request's KV RAM reads -74 %; `off` = the control.
pub const DEFAULT_FN_VRAM_ELASTIC: bool = true;
/// FN51f-b: the static expert slots a rank keeps at least under the elastic carve.
pub const DEFAULT_FN_VRAM_ELASTIC_MIN_SLOTS: u32 = 6000;
/// FN51f-b: the region's borrow past the boot window (GiB a rank).
pub const DEFAULT_FN_VRAM_ELASTIC_KV_MAX_GIB: f64 = 0.5; // 1.5 until 2026-10-08: with the kernel's 2 GiB P2PDMA pool billed (FN71b) the 1.5 GiB carve leaves 5,702-5,723 static slots, under the 6,000 floor, and a PLAIN served boot was REFUSED (FN73b found it); 0.5 fits (6,112 static slots, the tier complete) and still lends 360 slots at boot (FN51f-c). FN76 re-sizes the budget; revisit then.
/// FN51f-c: `--fn-vram-elastic-refill` when absent -- `lazy` (FN51f-b's law): `eager` is bit-exact
/// but showed no win at the served boot (no refill dip: prefill passes already refill; LEDGER FN51f-c).
pub const DEFAULT_FN_VRAM_ELASTIC_REFILL: &str = "lazy";
/// FN130: `--fn-vram-elastic-demand` when absent -- `on` (a served boot reclaims on demand; a
/// non-served boot runs it strict, [`elastic_demand_strict`]).
pub const DEFAULT_FN_VRAM_ELASTIC_DEMAND: &str = "on";

/// FN130: the demand law's form as parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticDemandForm {
    /// Reclaim on demand.
    On,
    /// Refuse loudly where a demand reclaim would be needed.
    Strict,
    /// No demand site (the control).
    Off,
}

/// FN130: `--fn-vram-elastic-demand`.
///
/// # Errors
/// An unknown form.
pub fn elastic_demand_form(f: &FnFlags) -> anyhow::Result<ElasticDemandForm> {
    Ok(match f.fn_vram_elastic_demand.as_str() {
        "on" => ElasticDemandForm::On,
        "strict" => ElasticDemandForm::Strict,
        "off" => ElasticDemandForm::Off,
        o => anyhow::bail!("--fn-vram-elastic-demand {o}: on | strict | off"),
    })
}

/// FN130 (FN120e's pattern): a demand site REFUSES under `--fn-vram-elastic-demand strict` or on
/// a boot that is not `--fn-serve` (the gates and the harness); a served boot reclaims.
#[must_use]
pub fn elastic_demand_strict(f: &FnFlags) -> bool {
    f.fn_vram_elastic_demand == "strict" || f.fn_serve.is_none()
}
/// FN53b: the bounce a rank (two halves of 32 records: FN53a's balanced tail at 1,024 is 22 a
/// layer).
pub const FN_MEM_VALVE_BOUNCE: u32 = 64;
/// FN53b: a prefill pass of at least this many rows reads its tail blind one layer ahead
/// (FN53a: a 2,048-row chunk routes ~every record; a shorter one reads its tail on demand).
pub const FN_MEM_VALVE_PREFETCH_ROWS: u64 = 1024;

/// FN58c: the halves of the PLE overlap (`--fn-ple-overlap`): `pre` = the committed token's rows
/// gathered under the drafter round (the pre-hook, half A); `side` = the late rows' copy crosses on
/// a side stream (FN58b) vs in order on the step's own stream between the layer-0 and layer-1
/// graphs (no extra stream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleOverlap {
    /// Half A: the pre-hook under the drafter round.
    pub pre: bool,
    /// The late rows' copy on a side stream (`false`: in order on the step's stream).
    pub side: bool,
}

/// FN57: the Flash-Next sleeper when `--host-wait` is absent: **`block`** since FN57 (2026-10-05,
/// hold 2, 4 + 4 interleaved with `--fn-pick device`: API ids 25/25 on every boot, conc-8 208-211
/// vs 185-191 tok/s in all 8 passes, single 96.7-99.2 vs 89.6-94.1, rank CPU 178 -> 106 %).
/// `spin` (the runtime's active wait, the pre-FN57 sleeper) is the control; `aimed` LOSES here
/// (b1 iteration +8 ms: the aim's sleep overshoots a step whose length moves with the tier).
pub const DEFAULT_FN_HOST_WAIT: saber_exec::HostWait = saber_exec::HostWait::Block;

/// FN57: the pick when `--fn-pick` is absent: **`device`** since FN57 (the seam top-64 on the card
/// and the memoized log-sum-exps: pick 4.0 -> 0.95 ms a served step, bytes identical); `host` = the
/// control.
pub const DEFAULT_FN_PICK: &str = "device";

/// FN55c: the vision form when `--fn-vision` is absent: `lazy` since 2026-10-08 (the user's word)
/// -- the tower's region is lent to the expert tier until the first image request loads it
/// (~240 ms), released after `--fn-vision-idle-s`; text bytes and text speed are `off`'s (LEDGER
/// FN55c). Its lent region keeps a RAM shadow per lent record (+0.95 GiB a rank at 262,144), which
/// the fp8 KV default (FN68b, -4.3 GiB pinned) pays for. `off` = the text-only control.
pub const DEFAULT_FN_VISION: &str = "lazy"; // FN55c: lazy (the tower lent to the experts until the first image), the user's word 2026-10-08; off = the text-only control

/// FN55c: the lendable slots of a budget's `fn.vision` class (0 without one).
fn lazy_slots(b: &Ep2Budget) -> u64 {
    b.runtime
        .iter()
        .find(|c| c.name == super::vision::CLASS)
        .map_or(0, |c| c.bytes / b.stride)
}

/// FN55c: the tower's arena in patches when `--fn-vision-max-patches` is absent (FN55b's).
pub const DEFAULT_FN_VISION_MAX_PATCHES: u32 = 16_384;

/// FN55c: `--fn-vision-idle-s` (seconds with no image request before the lazy tower leaves).
pub const DEFAULT_FN_VISION_IDLE_S: u64 = 300;

/// FN125: `--fn-image-cache` when absent. FLIPPED to `on` on hold h1 (2026-10-14; LEDGER FN125:
/// agent A == V / A == A every turn, images and API c1 unchanged, the agent's returning-turn TTFT
/// 6.8 -> 1.3 s, BetterBench inside the spread): image prompts keyed by their content hashes into
/// the tree / the cache, the tower deferred. `off` = FN55b's bypass, the control (a cold image
/// prompt may answer differently from it through the tree's cut law -- FN113's move).
pub const DEFAULT_FN_IMAGE_CACHE: &str = "on";

/// FN125: whether image prompts meet the prefix tree / prompt cache (`--fn-image-cache on`).
#[must_use]
pub fn image_cache_on(f: &FnFlags) -> bool {
    f.fn_image_cache
        .as_deref()
        .unwrap_or(DEFAULT_FN_IMAGE_CACHE)
        == "on"
}

/// FN125: whether an encode DEFERS its tower run: the arm is on and the boot has a cache (`pc`)
/// whose hits could make the run unnecessary (without one every image prompt is computed: the
/// tower runs at encode, FN55b's form).
#[must_use]
pub fn image_cache_defers(f: &FnFlags, pc: bool) -> bool {
    pc && image_cache_on(f)
}

/// FN55c: where the tower lives at this boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionMode {
    /// A text-only boot (an image is refused by name).
    Off,
    /// FN55b: the tower resident from boot (`fn.vision` billed before the expert slots).
    On,
    /// FN55c: the tower's region (`fn.vision`) LENT to the expert tier at boot; the first image
    /// request reclaims it and uploads the tower, the idle release lends it back.
    Lazy,
}

/// FN57: the thread placement when `--fn-pin` is absent: the scheduler's (today's).
pub const DEFAULT_FN_PIN: &str = "off";

/// FN57: `--host-wait` for Flash-Next (absent = [`DEFAULT_FN_HOST_WAIT`]).
///
/// # Errors
/// An unknown arm name.
pub fn host_wait_fn(v: Option<&str>) -> Result<saber_exec::HostWait, String> {
    v.map_or(Ok(DEFAULT_FN_HOST_WAIT), str::parse)
}

/// FN56: the host levers of one boot (`--fn-host-levers`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostLevers {
    /// The pick's seam top-k on parallel host threads.
    pub pick: bool,
    /// A verify step's PLE rows in one gather.
    pub ple: bool,
    /// A verify step's tier end deferred under the drafter round.
    pub defer: bool,
    /// The next step's expert tables built (and uploaded) by the deferred tier end.
    pub begin: bool,
}

impl std::str::FromStr for HostLevers {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let mut l = HostLevers::default();
        for t in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match t {
                "none" => {}
                "all" => {
                    l = HostLevers {
                        pick: true,
                        ple: true,
                        defer: true,
                        begin: true,
                    }
                }
                "pick" => l.pick = true,
                "ple" => l.ple = true,
                "defer" => l.defer = true,
                "begin" => l.begin = true,
                _ => {
                    return Err(format!(
                        "--fn-host-levers: pick | ple | defer | begin | all | none, not {t}"
                    ))
                }
            }
        }
        Ok(l)
    }
}

/// FN37: the producer-rr law when neither `--fn-hc-rr-producer` nor `--no-fn-hc-rr-producer` is
/// given.
pub const DEFAULT_FN_HC_RR_PRODUCER: bool = false;

/// FN78: the decode-form reduce band law (`--fn-decode-ar`): `exact` = decode forms of <= 64 rows
/// reduce on the exact kernel (MTP depth >= 3 at b8 keeps the width-1 byte law); `band` = the
/// payload band (the pre-FN78 law, the control).
pub const DEFAULT_FN_DECODE_AR: &str = "exact";

/// FN96: the state pools' layout (`--fn-pools-layout`): `seq` = the sequence-major regions (THE
/// DEFAULT since FN96, 2026-10-09: bytes identical, TTFT p50 -15 ms with the lend), `layer` = FN13's
/// per-layer pools (the control).
pub const DEFAULT_FN_POOLS_LAYOUT: &str = "seq";
/// FN100: the row-scaled buffers' layout (`one` = FN100's; `separate` = the control).
pub const DEFAULT_FN_BUFFERS_LAYOUT: &str = "one";

/// FN99: the token embedding half's placement (`--fn-embed-placement`): `ram` = pinned host RAM
/// read in place by the drafter chain (**THE DEFAULT since FN107**, 2026-10-09, the user's word
/// "yes to flip 1, 4": FN99 h1 TTFT p50 90.0 -> 83.0 ms, conc-8 299.0 -> 311.6 tok/s, +243 static
/// slots a rank, bit-exact, pinned-neutral; a plain boot whose L2 would go incomplete STANDS DOWN
/// to `vram`), `vram` = FN22's resident half (the control).
pub const DEFAULT_FN_EMBED_PLACEMENT: &str = "ram";

impl FnFlags {
    /// FN96: the state pools' layout (`--fn-pools-layout`; clap admits only the two names).
    #[must_use]
    pub fn pools_layout(&self) -> super::seqpool::PoolsLayout {
        super::seqpool::PoolsLayout::parse(&self.fn_pools_layout)
            .unwrap_or(super::seqpool::PoolsLayout::Seq)
    }

    /// FN106d: the page form of the pinned host allocations (`--fn-pinned-pages`, absent =
    /// [`DEFAULT_FN_PINNED_PAGES`]).
    pub fn pinned_pages(&self) -> anyhow::Result<saber_hip::PinnedPages> {
        self.fn_pinned_pages
            .as_deref()
            .map_or(Ok(DEFAULT_FN_PINNED_PAGES), saber_hip::PinnedPages::parse)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// FN99: the embedding half is PLANNED in pinned host RAM (`--fn-embed-placement`, absent =
    /// [`DEFAULT_FN_EMBED_PLACEMENT`]). The placement a boot RUNS is the plan's
    /// (`Boot::host.embed_per_rank > 0`): a plain boot may stand down to `vram` (FN107).
    #[must_use]
    pub fn embed_in_ram(&self) -> bool {
        self.fn_embed_placement
            .as_deref()
            .unwrap_or(DEFAULT_FN_EMBED_PLACEMENT)
            == "ram"
    }

    /// FN107: `--fn-embed-placement` was given (an explicit `ram` is refused where a plain boot
    /// stands down).
    #[must_use]
    pub fn embed_explicit(&self) -> bool {
        self.fn_embed_placement.is_some()
    }

    /// FN56: the host levers of this boot.
    /// # Errors
    /// An unknown lever name.
    pub fn host_levers(&self) -> anyhow::Result<HostLevers> {
        self.fn_host_levers
            .as_deref()
            .unwrap_or(DEFAULT_FN_HOST_LEVERS)
            .parse()
            .map_err(|e: String| anyhow::anyhow!(e))
    }

    /// FN112: the overlap scheduler (`--fn-overlap off|on`, absent = [`DEFAULT_FN_OVERLAP`] = `on`).
    /// # Errors
    /// A value other than `off` / `on`.
    pub fn overlap(&self) -> anyhow::Result<bool> {
        match self.fn_overlap.as_deref() {
            None => Ok(DEFAULT_FN_OVERLAP),
            Some("off") => Ok(false),
            Some("on") => Ok(true),
            Some(x) => anyhow::bail!("--fn-overlap {x}: expected off | on"),
        }
    }

    /// FN119: the device-step form (`--fn-device-step`, absent = [`super::dstep::DEFAULT_FN_DEVICE_STEP`]).
    /// # Errors
    /// An unknown form.
    pub fn device_step(&self) -> anyhow::Result<super::dstep::DeviceStep> {
        match self.fn_device_step.as_deref() {
            None => Ok(super::dstep::DEFAULT_FN_DEVICE_STEP),
            Some(x) => x.parse().map_err(|e: String| anyhow::anyhow!(e)),
        }
    }

    /// LR-N: coupled drafting (`--spec-couple`, absent = [`DEFAULT_SPEC_COUPLE`] = off).
    /// # Errors
    /// A value other than `off` / `on`.
    pub fn spec_couple(&self) -> anyhow::Result<bool> {
        match self.spec_couple.as_deref() {
            None => Ok(DEFAULT_SPEC_COUPLE),
            Some("off") => Ok(false),
            Some("on") => Ok(true),
            Some(x) => anyhow::bail!("--spec-couple {x}: expected off | on"),
        }
    }

    /// FN117b: the host-hand slices (`--fn-hand`, absent = `none`).
    /// # Errors
    /// An unknown item ([`super::hand::Hand::parse`]).
    pub fn hand(&self) -> anyhow::Result<super::hand::Hand> {
        super::hand::Hand::parse(self.fn_hand.as_deref())
    }

    /// FN57: whether the pick runs on the card this boot.
    #[must_use]
    pub fn device_pick(&self) -> bool {
        self.fn_pick.as_deref().unwrap_or(DEFAULT_FN_PICK) == "device"
    }

    /// FN57: whether the rank's threads are pinned this boot.
    #[must_use]
    pub fn pin_threads(&self) -> bool {
        self.fn_pin.as_deref().unwrap_or(DEFAULT_FN_PIN) == "auto"
    }

    /// FN121: the `--fn-ple-prefetch` form of this boot (absent = `off`, the control).
    /// # Errors
    /// A value other than `off`, `all` or a comma set of `cand`, `keep`, `embed`; `keep` with a
    /// zero period.
    pub fn ple_form(&self) -> anyhow::Result<super::ple121::PleForm> {
        let f = super::ple121::PleForm::parse(
            self.fn_ple_prefetch
                .as_deref()
                .unwrap_or(super::ple121::DEFAULT_FN_PLE_PREFETCH)
                .trim(),
        )?;
        ensure!(
            !f.keep || self.fn_ple_keep_ms > 0,
            "--fn-ple-prefetch keep needs --fn-ple-keep-ms > 0"
        );
        Ok(f)
    }

    /// FN58b / FN58c: the PLE overlap of this boot (`None` = `off`, the control).
    /// # Errors
    /// A value other than `off` / `layer1` / `layer1-l0only` / `layer1-inline` /
    /// `layer1-inline-l0only`.
    pub fn ple_overlap(&self) -> anyhow::Result<Option<PleOverlap>> {
        let t = self
            .fn_ple_overlap
            .as_deref()
            .unwrap_or(DEFAULT_FN_PLE_OVERLAP)
            .trim();
        let (pre, side) = match t {
            "off" => return Ok(None),
            "layer1" => (true, true),
            "layer1-l0only" => (false, true),
            "layer1-inline" => (true, false),
            "layer1-inline-l0only" => (false, false),
            t => anyhow::bail!(
                "--fn-ple-overlap off | layer1 | layer1-l0only | layer1-inline | layer1-inline-l0only, not {t}"
            ),
        };
        Ok(Some(PleOverlap { pre, side }))
    }

    /// FN64d / FN64e: how the lazy runs' graph execs are billed ([`ExecBill`]); not lazy =
    /// [`ExecBill::Unbilled`].
    /// # Errors
    /// A budget without `--fn-graph-split lazy`, or one other than `0` (unbilled) below one exec a
    /// form's head + run (4 MiB) or above what today's unbilled execs take
    /// ([`FN64D_UNBILLED_EXECS`] x 2 MiB).
    pub fn graph_exec_bill(&self) -> anyhow::Result<ExecBill> {
        let lazy = self
            .fn_graph_split
            .as_deref()
            .unwrap_or(DEFAULT_FN_GRAPH_SPLIT)
            .trim()
            == "lazy";
        anyhow::ensure!(
            lazy || (self.fn_graph_exec_budget_mib.is_none() && !self.fn_graph_exec_charge),
            "--fn-graph-exec-budget-mib / --fn-graph-exec-charge need --fn-graph-split lazy"
        );
        if !lazy {
            return Ok(ExecBill::Unbilled);
        }
        if self.graph_form_one()? {
            anyhow::ensure!(
                self.fn_graph_exec_budget_mib.is_none() && !self.fn_graph_exec_charge,
                "--fn-graph-form one bills its own execs (FN98): not with --fn-graph-exec-budget-mib / --fn-graph-exec-charge"
            );
            return Ok(ExecBill::One);
        }
        if self.fn_graph_exec_charge {
            anyhow::ensure!(
                self.fn_graph_exec_budget_mib.is_none(),
                "--fn-graph-exec-charge and --fn-graph-exec-budget-mib are two bills; pick one"
            );
            return Ok(ExecBill::Charge);
        }
        let Some(n) = self.fn_graph_exec_budget_mib else {
            return Ok(DEFAULT_FN_GRAPH_EXEC_BILL);
        };
        if n == 0 {
            return Ok(ExecBill::Unbilled);
        }
        let top = FN64D_UNBILLED_EXECS * GRAPH_EXEC_POOL_MIB;
        anyhow::ensure!(
            (2 * GRAPH_EXEC_POOL_MIB..=top).contains(&n),
            "--fn-graph-exec-budget-mib 0 (unbilled) or {}..={top}, not {n}",
            2 * GRAPH_EXEC_POOL_MIB
        );
        Ok(ExecBill::Budget(n))
    }

    /// FN64d / FN64e: the ledger's terms of the exec bill -- (bytes the plan budget grows by,
    /// the `fn.graph.exec` class bytes); `(0, 0)` unbilled; the charge grows by its own bytes.
    /// # Errors
    /// As [`Self::graph_exec_bill`].
    pub fn graph_exec_ledger(&self) -> anyhow::Result<(u64, u64)> {
        Ok(match self.graph_exec_bill()? {
            ExecBill::Unbilled => (0, 0),
            ExecBill::Budget(n) => (
                (FN64D_UNBILLED_EXECS * GRAPH_EXEC_POOL_MIB) << 20,
                (n / GRAPH_EXEC_POOL_MIB * GRAPH_EXEC_POOL_MIB) << 20,
            ),
            ExecBill::Charge => (FN64E_CHARGE_MIB << 20, FN64E_CHARGE_MIB << 20),
            ExecBill::One => (FN98_GROW_MIB << 20, FN98_CLASS_MIB << 20),
        })
    }

    /// FN98: the NEGATIVE CONTROL of the in-graph gate (`--fn-graph-form one-open`, never a
    /// serving form): the `one` form with its gate nodes never armed (every wait passes at once),
    /// so a step reads expert slots whose copies may still be in flight -- under a forced miss
    /// (`--fn-evict-all`) its bytes must differ where `one`'s equal `lazy`'s.
    #[must_use]
    pub fn graph_form_open(&self) -> bool {
        self.fn_graph_form.as_deref().map(str::trim) == Some("one-open")
    }

    /// FN98: whether this boot runs ONE graph exec per step form (`--fn-graph-form one`).
    /// # Errors
    /// A value other than `one` / `lazy`; `one` without `--fn-graph-split lazy`, or with the lazy
    /// run record (the run table is not consulted).
    pub fn graph_form_one(&self) -> anyhow::Result<bool> {
        let lazy_arm = self.fn_graph_exec_budget_mib.is_some()
            || self.fn_graph_exec_charge
            || self.fn_graph_lazy_record.is_some()
            || self
                .fn_graph_split
                .as_deref()
                .is_some_and(|x| x.trim() != "lazy");
        let t = match self.fn_graph_form.as_deref() {
            Some(x) => x.trim(),
            None if lazy_arm => "lazy",
            None => DEFAULT_FN_GRAPH_FORM,
        };
        let one = match t {
            "one" | "one-open" => true,
            "lazy" => false,
            t => {
                anyhow::bail!("--fn-graph-form one | lazy (| one-open, the gate control), not {t}")
            }
        };
        if one {
            let lazy = self
                .fn_graph_split
                .as_deref()
                .unwrap_or(DEFAULT_FN_GRAPH_SPLIT)
                .trim()
                == "lazy";
            anyhow::ensure!(
                lazy,
                "--fn-graph-form one needs --fn-graph-split lazy (its gate blocks are the lazy quantum)"
            );
            anyhow::ensure!(
                self.fn_graph_lazy_record.is_none(),
                "--fn-graph-form one consults no lazy run table: not with --fn-graph-lazy-record"
            );
        }
        Ok(one)
    }

    /// FN14 / FN64: the graph split of this boot.
    /// # Errors
    /// A value other than `layer` / `step` / `lazy`, a quantum outside 1..=24 or a cache below 1,
    /// or a lazy parameter without `lazy`.
    pub fn graph_split(&self) -> anyhow::Result<GraphSplit> {
        let t = self
            .fn_graph_split
            .as_deref()
            .unwrap_or(DEFAULT_FN_GRAPH_SPLIT)
            .trim();
        let g = match t {
            "layer" => GraphSplit::Layer,
            "step" => GraphSplit::Step,
            "lazy" => {
                let q = self.fn_graph_lazy_q.unwrap_or(DEFAULT_FN_GRAPH_LAZY_Q);
                let cache = self
                    .fn_graph_lazy_cache
                    .unwrap_or(DEFAULT_FN_GRAPH_LAZY_CACHE);
                anyhow::ensure!((1..=24).contains(&q), "--fn-graph-lazy-q 1..=24, not {q}");
                anyhow::ensure!(cache >= 1, "--fn-graph-lazy-cache >= 1");
                // FN64d / FN64e: a billed exec budget / the charge bounds the runs a rank holds
                let cache = match self.graph_exec_bill()?.run_cap() {
                    Some(c) => cache.min(c),
                    None => cache,
                };
                let pre = match self.fn_graph_lazy_pre.as_deref().map(str::trim) {
                    None => DEFAULT_FN_GRAPH_LAZY_PRE,
                    Some("on") => LazyPre::FirstBand,
                    Some("all") => LazyPre::All,
                    Some("off") => LazyPre::Off,
                    Some("recorded") => LazyPre::Recorded,
                    Some(x) => {
                        anyhow::bail!("--fn-graph-lazy-pre on | all | recorded | off, not {x}")
                    }
                };
                GraphSplit::Lazy { q, cache, pre }
            }
            t => anyhow::bail!("--fn-graph-split layer | step | lazy, not {t}"),
        };
        anyhow::ensure!(
            matches!(g, GraphSplit::Lazy { .. })
                || (self.fn_graph_lazy_q.is_none()
                    && self.fn_graph_lazy_cache.is_none()
                    && self.fn_graph_lazy_pre.is_none()
                    && self.fn_graph_lazy_record.is_none()
                    && self.fn_graph_lazy_set.is_none()),
            "--fn-graph-lazy-q / --fn-graph-lazy-cache / --fn-graph-lazy-pre / --fn-graph-lazy-record / --fn-graph-lazy-set need --fn-graph-split lazy"
        );
        anyhow::ensure!(
            self.fn_graph_lazy_set.is_none()
                || matches!(
                    g,
                    GraphSplit::Lazy {
                        pre: LazyPre::Recorded,
                        ..
                    }
                ),
            "--fn-graph-lazy-set needs --fn-graph-lazy-pre recorded"
        );
        Ok(g)
    }

    /// FN30c: whether the mover's hold law is on for this boot.
    #[must_use]
    pub fn mover_hold(&self) -> bool {
        !self.no_fn_mover_hold && (self.fn_mover_hold || DEFAULT_FN_MOVER_HOLD)
    }

    /// FN37: whether the block-output reduce writes the next mix's x' and rr this boot.
    #[must_use]
    pub fn hc_rr_producer(&self) -> bool {
        !self.no_fn_hc_rr_producer && (self.fn_hc_rr_producer || DEFAULT_FN_HC_RR_PRODUCER)
    }
}
/// The prompt-set harness's default of `--fn-mtp`: SPEC 0 (FN13-FN15's byte gates are SPEC-0 gates;
/// `--fn-sched grow` / `--fn-turns 2` refuse MTP).
pub const DEFAULT_HARNESS_MTP: FnMtp = FnMtp::Off;

impl FnFlags {
    /// The `--fn-mtp` law when the flag is absent: [`DEFAULT_SERVE_MTP`] under `--fn-serve`,
    /// [`DEFAULT_HARNESS_MTP`] otherwise.
    #[must_use]
    pub fn default_mtp(&self) -> FnMtp {
        if self.fn_serve.is_some() {
            DEFAULT_SERVE_MTP
        } else {
            DEFAULT_HARNESS_MTP
        }
    }

    /// FN73: `--fail-rank-after-steps R:N` for `rank` (`Some(N)` when this rank is R).
    pub fn fail_after(&self, rank: u32) -> anyhow::Result<Option<u64>> {
        let Some(spec) = self.fail_rank_after_steps.as_deref() else {
            return Ok(None);
        };
        let (r, n) = spec
            .split_once(':')
            .with_context(|| format!("--fail-rank-after-steps {spec}: the form is R:N"))?;
        let (r, n): (u32, u64) = (
            r.parse().context("--fail-rank-after-steps R")?,
            n.parse().context("--fail-rank-after-steps N")?,
        );
        ensure!(
            r < 2 && n > 0,
            "--fail-rank-after-steps {spec}: R is 0|1, N > 0"
        );
        Ok((r == rank).then_some(n))
    }

    /// These flags with `fn_mtp` resolved (`--fn-mtp` if given, else [`Self::default_mtp`]).
    /// Every rank parses the same argv, so the worker form's rank 1 resolves the same law.
    #[must_use]
    pub fn resolved(&self) -> FnFlags {
        let mut f = self.clone();
        f.fn_mtp = self.fn_mtp_arg.unwrap_or_else(|| self.default_mtp());
        if f.fn_ar == "host" {
            f.fn_ar_exact = true;
            f.fn_ar_prologue = "off".into();
            f.fn_moe_ar_push = "off".into();
            f.fn_mixer_ar_push = "off".into();
            f.no_fn_hc_rr_producer = true;
            f.fn_device_step = Some("off".into());
            f.fn_first_step = "pass".into();
            f.fn_hand = Some("none".into());
            f.spec_couple = Some("off".into());
            f.no_fn_mtp_device_chain = true;
            f.fn_mtp = self.fn_mtp_arg.unwrap_or(FnMtp::Off);
            f.fn_mtp_arg = Some(f.fn_mtp);
        }
        f
    }

    /// FN68: the main K/V pages' dtype (`--fn-kv-dtype`).
    #[must_use]
    pub fn kv_dtype(&self) -> saber_mem::kv_tier::KvDtype {
        if self.fn_kv_dtype == "fp8" {
            saber_mem::kv_tier::KvDtype::Fp8
        } else {
            saber_mem::kv_tier::KvDtype::Bf16
        }
    }

    /// FN68d: the MTP layer's K/V page dtype (`--fn-kv-mtp-dtype`; the main dtype under bf16
    /// pages and in the one-process form).
    #[must_use]
    pub fn kv_mtp_dtype(&self) -> saber_mem::kv_tier::KvDtype {
        use saber_mem::kv_tier::KvDtype;
        let main = self.kv_dtype();
        if main == KvDtype::Bf16 || self.fn_one_process {
            return main;
        }
        match self
            .fn_kv_mtp_dtype
            .as_deref()
            .unwrap_or(DEFAULT_FN_KV_MTP_DTYPE)
        {
            "bf16" => KvDtype::Bf16,
            _ => KvDtype::Fp8,
        }
    }

    /// FN68b: the fp8 pages' scale law (`None` under bf16): `--fn-kv-scales` FILE, `unit` (FN68's
    /// 1.0), or `pack` / absent = the pack's calibrated [`super::kvscale::PACK_FILE`] (refused
    /// when the pack has none: an fp8 boot never silently runs FN68's law).
    pub fn kv_scales(&self, pack: &Path) -> anyhow::Result<Option<super::kvscale::KvScales>> {
        use super::kvscale::{KvScales, PACK_FILE};
        if self.kv_dtype() != saber_mem::kv_tier::KvDtype::Fp8 {
            return Ok(None);
        }
        Ok(Some(match self.fn_kv_scales.as_deref() {
            Some("unit") => KvScales::unit(),
            None | Some("pack") => {
                let p = pack.join(PACK_FILE);
                ensure!(
                    p.is_file(),
                    "--fn-kv-dtype fp8: the pack {} has no {PACK_FILE} (the calibrated scale law, \
                     FN68b); pass --fn-kv-scales FILE, or --fn-kv-scales unit for FN68's 1.0",
                    pack.display()
                );
                KvScales::load(&p)?
            }
            Some(f) => KvScales::load(Path::new(f))?,
        }))
    }

    /// FN61d: the decode forms read the W8 trunk records (`--fn-trunk w8` or `w8-head`).
    #[must_use]
    pub fn trunk8(&self) -> bool {
        self.fn_trunk != "bf16"
    }

    /// FN61e: prefill reads the W8 records too and the bf16 trunk records are dropped (`w8`,
    /// `w8-head`; not FN61d's `w8-decode`).
    #[must_use]
    pub fn trunk8_prefill(&self) -> bool {
        matches!(
            self.fn_trunk.as_str(),
            "w8" | "w8-head" | "w8-hc8" | "w8-hc8-head"
        )
    }

    /// FN61d: the head too runs W8A8 (`--fn-trunk w8-head`, FN61c's form; `w8` keeps it bf16).
    #[must_use]
    pub fn trunk8_head(&self) -> bool {
        self.fn_trunk == "w8-head" || self.fn_trunk == "w8-hc8-head"
    }

    /// FN108: the target's HC mixing matrices run at fp8 (`--fn-trunk w8-hc8` / `w8-hc8-head`):
    /// every target mix launches the `_h8` twin of its planned symbol on the pack's `shared/hc8.*`
    /// records and the bf16 mixers are not uploaded (the MTP layer's three stay bf16).
    #[must_use]
    pub fn hc8(&self) -> bool {
        self.fn_trunk == "w8-hc8" || self.fn_trunk == "w8-hc8-head"
    }

    /// The MTP width law of these flags (`None` at SPEC 0).
    ///
    /// # Errors
    /// A `--fn-mtp-table` the parser refuses, or one past the verify cap.
    pub fn mtp_law(&self) -> anyhow::Result<Option<super::worker::verify::MtpLaw>> {
        super::worker::verify::MtpLaw::from_flags(self)
    }

    /// The longest chain any step drafts (0 at SPEC 0): `k`, or the auto table's widest width - 1.
    #[must_use]
    pub fn mtp_k(&self) -> u32 {
        self.mtp_law().ok().flatten().map_or(0, |l| l.kmax())
    }

    /// FN17: a scripted serve schedule (`--fn-sched padgate | mixgate`): the serve plane's own job
    /// path driven by a script, no HTTP.
    #[must_use]
    pub fn scripted(&self) -> bool {
        matches!(self.fn_sched.as_str(), "padgate" | "mixgate")
    }

    /// FN80b-2: the TP-split expert form (`--fn-experts tp2`).
    #[must_use]
    pub fn experts_tp2(&self) -> bool {
        self.fn_experts == "tp2"
    }

    /// FN150: the expert tier's record format (`--fn-experts-fmt`).
    #[must_use]
    pub fn experts_fmt(&self) -> saber_mem::xtier::XtierFormat {
        if self.fn_experts_fmt == "nvfp4" {
            saber_mem::xtier::XtierFormat::Nvfp4E4m3G16
        } else {
            saber_mem::xtier::XtierFormat::Mxfp4E8m0
        }
    }

    /// FN114c: the shared expert split across the EP ranks (`--fn-shared-expert split`).
    #[must_use]
    pub fn shared_split(&self) -> bool {
        self.fn_shared_expert == "split"
    }

    /// FN80d: the MTP expert form this boot runs. `--fn-mtp-experts auto` (the flag's default)
    /// resolves BY THE TARGET'S EXPERT FORM: [`DEFAULT_FN_MTP_EXPERTS`] (`slab4`, FN110b) under
    /// `--fn-experts ep`, [`DEFAULT_FN_MTP_EXPERTS_TP2`] (`slab4` since FN134; FN80d's `fp8`) under
    /// `tp2`; an explicit form is taken as given.
    #[must_use]
    pub fn mtp_experts(&self) -> &str {
        match self.fn_mtp_experts.as_str() {
            "auto" if self.experts_tp2() => DEFAULT_FN_MTP_EXPERTS_TP2,
            "auto" => DEFAULT_FN_MTP_EXPERTS,
            x => x,
        }
    }

    /// FN138: the MTP expert form with its split ([`super::exec::MtpX::resolve`]): `--fn-mtp-split
    /// tp2` needs the slab4 form and `--fn-experts tp2` (refused by name otherwise).
    pub fn mtp_x(&self) -> anyhow::Result<super::exec::MtpX> {
        let x = super::exec::MtpX::resolve(self.mtp_experts(), &self.fn_mtp_split)?;
        ensure!(
            x != super::exec::MtpX::Slab4Tp2 || self.experts_tp2(),
            "--fn-mtp-split tp2 runs the MTP MoE at the target's tp2 dims (needs --fn-experts tp2)"
        );
        Ok(x)
    }

    /// A 27B pack refuses every Flash-Next flag that is not at its default.
    pub fn refuse_on_27b(&self) -> anyhow::Result<()> {
        let d = self.fn_out.is_none()
            && self.fn_experts == super::xtp2::DEFAULT_FN_EXPERTS
            && self.fn_experts_fmt == super::xtp2::DEFAULT_FN_EXPERTS_FMT
            && self.fn_mtp_split == "ep"
            && self.fn_shared_expert == super::shx114::DEFAULT_FN_SHARED_EXPERT
            && self.fn_prompts.is_empty()
            && self.fn_long.is_empty()
            && !self.fn_dump
            && self.fn_prefill_logits_dir.is_none()
            && !self.no_fn_capture
            && !self.fn_capture_verify
            && self.fn_poison_step.is_none()
            && self.fn_sched == "groups"
            && !self.fn_pad_control
            && self.fn_route_trace.is_none()
            && self.fn_route_record.is_none()
            && self.fn_step_trace.is_none()
            && !self.fn_ledger_exact
            && !self.no_fn_ledger_exact
            && !self.fn_verify_slots
            && !self.no_fn_mixed
            && !self.fn_mixed
            && self.fn_graph_split.is_none()
            && self.fn_graph_lazy_q.is_none()
            && self.fn_graph_lazy_cache.is_none()
            && self.fn_graph_lazy_pre.is_none()
            && self.fn_graph_lazy_record.is_none()
            && self.fn_graph_lazy_set.is_none()
            && self.fn_graph_form.is_none()
            && self.fn_serve.is_none()
            && self.fn_chat_template.is_none()
            && self.fn_thinking == "on"
            && self.fn_chunk == 0
            // FN44: absent, not "equal to the default" -- an explicit value is refused too
            && self.fn_max_chunk.is_none()
            && self.fn_wide_when.is_none()
            && self.fn_prefill_yield.is_none()
            && self.fn_kv_lazy.is_none()
            && self.fn_kv_admit_slab_tokens.is_none()
            && self.fn_arena_borrow.is_none()
            && !self.no_fn_arena_borrow
            && !self.fn_borrow_cut
            && !self.no_fn_lend_fill
            && self.fn_poison.is_none()
            && self.fn_control.is_none()
            && self.fn_ple_drive_keepalive_ms.is_none()
            && self.fn_ple_prefetch.is_none()
            && self.fn_ple_keep_ms == super::ple121::DEFAULT_FN_PLE_KEEP_MS
            && self.fn_ple_trace.is_none()
            && self.fn_pinned_pages.is_none()
            && !self.fn_evict_all
            && !self.fn_op_timing
            && self.fn_decode == 0
            && self.fn_turns == 1
            && !self.fn_ar_exact
            && self.fn_tap_layer < 0
            && !self.fn_one_process
            // absent (or the old default `0`): a 27B boot is unchanged by FN16f's serve default
            && matches!(self.fn_mtp_arg, None | Some(FnMtp::Off))
            && self.fn_mtp_table.is_none()
            && !self.no_fn_mtp_capture
            && !self.no_fn_mtp_argmax
            && !self.no_fn_mtp_device_chain
            && !self.fn_mtp_chain_check
            && self.fn_mtp_conf_log.is_none()
            && !self.fn_mtp_chain_selftest
            && !self.fn_mtp_chain_graph
            && !self.fn_mtp_shadow
            && !self.fn_mtp_dynamic
            && self.fn_mtp_depth == DEFAULT_FN_MTP_DEPTH
            && !self.fn_tap_mtp
            && !self.fn_mtp_no_rollback
            && self.fn_verify_class == super::exec::VerifyClass::Width1
            && self.fn_decode_ar == DEFAULT_FN_DECODE_AR
            && !self.fn_row_sha
            && !self.fn_glue_control
            && self.fn_moe_train.to_string() == DEFAULT_FN_MOE_TRAIN
            && self.fn_moe_miss_order == DEFAULT_FN_MOE_MISS_ORDER
            && self.fn_ar_prologue == DEFAULT_FN_AR_PROLOGUE
            && self.fn_ar == DEFAULT_FN_AR
            && self.fn_moe_ar_push == DEFAULT_FN_MOE_AR_PUSH
            && self.fn_mixer_ar_push == DEFAULT_FN_MIXER_AR_PUSH
            && self.fn_first_step == DEFAULT_FN_FIRST_STEP
            && self.fn_fuse == DEFAULT_FN_FUSE
            && self.fn_tiles == DEFAULT_FN_TILES
            && self.fn_sel128 == DEFAULT_FN_SEL128
            && self.fn_hc_down == DEFAULT_FN_HC_DOWN
            && self.fn_k141 == DEFAULT_FN_K141
            && self.fn_route163 == DEFAULT_FN_ROUTE163
            && self.fn_prefill_tiles == DEFAULT_FN_PREFILL_TILES
            && self.fn_glue == DEFAULT_FN_GLUE
            && self.fn_dataflow == DEFAULT_FN_DATAFLOW
            && !self.fn_route_control
            && self.fn_diag_probe.is_none()
            && self.fn_diag_gap_us == 0
            && !self.fn_diag_alias_ram
            && !self.fn_pregate_capture
            && self.fn_draft_head == DEFAULT_FN_DRAFT_HEAD
            && self.fn_draft_head_k == saber_kernels::fn33::DEFAULT_K
            && !self.fn_mover_hold
            && !self.no_fn_mover_hold
            && self.fn_host_levers.is_none()
            && self.fn_overlap.is_none()
            && self.fn_device_step.is_none()
            && self.spec_couple.is_none()
            && self.fn_hand.is_none()
            && self.fn_hand_skew_us == 0
            && self.fn_pick.is_none()
            && self.fn_pin.is_none()
            && self.fn_ple_overlap.is_none()
            && self.fn_mem_valve == "off"
            && self.fn_sdma_lanes == DEFAULT_FN_SDMA_LANES
            && self.fn_mem_valve_max_tail == DEFAULT_FN_MEM_VALVE_MAX_TAIL
            && self.fn_mem_valve_trigger == DEFAULT_FN_MEM_VALVE_TRIGGER
            && self.fn_valve_graphs == DEFAULT_FN_VALVE_GRAPHS
            && self.fn_vram_elastic.is_none()
            && self.fn_vram_elastic_min_slots == DEFAULT_FN_VRAM_ELASTIC_MIN_SLOTS
            && self.fn_vram_elastic_kv_max_gib == DEFAULT_FN_VRAM_ELASTIC_KV_MAX_GIB
            && self.fn_vram_elastic_refill == DEFAULT_FN_VRAM_ELASTIC_REFILL
            && self.fn_vram_elastic_demand == DEFAULT_FN_VRAM_ELASTIC_DEMAND
            && self.fn_plane_dump.is_none()
            && !self.fn_hc_rr_producer
            && !self.no_fn_hc_rr_producer
            && self.fn_mtp_ref.is_none()
            && self.fn_kv_tier.is_none()
            && self.fn_kv_hot_gib.is_none()
            && self.fn_kv_ram_gib.is_none()
            && !self.fn_kv_pages
            && !self.fn_kv_drop_flush
            && !self.no_fn_kv_gather
            && !self.fn_kv_ram_overcommit
            && self.fn_kv_ssd_gib.is_none()
            && self.fn_kv_pcache_gib.is_none()
            && !self.no_fn_kv_pcache
            && !self.fn_kv_pcache_sync
            && !self.fn_kv_pcache_verify
            && self.fn_kv_pcache_min_tokens.is_none()
            && self.warm_prompts.is_none()
            && self.fn_kv_pcache_align == DEFAULT_FN_KV_PCACHE_ALIGN
            && self.fn_prefix_tree.is_none()
            && self.fn_image_cache.is_none()
            && self.fn_prefix_tree_ram_gib == DEFAULT_FN_PREFIX_TREE_RAM_GIB
            && self.fn_prefix_tree_end == DEFAULT_FN_PREFIX_TREE_END
            && self.fn_prefix_tree_demote == "on"
            && self.fn_prefix_tree_min_gap == saber_mem::prefix_tree::DEFAULT_MIN_GAP
            && self.fn_prefix_tree_sys_min == saber_mem::prefix_tree::DEFAULT_SYS_MIN
            && self.fn_prefix_tree_cuts == "today"
            && self.fn_pools_ctx.is_none()
            && self.fn_pools_layout == DEFAULT_FN_POOLS_LAYOUT
            && self.fn_buffers_layout == DEFAULT_FN_BUFFERS_LAYOUT
            && self.fn_embed_placement.is_none()
            && self.fn_qsa == saber_kernels::fn19::QsaMode::Sparse;
        ensure!(d, "--fn-* flags are for a Flash-Next pack (manifest model qwen4_exp); this pack is the 27B's");
        Ok(())
    }
}

/// One request.
#[derive(Debug, Clone)]
pub struct Prompt {
    pub id: String,
    pub ids: Vec<u32>,
    /// FN3's per-step top-20 ids of its greedy continuation (`gen_top_ids.npy`, step `t` = the
    /// distribution that chose generated token `t`): the KL gate's probe ids (FN14). Empty when
    /// the reference has no continuation for this prompt.
    pub ref_top: Vec<Vec<u32>>,
}

/// FN3's `gen_top_ids.npy` (`[steps, 20]` i32), when the prompt directory has one.
fn load_ref_top(dir: &Path) -> anyhow::Result<Vec<Vec<u32>>> {
    let p = dir.join("gen_top_ids.npy");
    if !p.is_file() {
        return Ok(Vec::new());
    }
    let b = npy_payload(&p)?;
    ensure!(b.len() % 80 == 0, "{}: not [steps, 20] i32", p.display());
    Ok(b.chunks_exact(80)
        .map(|r| {
            r.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect())
}

/// The prompt set of a served boot that was given no `--prompt-ids`: one 295-token prompt (the
/// longest of FN3's 40, which is what the served geometry has always been sized from).
fn builtin_prompts() -> Vec<Prompt> {
    vec![Prompt {
        id: "p".into(),
        ids: (0..295u32).map(|i| 1000 + i).collect(),
        ref_top: Vec::new(),
    }]
}

pub(super) fn load_prompts(
    src: &Path,
    only: &[String],
    long: &[u32],
    max_len: u32,
) -> anyhow::Result<Vec<Prompt>> {
    let mut v = Vec::new();
    if src.is_dir() {
        let mut names: Vec<String> = std::fs::read_dir(src)?
            .filter_map(Result::ok)
            .filter(|e| e.path().join("meta.json").is_file())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        for n in names {
            let m: Value =
                serde_json::from_str(&std::fs::read_to_string(src.join(&n).join("meta.json"))?)?;
            let ids = m["token_ids"]
                .as_array()
                .context("token_ids")?
                .iter()
                .map(|x| x.as_u64().map(|y| y as u32).context("id"))
                .collect::<anyhow::Result<Vec<u32>>>()?;
            let ref_top = load_ref_top(&src.join(&n))?;
            v.push(Prompt {
                id: n,
                ids,
                ref_top,
            });
        }
    } else {
        let ids: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(src)?)?;
        v.push(Prompt {
            id: "p".into(),
            ids,
            ref_top: Vec::new(),
        });
    }
    let all: Vec<u32> = v.iter().flat_map(|p| p.ids.clone()).collect();
    if !only.is_empty() {
        v.retain(|p| only.contains(&p.id));
    }
    for &n in long {
        ensure!(!all.is_empty(), "no ids to build a synthetic prompt from");
        v.push(Prompt {
            id: format!("long{n}"),
            ids: (0..n as usize).map(|i| all[i % all.len()]).collect(),
            ref_top: Vec::new(),
        });
    }
    ensure!(!v.is_empty(), "no prompts");
    for p in &v {
        // FN19: past 2,048 tokens a prompt runs in 2,048-row chunks (QSA sparse above 2,050)
        ensure!(
            !p.ids.is_empty() && p.ids.len() as u32 <= max_len,
            "{}: {} tokens (1..={max_len}: FN19 runs a prompt past 2,048 in 2,048-row chunks, \
             QSA sparse above position 2,050)",
            p.id,
            p.ids.len()
        );
    }
    Ok(v)
}

/// `inv_freq = 1 / theta^(2i/64)` at theta 1e7 in f32 as the harness's numpy computes it
/// (`fn6_ref.cos_sin_cache`). DATA, not a formula: an f32 `powf` is not correctly rounded and
/// two libms disagree by one ulp at i = 2 (numpy 0x3ebaf81a, Rust's `powf` 0x3ebaf81b -- 117
/// bf16 table entries below position 2,048 move); the test holds each entry within one ulp of the
/// f64 formula and the whole table to the harness's bytes.
pub const INV_FREQ_1E7_ROT64: [u32; 32] = [
    0x3f80_0000,
    0x3f1a_b32b,
    0x3eba_f81a,
    0x3e61_f835,
    0x3e08_8d77,
    0x3da5_0957,
    0x3d47_763f,
    0x3cf1_1177,
    0x3c91_ad39,
    0x3c30_1052,
    0x3bd4_ca15,
    0x3b80_967d,
    0x3b1b_690d,
    0x3abb_d3ed,
    0x3a63_01e2,
    0x3a09_2e02,
    0x39a5_cb60,
    0x3948_60c2,
    0x38f2_2ce4,
    0x3892_587e,
    0x3830_df52,
    0x37d5_c443,
    0x3781_2dab,
    0x371c_1fc4,
    0x36bc_b0c1,
    0x3664_0cc6,
    0x3609_cf4a,
    0x35a6_8e4c,
    0x3549_4c57,
    0x34f3_499d,
    0x3493_048e,
    0x3431_af44,
];

/// The reference server's rotary cache as the harness builds it (`fn6_ref.cos_sin_cache`): inv_freq in f32
/// ([`INV_FREQ_1E7_ROT64`]), freqs in f32, cos / sin rounded to f32 (from the f64 of the f32
/// argument), then bf16 RNE. Flash-Next's theta is 1e7 (the only one this table carries).
#[must_use]
pub fn rope_table(max_pos: u32) -> Vec<u8> {
    let inv: Vec<f32> = INV_FREQ_1E7_ROT64
        .iter()
        .map(|&b| f32::from_bits(b))
        .collect();
    let rne = |f: f32| -> [u8; 2] {
        let u = f.to_bits();
        (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
    };
    let mut out = Vec::with_capacity(max_pos as usize * 128);
    for p in 0..max_pos {
        let fr: Vec<f32> = inv.iter().map(|&f| p as f32 * f).collect();
        for &f in &fr {
            out.extend_from_slice(&rne(f64::from(f).cos() as f32));
        }
        for &f in &fr {
            out.extend_from_slice(&rne(f64::from(f).sin() as f32));
        }
    }
    out
}

pub(super) fn manifest_ckpt(pack: &Path) -> anyhow::Result<PathBuf> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(pack.join("manifest.json"))?)?;
    Ok(PathBuf::from(
        v.pointer("/checkpoint/dir")
            .and_then(Value::as_str)
            .context("manifest checkpoint.dir")?,
    ))
}

/// The entry from `saber engine` on a Flash-Next pack.
pub fn main(
    pack: &Path,
    tp: u32,
    prebuilt: &Path,
    prompt_src: Option<&Path>,
    hip_lib: &str,
    f: &FnFlags,
) -> anyhow::Result<()> {
    // FN16f: `--fn-mtp` absent -> auto under `--fn-serve`, SPEC 0 in the harness
    if f.fn_ar == "host" {
        ensure!(tp == 2 && !f.fn_one_process, "--fn-ar host requires two worker processes and --tp 2");
        ensure!(
            !matches!(f.fn_mtp_arg, Some(FnMtp::Auto))
                && f.fn_mtp_ref.is_none()
                && !f.fn_mtp_shadow
                && !f.fn_mtp_dynamic
                && !f.fn_mtp_chain_selftest
                && !f.fn_mtp_chain_check
                && !f.fn_mtp_chain_graph
                && f.fn_mtp_conf_log.is_none(),
            "--fn-ar host supports SPEC=0 or a fixed host draft chain (--fn-mtp 1..4), without device-chain diagnostics"
        );
        println!("fnar host: registered host pages, exact BF16; MTP {:?}, host pick exchange and draft chain, device-step and peer producer fusions off", f.fn_mtp_arg.unwrap_or(FnMtp::Off));
    }
    let f = &f.resolved();
    let ckpt = manifest_ckpt(pack)?;
    let cfg = Qwen4ExpConfig::from_config_json(&std::fs::read_to_string(ckpt.join("config.json"))?)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    ensure!(
        cfg.num_layers() == LAYERS,
        "config: {} layers",
        cfg.num_layers()
    );
    println!(
        "fnmodel: {} manifest model qwen4_exp -> the Flash-Next path (config: {} layers, hidden {}, {} experts top-{}, \
         PLE layer {:?}, checkpoint {}); the 27B path is not entered",
        pack.display(),
        cfg.num_layers(),
        H,
        cfg.moe.num_experts,
        cfg.moe.top_k,
        cfg.ple.layer_indices(),
        ckpt.display()
    );
    // FN17: the pad-law gate's long prompt runs in chunks of MAX_ROWS (the serve plane's cut)
    // FN19: every prompt up to the rope table (past 2,048: 2,048-row chunks)
    let max_len = ROPE_POS_MAX;
    // `--prompt-ids` absent: only a served boot may go without (the harness runs the prompts; a
    // server sizes its pools from `--fn-max-ctx`). The built-in set keeps the longest FN3 prompt's
    // length (295 tokens), the one number the served boot's geometry reads from the prompts.
    let prompts = match prompt_src {
        Some(src) => load_prompts(src, &f.fn_prompts, &f.fn_long, max_len)?,
        None => {
            ensure!(
                f.fn_serve.is_some() && !f.fn_preflight,
                "--prompt-ids is required unless this is a --fn-serve boot"
            );
            builtin_prompts()
        }
    };
    let out = f.fn_out.clone().context("--fn-out DIR")?;
    if let Some(dir) = &f.fn_prefill_logits_dir {
        ensure!(
            f.fn_serve.is_none() && !f.fn_one_process && f.fn_conc == [1] && f.fn_repeat == 1 && f.fn_decode == 0,
            "--fn-prefill-logits-dir requires offline TP=2 workers, --fn-conc 1, --fn-repeat 1 and --fn-decode 0"
        );
        std::fs::create_dir_all(dir)?;
    }
    std::fs::create_dir_all(&out)?;
    // FN54: the host keep-awake arm lives in rank 0's process for the whole boot
    let _keepalive = if f.fn_control.as_deref() == Some("host-keepalive") && f.fn_rank == 0 {
        let (k, line) = super::keepalive::start(super::keepalive::KEEPALIVE_CPU)?;
        println!("{line}");
        Some(k)
    } else {
        None
    };
    // FN90a: the PLE drive's keep-awake arm (rank 0's process; the drive is the ranks' one)
    // FN121 (`--fn-ple-prefetch keep`): the period `--fn-ple-keep-ms` (an explicit
    // `--fn-ple-drive-keepalive-ms` still wins) and one reader per block device the PLE files and
    // the embedding file sit on (both ranks read the same files: rank 0's readers cover both)
    let form121 = f.ple_form()?;
    let keep_ms = match (f.fn_ple_drive_keepalive_ms, form121.keep) {
        (Some(ms), _) => ms,
        (None, true) => f.fn_ple_keep_ms,
        (None, false) => DEFAULT_FN_PLE_DRIVE_KEEPALIVE_MS,
    };
    let _drive_keepalive = match keep_ms {
        ms if ms > 0 && f.fn_rank == 0 => {
            let idx = PleIndex::from_json(
                &std::fs::read_to_string(pack.join("ple/ple_index.json"))?,
                PleSourceKind::CkptBf16,
                PleGate::Default,
                None,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let p = idx.files.first().context("the PLE index names no file")?;
            let mut ks = Vec::new();
            if form121.keep {
                let recs = super::store::PackRecs::open(pack, &ckpt)?;
                let (emb, _, _) = recs.extent(0, "model.language_model.embed_tokens.weight")?;
                let mut all: Vec<std::path::PathBuf> = idx.files.clone();
                all.push(emb);
                let (devs, line) = super::keepalive::drive_cover(&all)?;
                println!("{line}");
                for d in devs {
                    let (k, line) = super::keepalive::start_drive(&d, ms)?;
                    println!("{line}");
                    ks.push(k);
                }
            } else {
                let (k, line) = super::keepalive::start_drive(p, ms)?;
                println!("{line}");
                ks.push(k);
            }
            Some(ks)
        }
        _ => None,
    };
    if f.fn_preflight {
        return preflight(
            pack,
            &ckpt,
            &cfg,
            prebuilt,
            &prompts,
            prompt_src.context("--fn-preflight needs --prompt-ids")?,
            f,
        );
    }
    ensure!(
        tp == 2 || !f.xtier.xtier_stage,
        "--xtier-stage (FN43's ring) stages the TP=2 expert tier; TP=1 has none"
    );
    // FN164: the default `on` is inert on a single-rank boot (no mixer-out reduce to fuse); the
    // `b1` arm is refused there by name
    ensure!(
        tp == 2 || mixer_ar_push(f) != super::dev::MixPush::B1,
        "--fn-mixer-ar-push b1 (FN164) fuses the TP=2 mixer-out all-reduce; a single-rank boot has none"
    );
    // the ledger plans (and refuses) on the host, before a HIP call
    let boot = if tp == 2 {
        Some(plan_boot(pack, prebuilt, &prompts, f)?)
    } else {
        None
    };
    // the worker form opens its HIP runtime inside each rank's process
    let hip0 = if tp == 2 && !f.fn_one_process {
        None
    } else {
        Some(crate::open_hip(hip_lib)?)
    };
    let r = match (tp, boot) {
        (1, _) => gate_a(
            hip0.as_ref().context("hip")?,
            pack,
            &ckpt,
            prebuilt,
            &prompts,
            &out,
            f,
        ),
        (2, Some(boot)) if !f.fn_one_process => {
            drop(hip0);
            return super::worker::engine_worker(
                hip_lib, boot, pack, &ckpt, &cfg, prebuilt, &prompts, &out, f,
            );
        }
        (2, Some(boot)) => {
            ensure!(
                f.fn_decode == 0 && f.fn_tap_layer < 0 && !f.fn_ar_exact,
                "the one-process form (--fn-one-process) is FN12's prefill control only"
            );
            let hip1 = crate::open_hip(hip_lib)?;
            let r = engine_tp2(
                [hip0.as_ref().context("hip")?, &hip1],
                boot,
                pack,
                &ckpt,
                &cfg,
                prebuilt,
                &prompts,
                &out,
                f,
            );
            drop(hip1);
            r
        }
        _ => bail!("Flash-Next: --tp 1 (gate A) or 2 (the engine)"),
    };
    drop(hip0);
    r
}

/// A `.npy` (v1/v2, little-endian, C order) payload.
pub(super) fn npy_payload(path: &Path) -> anyhow::Result<Vec<u8>> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    ensure!(
        b.len() > 10 && &b[..6] == b"\x93NUMPY",
        "{}: not .npy",
        path.display()
    );
    let (hl, at) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    Ok(b[at + hl..].to_vec())
}

/// The host pre-flight: every input the card would get, checked against FN3 where FN3 has it.
fn preflight(
    pack: &Path,
    ckpt: &Path,
    cfg: &Qwen4ExpConfig,
    prebuilt: &Path,
    prompts: &[Prompt],
    src: &Path,
    f: &FnFlags,
) -> anyhow::Result<()> {
    let ck = Ckpt::open(ckpt)?;
    let recs = PackRecs::open(pack, ckpt)?;
    let pre = format!("{P}layers.1.ple.ple_embedding.");
    let ph = PleHash::from_buffers(
        &cfg.ple,
        cfg.eos_token_id,
        &ck.raw(&format!("{pre}layer_multipliers"))?,
        &ck.raw(&format!("{pre}ngram_heads_vocab_sizes"))?,
        &ck.raw(&format!("{pre}ngram_heads_offsets"))?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let (mut ple_ok, mut emb_ok, mut n) = (0, 0, 0);
    for p in prompts.iter().filter(|p| src.join(&p.id).is_dir()) {
        n += 1;
        let ids = ph.ids(PleContext::start(cfg.eos_token_id), &p.ids);
        let want: Vec<u64> = npy_payload(&src.join(&p.id).join("ple_ids.npy"))?
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8")))
            .collect();
        ple_ok += usize::from(ids == want);
        let e = embed_rows(&recs, &p.ids)?;
        let want: Vec<u8> = npy_payload(&src.join(&p.id).join("embed.npy"))?
            .chunks_exact(4)
            .flat_map(|c| {
                let u = u32::from_le_bytes(c.try_into().expect("4"));
                assert_eq!(u & 0xffff, 0, "FN3's embed is not a bf16 row");
                ((u >> 16) as u16).to_le_bytes()
            })
            .collect();
        emb_ok += usize::from(e == want);
    }
    println!("preflight: PLE ids == FN3 ple_ids {ple_ok}/{n}; embed rows (pack identity records) == FN3 embed {emb_ok}/{n}");
    let decls = Decls::load(prebuilt)?;
    let rules = FlashNextRules::prod()?;
    for (tp, m) in [(1u32, 295u32), (2, 295), (2, 2048)] {
        let dr = plan::dry_run(&decls, &rules, tp, m)?;
        println!(
            "preflight: TP={tp} m {m}: {} launches/chunk (layers {} + post {}), arena {:.1} MiB",
            dr.per_layer.iter().sum::<u32>() + dr.post,
            dr.per_layer.iter().sum::<u32>(),
            dr.post,
            dr.sizes.total() as f64 / 1048576.0
        );
    }
    let b = plan_boot(pack, prebuilt, prompts, f)?;
    for (r, x) in b.budgets.iter().enumerate() {
        println!("{}", x.boot_line());
        println!("{}", b.laws[r].boot_line(r as u32));
    }
    // FN100: the fn.buffers census (host)
    // FN134: at the boot's record stride (a half record under `--fn-experts tp2`)
    for l in buffers_census(f, prompts, super::xtp2::XGeom::of(f, pack)?.stride)?.0 {
        println!("{l}");
    }
    ensure!(ple_ok == n && emb_ok == n, "pre-flight FAILED");
    println!("PREFLIGHT PASS");
    Ok(())
}

// ============================================================================ shared pieces

/// A sequence's per-layer state on one rank (zeroed at the sequence's start).
pub struct Seq {
    pub st: Vec<LayerState>,
    pub bufs: Vec<DevicePtr>,
}

impl Seq {
    pub fn new(hip: &HipThread, dims: &Dims, max_ctx: u32, state_f32: bool) -> anyhow::Result<Seq> {
        let sb = if state_f32 { 4 } else { 2 };
        let nb = u64::from(max_ctx.div_ceil(16) + 1);
        let mut bufs = Vec::new();
        let take = |n: u64, bufs: &mut Vec<DevicePtr>| -> anyhow::Result<u64> {
            let d = hip.malloc(n.max(4))?;
            let a = d.addr().as_u64();
            bufs.push(d);
            Ok(a)
        };
        let mut st = Vec::new();
        for l in 0..LAYERS {
            let mut s = LayerState::default();
            if plan::kind(l) == saber_model::qwen4_exp::LayerKind::Qsa {
                s.kv = take(nb * dims.nkv * 16 * 512 * 2, &mut bufs)?;
                s.ring = take(RING * 128 * 2, &mut bufs)?;
                s.ringp = take(RING * 3 * 8, &mut bufs)?;
                s.comp = take(nb * 4 * 128 * 2, &mut bufs)?;
            } else {
                s.cst = take(dims.conv * 3 * 2, &mut bufs)?;
                s.h = [
                    take(dims.hv * 128 * 128 * sb, &mut bufs)?,
                    take(dims.hv * 128 * 128 * sb, &mut bufs)?,
                ];
            }
            if l == PLE_LAYER {
                s.ple_st = [
                    take(PLE_HIST * W * 2, &mut bufs)?,
                    take(PLE_HIST * W * 2, &mut bufs)?,
                ];
            }
            st.push(s);
        }
        Ok(Seq { st, bufs })
    }
    /// The start of a sequence: every state byte zero.
    pub fn zero(&self, hip: &HipThread, stream: &Stream) -> anyhow::Result<()> {
        for b in &self.bufs {
            hip.memset(b, 0, stream)?;
        }
        Ok(())
    }
}

/// Upload a chunk's constant tables at `off` of the rank's consts buffer.
pub(super) fn consts(
    dev: &RankDev<'_>,
    off: u64,
    m: u32,
    p0: u32,
    chunk: u32,
) -> anyhow::Result<ChunkConsts> {
    let nb = (p0 + m).div_ceil(16) + 1;
    let (b, offs) = ChunkConsts::host(m, p0, chunk, nb);
    dev.hip
        .memcpy_h2d_at_blocking(&dev.consts, off, &b, &dev.stream)?;
    Ok(ChunkConsts::at(
        dev.consts.addr().as_u64() + off,
        &offs,
        m,
        p0,
        chunk,
        nb,
        dev.cs.addr().as_u64(),
    ))
}

pub(super) fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

fn argmax(v: &[f32]) -> (usize, f32, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best.1 {
            second = best.1;
            best = (i, x);
        } else if x > second {
            second = x;
        }
    }
    (best.0, best.1, second)
}

/// A minimal `.npy` (v1.0) for i32 / f32 captures.
fn npy(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> anyhow::Result<()> {
    let sh = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut hdr = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {sh}, }}");
    while (10 + hdr.len() + 1) % 64 != 0 {
        hdr.push(' ');
    }
    hdr.push('\n');
    let mut b = Vec::with_capacity(10 + hdr.len() + data.len());
    b.extend_from_slice(b"\x93NUMPY\x01\x00");
    b.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
    b.extend_from_slice(hdr.as_bytes());
    b.extend_from_slice(data);
    std::fs::write(path, b)?;
    Ok(())
}

/// One request's capture (host), written as `flashnext-capture/1`.
#[derive(Default)]
pub(super) struct Cap {
    pub(super) t: usize,
    pub(super) resid: Vec<u8>,
    pub(super) att: Vec<u8>,
    pub(super) mlp: Vec<u8>,
    pub(super) ids: Vec<u8>,
    pub(super) rlog: Vec<u8>,
    pub(super) ple: Vec<u8>,
    pub(super) fin: Vec<u8>,
    pub(super) logits: Vec<f32>,
    /// FN19: the captured positions of a prompt past 2,048 tokens (the reference's `long_rows`);
    /// `None` = every position. The per-position kinds hold `t = keep.len()` rows.
    pub(super) keep: Option<Vec<usize>>,
    /// FN19: `[12 QSA layers, prompt - 2,051 sparse rows, 512]` i32, the selections of the rows at
    /// positions >= 2,051 (the reference's `idx_sel`); empty for a prompt of <= 2,051 tokens.
    pub(super) sel: Vec<u8>,
    pub(super) sel_rows: usize,
    /// FN19: the sparse rows' indexer queries `[12, sel_rows, 4, 128]` bf16 and every QSA layer's
    /// compressed keys `[12, prompt / 4, 128]` bf16 (the device's own selection inputs).
    pub(super) iq: Vec<u8>,
    pub(super) comp: Vec<Vec<u8>>,
}

/// FN19: the reference's captured positions of a prompt past 2,048 tokens
/// (`flashnext_ref.py long_rows`): every 512th row below 2,048, the dense/sparse boundary rows
/// 2,047..2,052, every 64th row above, the last 64.
#[must_use]
pub fn long_rows(t: usize) -> Vec<usize> {
    let mut r: std::collections::BTreeSet<usize> =
        (0..t.min(2048)).filter(|p| p % 512 == 511).collect();
    r.extend(2047..t.min(2053));
    r.extend((2048..t).filter(|p| p % 64 == 63));
    r.extend(t.saturating_sub(64)..t);
    r.into_iter().collect()
}

impl Cap {
    pub(super) fn new(t: usize) -> Cap {
        let l = LAYERS as usize;
        Cap {
            t,
            resid: vec![0; l * t * W as usize * 2],
            att: vec![0; l * t * H as usize * 2],
            mlp: vec![0; l * t * H as usize * 2],
            ids: vec![0; l * t * 40],
            rlog: vec![0; l * t * 1024],
            ple: vec![0; t * W as usize * 2],
            fin: vec![0; t * H as usize * 2],
            logits: Vec::new(),
            keep: None,
            sel: Vec::new(),
            sel_rows: 0,
            iq: Vec::new(),
            comp: Vec::new(),
        }
    }

    /// FN19: a prompt's capture -- every position up to 2,048 tokens ([`Cap::new`]); past it the
    /// reference's row subset (no mlp_out / router logits) + the sparse rows' selections.
    pub(super) fn for_prompt(n: usize) -> Cap {
        if n <= 2048 {
            return Cap::new(n);
        }
        let keep = long_rows(n);
        let t = keep.len();
        let l = LAYERS as usize;
        let sel_rows = n.saturating_sub(2051);
        Cap {
            t,
            resid: vec![0; l * t * W as usize * 2],
            att: vec![0; l * t * H as usize * 2],
            mlp: Vec::new(),
            ids: vec![0; l * t * 40],
            rlog: Vec::new(),
            ple: vec![0; t * W as usize * 2],
            fin: vec![0; t * H as usize * 2],
            logits: Vec::new(),
            keep: Some(keep),
            sel: vec![0; 12 * sel_rows * 512 * 4],
            sel_rows,
            iq: vec![0; 12 * sel_rows * 512 * 2],
            comp: vec![Vec::new(); 12],
        }
    }

    /// FN19: QSA layer `l`'s compressed keys of the whole prompt (`[groups, 128]` bf16).
    pub(super) fn put_comp(&mut self, l: usize, bytes: &[u8]) {
        if l % 4 == 3 && l / 4 < self.comp.len() {
            self.comp[l / 4] = bytes.to_vec();
        }
    }

    /// Put the rows of absolute positions `[a0, a0 + m)` (row width `w` bytes) of layer `l` into
    /// `buf`: all of them ([`Cap::put`]), or only the kept positions.
    pub(super) fn put_rows(
        buf: &mut [u8],
        t: usize,
        keep: Option<&[usize]>,
        l: usize,
        a0: usize,
        w: usize,
        rows: &[u8],
    ) {
        let Some(k) = keep else {
            return Cap::put(buf, t, l, a0, w, rows);
        };
        let m = rows.len() / w;
        for (j, &p) in k.iter().enumerate() {
            if p >= a0 && p < a0 + m {
                let at = (l * t + j) * w;
                buf[at..at + w].copy_from_slice(&rows[(p - a0) * w..(p - a0 + 1) * w]);
            }
        }
    }

    /// FN19: layer `l`'s selection `[m, 512]` i32 of a chunk at `p0`: its rows at positions
    /// >= 2,051 into the sparse-row buffer (a GDN layer or a dense chunk contributes nothing).
    pub(super) fn put_sel(&mut self, l: usize, p0: usize, sel: &[u8], iq: &[u8]) {
        if self.sel_rows == 0 || l % 4 != 3 {
            return;
        }
        let q = l / 4;
        let row = 512 * 4;
        let qrow = 512 * 2;
        let m = sel.len() / row;
        for r in 0..m {
            let p = p0 + r;
            if p >= 2051 && p - 2051 < self.sel_rows {
                let at = (q * self.sel_rows + (p - 2051)) * row;
                self.sel[at..at + row].copy_from_slice(&sel[r * row..(r + 1) * row]);
                let aq = (q * self.sel_rows + (p - 2051)) * qrow;
                self.iq[aq..aq + qrow].copy_from_slice(&iq[r * qrow..(r + 1) * qrow]);
            }
        }
    }
    /// Put rows `[r0, r0 + m)` of layer `l`'s `kind` (row width `w` bytes).
    pub(super) fn put(buf: &mut [u8], t: usize, l: usize, r0: usize, w: usize, rows: &[u8]) {
        let at = (l * t + r0) * w;
        buf[at..at + rows.len()].copy_from_slice(rows);
    }
    /// FN13's form gate: sha256 per layer of resid / attn_out / mlp_out / route ids (layer-major
    /// buffers: one contiguous slice a layer), the PLE output, the final hidden and the logits.
    pub(super) fn hashes(&self) -> Value {
        let t = self.t;
        let per = |b: &[u8], w: usize| -> Vec<String> {
            // FN19: a long prompt's capture carries no mlp_out (an empty buffer: no hashes)
            if b.is_empty() {
                return Vec::new();
            }
            (0..LAYERS as usize)
                .map(|l| sha256_hex(&b[l * t * w..(l + 1) * t * w]))
                .collect()
        };
        let lg: Vec<u8> = self.logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        json!({"resid": per(&self.resid, W as usize * 2), "attn_out": per(&self.att, H as usize * 2),
               "mlp_out": per(&self.mlp, H as usize * 2), "router_ids": per(&self.ids, 40),
               "ple_out": sha256_hex(&self.ple), "final_hidden": sha256_hex(&self.fin), "logits": sha256_hex(&lg)})
    }
    pub(super) fn write(&self, dir: &Path, p: &Prompt, producer: &str) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        let t = self.t;
        let l = LAYERS as usize;
        let mut kinds = serde_json::Map::new();
        let mut bin = |name: &str,
                       data: &[u8],
                       shape: Vec<usize>,
                       axes: Vec<&str>|
         -> anyhow::Result<()> {
            std::fs::write(dir.join(format!("{name}.bin")), data)?;
            kinds.insert(name.into(), json!({"file": format!("{name}.bin"), "dtype": "bf16", "shape": shape, "axes": axes}));
            Ok(())
        };
        bin(
            "resid",
            &self.resid,
            vec![l, t, W as usize],
            vec!["layer", "pos", "hc_hidden"],
        )?;
        bin(
            "attn_out",
            &self.att,
            vec![l, t, H as usize],
            vec!["layer", "pos", "hidden"],
        )?;
        if !self.mlp.is_empty() {
            bin(
                "mlp_out",
                &self.mlp,
                vec![l, t, H as usize],
                vec!["layer", "pos", "hidden"],
            )?;
        }
        if !self.rlog.is_empty() {
            bin(
                "router_logits",
                &self.rlog,
                vec![l, t, 512],
                vec!["layer", "pos", "expert"],
            )?;
        }
        bin(
            "ple_out",
            &self.ple,
            vec![t, W as usize],
            vec!["pos", "hc_hidden"],
        )?;
        bin(
            "final_hidden",
            &self.fin,
            vec![t, H as usize],
            vec!["pos", "hidden"],
        )?;
        npy(&dir.join("router_ids.npy"), "<i4", &[l, t, 10], &self.ids)?;
        kinds.insert("router_ids".into(), json!({"file": "router_ids.npy", "dtype": "i32", "shape": [l, t, 10], "axes": ["layer", "pos", "k"]}));
        let lg: Vec<u8> = self.logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        npy(&dir.join("logits_last.npy"), "<f4", &[VOCAB as usize], &lg)?;
        kinds.insert(
            "logits_last".into(),
            json!({"file": "logits_last.npy", "dtype": "f32", "shape": [VOCAB], "axes": ["vocab"]}),
        );
        if self.sel_rows > 0 {
            npy(
                &dir.join("idx_sel.npy"),
                "<i4",
                &[12, self.sel_rows, 512],
                &self.sel,
            )?;
            kinds.insert(
                "idx_sel".into(),
                json!({"file": "idx_sel.npy", "dtype": "i32",
                "shape": [12, self.sel_rows, 512], "axes": ["qsa_layer", "sparse_row", "k"]}),
            );
            let rows: Vec<u8> = (2051..2051 + self.sel_rows as i32)
                .flat_map(i32::to_le_bytes)
                .collect();
            npy(&dir.join("idx_rows.npy"), "<i4", &[self.sel_rows], &rows)?;
            kinds.insert(
                "idx_rows".into(),
                json!({"file": "idx_rows.npy", "dtype": "i32",
                "shape": [self.sel_rows], "axes": ["sparse_row"]}),
            );
            std::fs::write(dir.join("idx_q.bin"), &self.iq)?;
            kinds.insert(
                "idx_q".into(),
                json!({"file": "idx_q.bin", "dtype": "bf16", "shape": [12, self.sel_rows, 4, 128],
                "axes": ["qsa_layer", "sparse_row", "head", "dim"]}),
            );
            let g = self.comp.iter().map(Vec::len).max().unwrap_or(0) / 256;
            let mut ck = vec![0u8; 12 * g * 256];
            for (q, c) in self.comp.iter().enumerate() {
                ck[q * g * 256..q * g * 256 + c.len()].copy_from_slice(c);
            }
            std::fs::write(dir.join("idx_comp.bin"), &ck)?;
            kinds.insert(
                "idx_comp".into(),
                json!({"file": "idx_comp.bin", "dtype": "bf16", "shape": [12, g, 128],
                "axes": ["qsa_layer", "group", "dim"]}),
            );
        }
        let positions: Vec<usize> = self.keep.clone().unwrap_or_else(|| (0..t).collect());
        let meta = json!({
            "format": "flashnext-capture/1", "prompt_id": p.id, "token_ids": p.ids,
            "generated_ids": [], "positions": positions, "kinds": kinds,
            "producer": producer,
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
        Ok(())
    }
}

// ============================================================================ gate A (TP=1)

fn carry_bytes(sizes: &ArenaSizes) -> u64 {
    sizes.carry.div_ceil(ALIGN) * ALIGN
}

/// Gate A: the Rust executor at TP=1, layer-major like the harness's chain mode, every per-layer
/// output hashed (resid materialized, attn_out, mlp_out, route ids), then the final mixer + head.
#[allow(clippy::too_many_lines)]
fn gate_a(
    hip: &HipThread,
    pack: &Path,
    ckpt: &Path,
    prebuilt: &Path,
    prompts: &[Prompt],
    out: &Path,
    f: &FnFlags,
) -> anyhow::Result<()> {
    hip.set_device(0)?;
    let decls = Decls::load(prebuilt)?;
    let rules = FlashNextRules::prod()?;
    let lc = LoadCounters::new();
    let ps = PluginSet::load_named(hip, prebuilt, &PLUGINS, &lc)?;
    let table = ps.functions(hip)?;
    let ck = Ckpt::open(ckpt)?;
    let recs = PackRecs::open(pack, ckpt)?;
    let m_max = prompts
        .iter()
        .map(|p| p.ids.len() as u32)
        .max()
        .unwrap_or(1);
    // P27-9d: every position-indexed table covers the positions the run reaches; gate A's rope
    // table is ROPE_POS rows, so a longer prompt is refused here, by name, not read past.
    anyhow::ensure!(
        m_max <= ROPE_POS,
        "gate A's rope table holds {ROPE_POS} positions and a prompt is {m_max} tokens: the rope \
         kernels index it by position with no bound (P27-9d)"
    );
    let dr = plan::dry_run_cap(&decls, &rules, 1, m_max, true)?;
    let dims = Dims::new(1, 0);
    let mut dev = RankDev::new(hip, 0, dr.sizes, &rope_table(ROPE_POS))?;
    dev.poison = f.fn_poison;
    let mut em = Emitter::new(&decls, &rules, dims, Arena::carve(dev.base(), dr.sizes));
    em.qsa = f.fn_qsa;
    em.vclass = f.fn_verify_class;
    // FN30: layer-major prefill only (the seven-launch train); no sync region at TP=1
    em.moe_train = f.fn_moe_train;
    em.route_form = route_form(f);
    em.glue_v1 = f.fn_glue_control;
    em.kv8 = f.kv_dtype() == saber_mem::kv_tier::KvDtype::Fp8;
    em.kv8_mtp = em.kv8;
    ensure!(
        !kv_desc_on(f),
        "--fn-kv-dtype fp8's scale law (FN68b) runs in the worker form at TP=2; pass \
         --fn-kv-scales unit here (FN68's 1.0)"
    );
    if f.fn_control.as_deref() == Some("swap-arg") {
        em.ctl = Ctl::SwapArg;
    }
    println!(
        "gatea: TP=1 layer-major over {} prompts, depth {}, arena {:.1} MiB (scratch {:.1} / zero {:.2} / ones {:.2} / carry {:.1}), poison {:?}, control {:?}",
        prompts.len(),
        f.fn_depth,
        dr.sizes.total() as f64 / 1048576.0,
        dr.sizes.scratch as f64 / 1048576.0,
        dr.sizes.zero as f64 / 1048576.0,
        dr.sizes.ones as f64 / 1048576.0,
        dr.sizes.carry as f64 / 1048576.0,
        f.fn_poison,
        f.fn_control
    );
    // per prompt: embed, PLE rows, consts, sequence state, carry
    let eos = 248_044u32;
    let cfg = Qwen4ExpConfig::from_config_json(&std::fs::read_to_string(ckpt.join("config.json"))?)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let pre = format!("{P}layers.1.ple.ple_embedding.");
    let ph = PleHash::from_buffers(
        &cfg.ple,
        cfg.eos_token_id,
        &ck.raw(&format!("{pre}layer_multipliers"))?,
        &ck.raw(&format!("{pre}ngram_heads_vocab_sizes"))?,
        &ck.raw(&format!("{pre}ngram_heads_offsets"))?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let _ = eos;
    let idx = PleIndex::from_json(
        &std::fs::read_to_string(pack.join("ple/ple_index.json"))?,
        PleSourceKind::CkptBf16,
        PleGate::Default,
        None,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ple =
        PleTier::new(idx, OpenMode::Buffered, 32, 1 << 16).map_err(|e| anyhow::anyhow!("{e}"))?;
    struct Pp {
        m: u32,
        _embed: DevicePtr,
        rows: DevicePtr,
        cc: ChunkConsts,
        seq: Seq,
        carry_mem: DevicePtr,
        carry: [Bump; 2],
        inp: LayerIn,
        hashes: Vec<Value>,
    }
    let mut pp = Vec::new();
    let cb = carry_bytes(&dr.sizes);
    for (i, p) in prompts.iter().enumerate() {
        let m = p.ids.len() as u32;
        let e = embed_rows(&recs, &p.ids)?;
        let embed = hip.malloc(e.len() as u64)?;
        hip.memcpy_h2d_blocking(&embed, &e, &dev.stream)?;
        let ids = ph.ids(PleContext::start(cfg.eos_token_id), &p.ids);
        let mut staged = vec![0u8; ids.len() * 320];
        ple.gather(&ids, &mut staged)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let rows = hip.malloc(staged.len() as u64)?;
        hip.memcpy_h2d_blocking(&rows, &staged, &dev.stream)?;
        let cc = consts(&dev, i as u64 * (64 << 10), m, 0, 0)?;
        let seq = Seq::new(hip, &dims, m, false)?;
        seq.zero(hip, &dev.stream)?;
        let carry_mem = hip.malloc(2 * cb)?;
        let base = carry_mem.addr().as_u64();
        let carry = [Bump::new(base, cb), Bump::new(base + cb, cb)];
        let ea = embed.addr().as_u64();
        pp.push(Pp {
            m,
            _embed: embed,
            rows,
            cc,
            seq,
            carry_mem,
            carry,
            inp: LayerIn::Embed(ea),
            hashes: Vec::new(),
        });
    }
    hip.sync_stream(&dev.stream)?;
    let mut launch_bad = Vec::new();
    let t_all = Instant::now();
    for l in 0..f.fn_depth.min(LAYERS) {
        let t0 = Instant::now();
        let mut owned = Owned::new();
        let mut lw = model::dense_layer(
            &Src::Ckpt(&ck),
            &mut owned,
            hip,
            &dev.stream,
            l,
            false,
            false,
        )?;
        (lw.xgu, lw.xdn) = model::experts_tp1(&recs, &mut owned, hip, &dev.stream, l)?;
        let load_s = t0.elapsed().as_secs_f64();
        for q in &mut pp {
            let m = u64::from(q.m);
            em.arena.carry = q.carry;
            let lws = [lw];
            let mut hook = Resident(&lws);
            let o: LayerOut = emit_layer(
                std::slice::from_mut(&mut em),
                &lws,
                &[q.seq.st[l as usize]],
                &[q.inp],
                &[q.rows.addr().as_u64()],
                &[q.cc],
                &mut None,
                &mut hook,
            )?
            .remove(0);
            let want = plan::declared_form_qsa(
                &rules,
                l,
                1,
                q.m,
                saber_model::qwen4_exp::layer_graph::Form::Prefill,
                em.qsa_sparse(&q.cc),
                em.moe_train,
            );
            if o.launches != want {
                launch_bad.push(format!(
                    "L{l} m{}: {} launches vs declared {want}",
                    q.m, o.launches
                ));
            }
            let rs = em.arena.take(m * W * 2, super::kern::Seed::None)?;
            em.materialize(&o.pend, m, rs)?;
            dev.exec(&table, &mut em)?;
            let g = |a: u64, n: u64, base: &DevicePtr| -> anyhow::Result<String> {
                Ok(sha256_hex(&d2h_in(&dev, base, a, n as usize)?))
            };
            let arena = &dev.arena;
            q.hashes.push(json!({
                "layer": l,
                "resid": g(rs, m * W * 2, arena)?,
                "att": g(o.att, m * H * 2, arena)?,
                "mo": g(o.pend.mo, m * H * 2, &q.carry_mem)?,
                "ids": g(o.ids, m * 40, arena)?,
                "launches": o.launches,
            }));
            q.carry = em.arena.carry;
            q.inp = LayerIn::Pending(o.pend);
        }
        println!(
            "gatea: layer {l} ({:?}) weights {load_s:.1} s, {} prompts, {:.1} s total",
            plan::kind(l),
            pp.len(),
            t_all.elapsed().as_secs_f64()
        );
        drop(owned);
    }
    let mut res = serde_json::Map::new();
    if f.fn_depth >= LAYERS {
        let mut owned = Owned::new();
        let fm = model::final_mixer(&Src::Ckpt(&ck), &mut owned, hip, &dev.stream, false)?;
        let head = model::head(&Src::Ckpt(&ck), &mut owned, hip, &dev.stream)?;
        for (q, p) in pp.iter_mut().zip(prompts) {
            let m = u64::from(q.m);
            em.arena.carry = q.carry;
            let LayerIn::Pending(pend) = q.inp else {
                bail!("depth 48 without a pending triple")
            };
            let (fh, lg) = emit_post(&mut em, &fm, head, &pend, m, false)?;
            dev.exec(&table, &mut em)?;
            let fb = d2h_in(&dev, &dev.arena, fh, (m * H * 2) as usize)?;
            let lb = d2h_in(&dev, &dev.arena, lg, (VOCAB * 2) as usize)?;
            let (top, v1, v2) = argmax(&bf16_to_f32(&lb));
            res.insert(
                p.id.clone(),
                json!({"layers": q.hashes, "final": sha256_hex(&fb), "logits": sha256_hex(&lb),
                       "top1": top, "top1_logit": v1, "gap12": v1 - v2}),
            );
        }
    } else {
        for (q, p) in pp.iter().zip(prompts) {
            res.insert(p.id.clone(), json!({"layers": q.hashes}));
        }
    }
    let doc = json!({"mode": "gate_a", "tp": 1, "depth": f.fn_depth, "poison": f.fn_poison,
                     "control": f.fn_control, "launch_mismatches": launch_bad, "prompts": res,
                     "launches": dev.launches, "wall_s": t_all.elapsed().as_secs_f64()});
    let name = format!(
        "gate_a{}{}.json",
        f.fn_poison
            .map(|b| format!("_poison{b:02x}"))
            .unwrap_or_default(),
        f.fn_control
            .as_deref()
            .map(|c| format!("_{c}"))
            .unwrap_or_default()
    );
    std::fs::write(out.join(&name), serde_json::to_string_pretty(&doc)?)?;
    println!(
        "gatea: wrote {} ({} launches, {} launch-count mismatches)",
        out.join(&name).display(),
        dev.launches,
        launch_bad.len()
    );
    ensure!(
        launch_bad.is_empty(),
        "launch counts differ from the graph: {launch_bad:?}"
    );
    Ok(())
}

/// D2H from any allocation of the rank.
pub(super) fn d2h_in(
    dev: &RankDev<'_>,
    base: &DevicePtr,
    addr: u64,
    bytes: usize,
) -> anyhow::Result<Vec<u8>> {
    let b = base.addr().as_u64();
    ensure!(
        addr >= b && addr - b + bytes as u64 <= base.len(),
        "d2h outside its allocation"
    );
    let mut v = vec![0u8; bytes];
    // FN57: the boot's sleeper (an aimed arm keys the read by its address and length)
    dev.wait_stream(addr.rotate_left(17) ^ bytes as u64)?;
    let a = base.addr().offset(addr - b).context("offset")?;
    dev.hip.memcpy_d2h_addr_blocking(&mut v, a, &dev.stream)?;
    Ok(v)
}

// ============================================================================ the engine (TP=2)

/// One rank's tier: the xtier file, the mover, the tier, the xptr tables.
pub(super) struct TierRank<'h, 'x> {
    pub(super) tier: ExpertTier,
    pub(super) mover: DeviceMover<'x, HipCopyQueue<'h>>,
    pub(super) xptr: DevicePtr,
    pub(super) xhost: PinnedBuf,
    pub(super) shared: Vec<[u64; 6]>,
    /// FN80b-2: u64 words of one xptr table (`XN` = 3 x 257 under EP; 3 x 513 under
    /// `--fn-experts tp2`: 512 local experts + the shared expert's half).
    pub(super) xn: usize,
    /// FN30b (a TIMING control, bytes move): `--fn-diag-alias-ram` points every RAM-tier
    /// record address at a VRAM slot's (same sub-record offset) -- (ram_lo, ram_hi, vram_base,
    /// vram_slots, stride); the tier's accounting is unchanged.
    pub(super) alias: Option<(u64, u64, u64, u64, u64)>,
}

/// FN30b: the RAM -> VRAM alias of [`TierRank::alias`] (`--fn-diag-alias-ram`; None when off).
pub(super) fn fn30b_alias(
    on: bool,
    q: &HipCopyQueue<'_>,
    stride: u64,
    ram_slots: u32,
) -> Option<(u64, u64, u64, u64, u64)> {
    on.then(|| {
        let lo = q.ram_device_addr;
        let vslots = q.vram.len() / stride;
        println!("fn30b: TIMING CONTROL -- every RAM-tier xptr entry aliased to a VRAM slot (bytes move)");
        (lo, lo + (u64::from(ram_slots) + 1) * stride, q.vram_base(), vslots, stride)
    })
}

impl TierRank<'_, '_> {
    pub(super) fn tables(&self, l: u32, rank: u32) -> (Vec<u64>, Vec<u64>, usize) {
        let (mut gu, mut dn) = (vec![0u64; self.xn], vec![0u64; self.xn]);
        let fetch = self.tier.fill_xptr(l, &mut gu, &mut dn);
        if let Some((lo, hi, vb, vs, st)) = self.alias {
            for x in gu.iter_mut().chain(dn.iter_mut()) {
                if *x >= lo && *x < hi {
                    let o = *x - lo;
                    *x = vb + ((o / st) % vs) * st + o % st;
                }
            }
        }
        // the shared expert's entry follows the local experts': rank 0's whole expert under EP,
        // every rank's half under `--fn-experts tp2` (FN80b-2; `shared` is empty where none)
        let _ = rank;
        if !self.shared.is_empty() {
            let e = self.xn / 3 - 1;
            for k in 0..3 {
                gu[3 * e + k] = self.shared[l as usize][k];
                dn[3 * e + k] = self.shared[l as usize][3 + k];
            }
        }
        (gu, dn, fetch)
    }
    pub(super) fn put(&mut self, l: u32, gu: &[u64], dn: &[u64]) {
        let base = (l as usize * 2 * self.xn) * 8;
        let s = self.xhost.as_mut_slice();
        for (i, v) in gu.iter().chain(dn.iter()).enumerate() {
            s[base + i * 8..base + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    pub(super) fn addr(&self, l: u32) -> (u64, u64) {
        let xb = (self.xn * 8) as u64;
        let o = self.xptr.addr().as_u64() + u64::from(l) * 2 * xb;
        (o, o + xb)
    }
    pub(super) fn upload(
        &self,
        hip: &HipThread,
        st: &Stream,
        l0: u32,
        n: u32,
    ) -> anyhow::Result<()> {
        let xb = (self.xn * 8) as u64;
        let off = u64::from(l0) * 2 * xb;
        hip.run_batch(&[
            Job::MemcpyPinned {
                dev: &self.xptr,
                dev_offset: off,
                host: &self.xhost,
                host_offset: off,
                bytes: u64::from(n) * 2 * xb,
                kind: MemcpyKind::HostToDevice,
                stream: st,
            },
            Job::SyncStream { stream: st },
        ])?;
        Ok(())
    }
    /// FN117b (S2): [`Self::upload`] without the sync -- the copy queued on `st`; the caller
    /// guarantees the stream drains (the drafter round's ids read) before `xhost` is rewritten.
    pub(super) fn upload_async(
        &self,
        hip: &HipThread,
        st: &Stream,
        l0: u32,
        n: u32,
    ) -> anyhow::Result<()> {
        let xb = (self.xn * 8) as u64;
        let off = u64::from(l0) * 2 * xb;
        hip.run_batch(&[Job::MemcpyPinned {
            dev: &self.xptr,
            dev_offset: off,
            host: &self.xhost,
            host_offset: off,
            bytes: u64::from(n) * 2 * xb,
            kind: MemcpyKind::HostToDevice,
            stream: st,
        }])?;
        Ok(())
    }
}

/// FN80d: the forced-miss law's count -- the DISTINCT routed experts of the rank's OWN range
/// `[e_lo, e_lo + eloc)` in one layer's route ids (EP: the rank's 256; `--fn-experts tp2` and
/// TP=1: all 512). Until FN80d the range was `256 * rank .. + 256` in every form, so under tp2 a
/// rank counted half the records it reads (FN80b-2 h3's `reads != demand_distinct`).
pub(super) fn demand_distinct(routed: &[u32], e_lo: u32, eloc: u32) -> u64 {
    let mut d: Vec<u32> = routed
        .iter()
        .copied()
        .filter(|&e| e >= e_lo && e < e_lo + eloc)
        .collect();
    d.sort_unstable();
    d.dedup();
    d.len() as u64
}

/// The demand path (the forced-miss control): run the route half, read the routes back, resolve.
pub(super) struct Demand<'a, 'h, 'x> {
    pub(super) devs: &'a mut [RankDev<'h>],
    pub(super) tables: &'a [FunctionTable<'h>],
    pub(super) tiers: &'a mut [TierRank<'h, 'x>],
    pub(super) counts: Vec<PathCounts>,
    pub(super) nvme_distinct: Vec<u64>,
    pub(super) unrouted_fetch: Vec<u64>,
}

#[allow(clippy::needless_range_loop)]
impl XptrHook for Demand<'_, '_, '_> {
    fn tables(
        &mut self,
        ems: &mut [Emitter<'_>],
        layer: u32,
        ids: &[u64],
        m: u64,
    ) -> anyhow::Result<Vec<(u64, u64)>> {
        let mut out = Vec::new();
        for r in 0..ems.len() {
            self.devs[r].exec(&self.tables[r], &mut ems[r])?;
        }
        // FN53b: THE LAYER-AHEAD TAIL READER -- a wide prefill pass reads this layer's drive
        // records into the bounce now, while the device runs the previous layer's experts and
        // this layer's front (inert without a valve bounce or on a decode pass)
        if m >= FN_MEM_VALVE_PREFETCH_ROWS {
            for t in self.tiers.iter_mut() {
                t.tier
                    .bounce_prefetch(layer, &mut t.mover)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
        }
        for r in 0..ems.len() {
            let dev = &self.devs[r];
            let b = d2h_in(dev, &dev.arena, ids[r], (m * 40) as usize)?;
            let routed: Vec<u32> = b
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let t = &mut self.tiers[r];
            let c = t
                .tier
                .resolve(layer, &routed, &mut t.mover)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            t.tier
                .check_dispatch(layer, &routed)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            self.counts[r].resident += c.resident;
            self.counts[r].ram += c.ram;
            self.counts[r].nvme += c.nvme;
            let rank = ems[r].dims.rank;
            self.nvme_distinct[r] += demand_distinct(&routed, ems[r].dims.e_lo, ems[r].dims.eloc);
            // entries of UNROUTED experts may stay Fetch (null); check_dispatch above proved every
            // routed one of this rank is Vram or Ram
            let (gu, dn, fetch) = t.tables(layer, rank);
            self.unrouted_fetch[r] += fetch as u64;
            t.put(layer, &gu, &dn);
            t.upload(dev.hip, &dev.stream, layer, 1)?;
            out.push(t.addr(layer));
        }
        Ok(out)
    }
}

/// FN71c: open the card's P2PDMA pool reader for the boot fill / the valve's bounce: the card is
/// the HIP thread's current device, its `allocate` file `<dir>/<bdf>` when the launcher mounted it
/// (a container: sysfs is read-only), else sysfs in place. `Err` = the reason the pool route is
/// refused (no pool on this boot, not published, held by another reader, the probe failed).
pub(super) fn open_pool_reader(
    hip: &HipThread,
    xt: &XtierFile,
    dir: &Path,
) -> anyhow::Result<std::result::Result<saber_hsa::NvmeVram, String>> {
    let bdfid = crate::fn63_hsa::hsa_bdfid(hip)?;
    let bdf = saber_hsa::bdf_string(bdfid);
    let mounted = dir.join(&bdf);
    let pool = if mounted.exists() {
        saber_hsa::PoolAsk::At {
            sysfs_dir: saber_hsa::pool_sysfs_dir(bdfid),
            allocate: mounted,
        }
    } else {
        saber_hsa::PoolAsk::Auto
    };
    let opts = saber_hsa::NvmeOptions {
        slots: 2,
        slot_bytes: 1 << 20,
        window_bytes: 0,
        pool,
    };
    let nv = match saber_hsa::NvmeVram::open_with(
        saber_hsa::DEFAULT_HSA_SONAME,
        bdfid,
        &opts,
        (xt.file(), xt.header().record_offset(0)),
    ) {
        Ok(nv) => nv,
        Err(e) => return Ok(Err(format!("{bdf}: the reader did not open: {e}"))),
    };
    if nv.route() != saber_hsa::NvmeRoute::Pool {
        return Ok(Err(format!(
            "{bdf}: {}",
            nv.p2p_refusal().unwrap_or("the pool route was not chosen")
        )));
    }
    Ok(Ok(nv))
}

/// FN71c: the pool reader a boot needs (`--xtier-boot-fill auto|pool`, `--xtier-nvme-p2p on`):
/// `None` = the host route (asked, or `auto` with the pool refused: the reason is printed); a
/// refusal is FATAL under `--xtier-boot-fill pool` or `--xtier-nvme-p2p on`.
pub(super) fn boot_pool_reader(
    hip: &HipThread,
    xt: &XtierFile,
    x: &XtierFlags,
    rank: usize,
) -> anyhow::Result<Option<saber_hsa::NvmeVram>> {
    let (try_pool, must) = x.boot_fill_pool();
    if !try_pool && !x.nvme_p2p() {
        return Ok(None);
    }
    match open_pool_reader(hip, xt, &x.xtier_p2p_dir)? {
        Ok(nv) => Ok(Some(nv)),
        Err(why) => {
            ensure!(
                !must && !x.nvme_p2p(),
                "rank {rank}: the P2PDMA pool is refused ({why}) and {} needs it",
                if must {
                    "--xtier-boot-fill pool"
                } else {
                    "--xtier-nvme-p2p on"
                }
            );
            println!(
                "fnfill rank{rank}: the P2PDMA pool is refused ({why}) -- the boot fill runs the host route"
            );
            Ok(None)
        }
    }
}

/// FN71c: the boot fill's ring over a pool reader (`saber_mem::p2p_fill::PoolFillIo`).
struct PoolRing<'a> {
    nv: &'a mut saber_hsa::NvmeVram,
    xt: &'a XtierFile,
    gpu: u64,
    stride: u64,
}

impl saber_mem::p2p_fill::PoolFillIo for PoolRing<'_> {
    type Err = anyhow::Error;
    fn read(&mut self, items: &[(u64, u32)]) -> anyhow::Result<()> {
        let it: Vec<(u64, u64)> = items
            .iter()
            .map(|&(off, s)| (off, self.gpu + u64::from(s) * self.stride))
            .collect();
        self.nv
            .read_pool_many(
                self.xt.file(),
                &it,
                self.stride,
                saber_mem::p2p_fill::FILL_QD,
            )
            .map_err(|e| anyhow::anyhow!("boot fill: {e}"))
    }
    fn copy(&mut self, slot: u32, dst: u64) -> anyhow::Result<u64> {
        self.nv
            .copy_from_pool(
                dst,
                self.gpu + u64::from(slot) * self.stride,
                self.stride,
                &[],
            )
            .map_err(|e| anyhow::anyhow!("boot fill copy: {e}"))
    }
    fn wait(&mut self, t: u64) -> anyhow::Result<()> {
        self.nv
            .wait(t)
            .map_err(|e| anyhow::anyhow!("boot fill wait: {e}"))
    }
}

/// FN71c: how [`warm`] fills the VRAM slots.
pub(super) enum WarmFill<'a> {
    /// Drive -> the pinned stage (sha256 checked) -> H2D (the pre-FN71c law, the control).
    Host,
    /// Drive -> the P2PDMA pool ring -> an SDMA copy into each slot; then `check` slots are read
    /// back through the stage and checked against the sha256 table (`u32::MAX` = every slot).
    Pool {
        nv: &'a mut saber_hsa::NvmeVram,
        check: u32,
    },
}

/// FN71c: the boot fill's sampled readback: this many VRAM slots (evenly spread) are read back and
/// checked against the sha256 table after a pool fill (every slot under `--fn-verify-slots`).
pub const POOL_FILL_CHECK: u32 = 64;

/// What [`warm`] did: (VRAM records, RAM records, records that differ from the sha256 table).
pub(super) type WarmOut = (u64, u64, u64);

/// Warm a rank's tier from its plan: RAM slots in place, VRAM slots through a pinned staging ring;
/// every record's bytes checked against the xtier's sha256 table on the way. FN71c: under
/// [`WarmFill::Pool`] the VRAM slots come drive -> the card's P2PDMA pool -> the slot (no host
/// DRAM), checked by a readback of `check` slots after the fill; the `fnfill` line names the
/// route, the ring and the times.
pub(super) fn warm(
    hip: &HipThread,
    xt: &XtierFile,
    tr: &mut TierRank<'_, '_>,
    stage: &mut PinnedBuf,
    fill: WarmFill<'_>,
    rank: usize,
) -> anyhow::Result<WarmOut> {
    let plan = tr.tier.warm_plan();
    let st = xt.header().stride as usize;
    let ram: Vec<(u64, u32)> = plan
        .iter()
        .filter_map(|&(i, p)| {
            if let Placement::Ram(s) = p {
                Some((i, s))
            } else {
                None
            }
        })
        .collect();
    let vram: Vec<(u64, u32)> = plan
        .iter()
        .filter_map(|&(i, p)| {
            if let Placement::Vram(s) = p {
                Some((i, s))
            } else {
                None
            }
        })
        .collect();
    let mut bad = 0u64;
    let verify = |slab: &[u8], pairs: &[(u64, u32)]| -> u64 {
        let n = 16usize;
        let per = pairs.len().div_ceil(n).max(1);
        std::thread::scope(|s| {
            let hs: Vec<_> = pairs
                .chunks(per)
                .map(|ch| {
                    s.spawn(move || {
                        ch.iter()
                            .filter(|&&(i, k)| {
                                !xt.verify(i, &slab[k as usize * st..(k as usize + 1) * st])
                            })
                            .count() as u64
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(u64::MAX)).sum()
        })
    };
    let t0 = Instant::now();
    // FN53b: each read into the region holding its slot (the slab; + the valve band's segments)
    {
        use saber_mem::device_mover::CopyQueue as _;
        for (first, buf) in tr.mover.queue_mut().ram_regions_mut() {
            let n = (buf.len() / st) as u32;
            let sub: Vec<(u64, u32)> = ram
                .iter()
                .filter(|r| r.1 >= first && r.1 - first < n)
                .map(|&(i, s)| (i, s - first))
                .collect();
            for ch in sub.chunks(256) {
                parallel_reads(xt, buf, st, ch, 32).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            bad += verify(buf, &sub);
        }
    }
    let t_ram = t0.elapsed().as_secs_f64();
    let per = stage.len() / st;
    if let WarmFill::Pool { nv, check } = fill {
        let t1 = Instant::now();
        let m = nv
            .pool()
            .ok_or_else(|| anyhow::anyhow!("boot fill: the reader has no pool"))?;
        let ring = saber_mem::p2p_fill::ring_slots(
            m.bytes,
            st as u64,
            saber_mem::p2p_fill::RING_CAP_BYTES,
            saber_mem::p2p_fill::FILL_QD,
        )
        .map_err(|e| anyhow::anyhow!("boot fill: {e}"))?;
        let vb = tr.mover.queue().vram_base();
        let vlen = tr.mover.queue().vram.len();
        nv.register_device(vb, vlen);
        let h = xt.header().clone();
        let recs: Vec<(u64, u64)> = vram
            .iter()
            .map(|&(i, s)| (h.record_offset(i), vb + u64::from(s) * st as u64))
            .collect();
        let before = nv.counts;
        let c = {
            let mut io = PoolRing {
                nv,
                xt,
                gpu: m.gpu,
                stride: st as u64,
            };
            saber_mem::p2p_fill::fill_through_ring(
                &mut io,
                &recs,
                ring,
                saber_mem::p2p_fill::FILL_QD,
            )?
        };
        let t_fill = t1.elapsed().as_secs_f64();
        // the readback: `check` slots evenly spread (all of them under --fn-verify-slots)
        let t2 = Instant::now();
        let n = vram.len();
        let pick: Vec<(u64, u32)> = if check as usize >= n {
            vram.clone()
        } else {
            (0..check as usize)
                .map(|k| vram[k * n / check.max(1) as usize])
                .collect()
        };
        for ch in pick.chunks(per) {
            let q = tr.mover.queue();
            let jobs: Vec<Job<'_>> = ch
                .iter()
                .enumerate()
                .map(|(k, &(_, s))| Job::MemcpyPinned {
                    dev: &q.vram,
                    dev_offset: u64::from(s) * st as u64,
                    host: stage,
                    host_offset: (k * st) as u64,
                    bytes: st as u64,
                    kind: MemcpyKind::DeviceToHost,
                    stream: &q.stream,
                })
                .chain(std::iter::once(Job::SyncStream { stream: &q.stream }))
                .collect();
            hip.run_batch(&jobs)?;
            let local: Vec<(u64, u32)> = ch
                .iter()
                .enumerate()
                .map(|(k, &(i, _))| (i, k as u32))
                .collect();
            bad += verify(stage.as_slice(), &local);
        }
        let nc = nv.counts;
        println!(
            "fnfill rank{rank}: route POOL -- {} VRAM records drive -> P2PDMA pool ({} MiB at {:#x}; ring {ring} slots = {:.0} MiB, QD {}) -> SDMA copy into the slot in {:.2} s ({:.2} GB/s; pool reads {} / {:.2} GB, copies {}, {} batches, {} ring waits); RAM L2 {} records drive -> RAM in {:.2} s; readback {} slots checked in {:.2} s, {bad} differ",
            vram.len(),
            m.bytes >> 20,
            m.gpu,
            (u64::from(ring) * st as u64) as f64 / 1048576.0,
            saber_mem::p2p_fill::FILL_QD,
            t_fill,
            (vram.len() * st) as f64 / t_fill.max(1e-9) / 1e9,
            nc.pool_reads - before.pool_reads,
            (nc.pool_bytes - before.pool_bytes) as f64 / 1e9,
            nc.pool_copies - before.pool_copies,
            c.batches,
            c.reuse_waits,
            ram.len(),
            t_ram,
            pick.len(),
            t2.elapsed().as_secs_f64(),
        );
        return Ok((vram.len() as u64, ram.len() as u64, bad));
    }
    let t1 = Instant::now();
    for ch in vram.chunks(per) {
        let reads: Vec<(u64, u32)> = ch
            .iter()
            .enumerate()
            .map(|(k, &(i, _))| (i, k as u32))
            .collect();
        parallel_reads(xt, stage.as_mut_slice(), st, &reads, 32)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        bad += verify(stage.as_slice(), &reads);
        let q = tr.mover.queue();
        let jobs: Vec<Job<'_>> = ch
            .iter()
            .enumerate()
            .map(|(k, &(_, s))| Job::MemcpyPinned {
                dev: &q.vram,
                dev_offset: u64::from(s) * st as u64,
                host: stage,
                host_offset: (k * st) as u64,
                bytes: st as u64,
                kind: MemcpyKind::HostToDevice,
                stream: &q.stream,
            })
            .chain(std::iter::once(Job::SyncStream { stream: &q.stream }))
            .collect();
        hip.run_batch(&jobs)?;
    }
    let t_fill = t1.elapsed().as_secs_f64();
    println!(
        "fnfill rank{rank}: route HOST -- {} VRAM records drive -> pinned stage (sha256 checked) -> H2D in {:.2} s ({:.2} GB/s); RAM L2 {} records drive -> RAM in {:.2} s",
        vram.len(),
        t_fill,
        (vram.len() * st) as f64 / t_fill.max(1e-9) / 1e9,
        ram.len(),
        t_ram,
    );
    Ok((vram.len() as u64, ram.len() as u64, bad))
}

pub(super) fn mem_total() -> anyhow::Result<u64> {
    let m = std::fs::read_to_string("/proc/meminfo")?;
    let kb: u64 = m
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .context("MemTotal")?
        .parse()?;
    Ok(kb * 1024)
}

/// The next request's PLE rows gathered ahead: (request index, row ids, (distinct, cache hits,
/// NVMe reads, read s), gather ms).
type Prefetched = (usize, Vec<u64>, (u64, u64, u64, f64), f64);

/// One request's timing and counters.
#[derive(Debug, Clone, Default)]
struct ReqStat {
    id: String,
    tokens: u32,
    chunks: u32,
    wall_ms: f64,
    host_ms: f64,
    ttft_ms: f64,
    top1: usize,
    top1_logit: f32,
    gap12: f32,
    tier: [PathCounts; 2],
    ple: (u64, u64, u64, f64),
    ple_hidden: bool,
    launches: u64,
    /// Demand path: distinct routed local records per rank (the NVMe reads' expected count).
    distinct: [u64; 2],
    /// Op-timing arm: summed kernel ms (rank 0) and per (layer kind | group).
    busy_ms: f64,
    groups: BTreeMap<String, f64>,
}

/// What the boot plans on the HOST before a card is touched: the kernel declarations, the arena
/// (sized by emitting the widest chunk), the per-rank byte ledger (which REFUSES an over-commit
/// here, before any HIP call), the host budget and the tier laws.
pub struct Boot {
    pub decls: Decls,
    pub rules: FlashNextRules,
    pub dr: plan::DryRun,
    /// FN42b: under `--fn-arena-borrow on` (a wide rung), the NARROW (2,048-row) arena sizes:
    /// the wide arena's scratch past them is lent to the expert tier between wide chunks.
    pub narrow: Option<super::kern::ArenaSizes>,
    pub budgets: Vec<Ep2Budget>,
    pub host: HostBudget,
    pub laws: Vec<crate::xtier_flags::TierLaw>,
    /// FN55c: the lazy tower's arena in patches when the boot chose it (`--fn-vision-max-patches`
    /// absent and the default region does not pin); the worker plans with it.
    pub vision_patches: Option<u32>,
    /// FN55c: the RAM L2 was capped below the complement (records stay NVMe-only).
    pub l2_capped: bool,
    /// FN107 (X1): the P2P pool's slots a rank planned EXCLUSIVE (counted static for the L2's
    /// complement; 0 = off / no pool / FN76's inclusive pool). The worker refuses a pool whose
    /// slot count differs.
    pub pool_excl: u64,
    /// FN80b-2: the expert tier's geometry (`--fn-experts`; from the split marker of the header).
    pub xg: super::xtp2::XGeom,
    /// FN120e: the glue-fold census and the region it sized (the class `fn.glue120`).
    pub glue: GlueCensus,
}

/// FN18: the default runtime ledger (`--fn-ledger-exact` / `--no-fn-ledger-exact`). FN18c
/// (2026-09-29, hold 2's paired win): the EXACT ledger is the default (+398 VRAM expert slots a rank
/// at the served boot); `--no-fn-ledger-exact` = FN7's planned maximum, the control. The weight
/// classes (FN22's `embed` half among them, resident for the drafter's device chain) are the pack
/// plan's and are never touched by either ledger.
pub const DEFAULT_FN_LEDGER_EXACT: bool = true;

/// FN33: the drafter's head on the device chain (`--fn-draft-head`): **`int2` IS THE DEFAULT**
/// (the 27B's int2 scorer's top 80 + the head's own GEMM cell re-scoring them; hold 1, 210 W:
/// BetterBench --quick single 75.9/77.3 -> 87.8/85.1 tok/s, conc-8 144.8/163.8 -> 173.6/166.1,
/// harness b1 31.15/33.84 -> 29.61/32.81 ms/step, acceptance 2.965 -> 2.956 tokens a step, gate D
/// 25/25 + 1,600/1,600 rows; LEDGER FN33); `bf16` = FN22's pass, the control.
pub const DEFAULT_FN_DRAFT_HEAD: saber_kernels::fn33::DraftHead =
    saber_kernels::fn33::DraftHead::Int2;

/// FN49b: the MoE route's lowering by the flags -- FN49a's DPP symbols by default,
/// `--fn-route-control` = the pre-FN49b symbols.
#[must_use]
pub fn route_form(f: &FnFlags) -> saber_kernels::moe::RouteForm {
    if f.fn_route_control {
        saber_kernels::moe::RouteForm::Shipped
    } else {
        saber_kernels::moe::RouteForm::Dpp
    }
}

/// FN18c: the warm-start profile's path inside the pack (`fn7-route-profile/1`; FN18's real-traffic
/// profile, 47,361 served steps) -- the default when neither `--xtier-profile` nor `--fn-profile`
/// is given.
pub const DEFAULT_PROFILE_IN_PACK: &str = "profile/route_profile.json";
/// FN65: the user's own warm-start profile, an ADDED pack member (`--fn-profile user`; written by
/// `saber flashnext-profile rebuild`; the pack's own profile is never rewritten).
pub const USER_PROFILE_IN_PACK: &str = "profile/route_profile.user.json";
/// FN65: target steps a recorded route file holds (~0.2-0.8 GB a file at served mixes).
pub const DEFAULT_FN_ROUTE_RECORD_STEPS: u64 = 20_000;
/// FN65: the recorder's ring on the drive, GiB.
pub const DEFAULT_FN_ROUTE_RECORD_GIB: f64 = 16.0;

impl FnFlags {
    /// FN18: whether the exact runtime ledger is on.
    #[must_use]
    pub fn ledger_exact(&self) -> bool {
        if self.fn_ledger_exact {
            true
        } else if self.no_fn_ledger_exact {
            false
        } else {
            DEFAULT_FN_LEDGER_EXACT
        }
    }

    /// FN18c: the warm-start profile: `--xtier-profile`, else `--fn-profile`, else the pack's
    /// [`DEFAULT_PROFILE_IN_PACK`].
    /// FN65: `--fn-profile pack` = the pack's file, `user` = the pack's added
    /// [`USER_PROFILE_IN_PACK`] member, anything else a path.
    #[must_use]
    pub fn profile_path(&self, pack: &Path) -> PathBuf {
        if let Some(p) = &self.xtier.xtier_profile {
            return p.clone();
        }
        match self.fn_profile.as_deref() {
            None => pack.join(DEFAULT_PROFILE_IN_PACK),
            Some(p) if p.as_os_str() == "pack" => pack.join(DEFAULT_PROFILE_IN_PACK),
            Some(p) if p.as_os_str() == "user" => pack.join(USER_PROFILE_IN_PACK),
            Some(p) => p.to_path_buf(),
        }
    }

    /// FN65: the recorder's configuration (`--fn-route-record DIR`).
    #[must_use]
    pub fn record_cfg(
        &self,
        dir: &Path,
        layers: u32,
        topk: u32,
        experts: u32,
    ) -> saber_mem::route_trace::RecordCfg {
        saber_mem::route_trace::RecordCfg {
            dir: dir.to_path_buf(),
            layers,
            topk,
            experts,
            rotate_steps: self.fn_route_record_steps.max(1),
            max_bytes: (self.fn_route_record_gib.max(0.0) * 1073741824.0) as u64,
            max_inflight: 512 << 20,
        }
    }
}

/// FN18: rank `r`'s runtime classes as ALLOCATED (the exact ledger): the pools (their plan is
/// asserted equal to the allocation at boot), the prefill arena (decode steps and the drafter's
/// passes run inside it, asserted at boot), the fixed buffers, the reduce scratch and the xptr
/// tables.
pub fn exact_runtime(
    f: &FnFlags,
    prompts: &[Prompt],
    r: u32,
    arena: u64,
    glue: super::exec::GlueRegion,
) -> anyhow::Result<Vec<saber_mem::flashnext_budget::Class>> {
    use saber_mem::flashnext_budget::Class;
    let (max_ctx, pool_w) = super::worker::pools_geometry(f, prompts)?;
    let dm = super::exec::Dims::new(2, r);
    let kv = f.kv_geom(max_ctx);
    let pools = super::worker::Pools::plan_bytes(
        &dm,
        max_ctx,
        f.fn_gdn_state_f32,
        pool_w,
        u64::from(3 + pool_w - 1),
        kv.is_some(),
        kv.map_or(0, |g| g.ix_pages()),
        f.kv_dtype(),
        f.kv_mtp_dtype(),
        f.pools_layout(),
    );
    let (h, w, rows) = (super::exec::H, super::exec::W, u64::from(f.max_chunk()));
    let mtp = if f.fn_mtp.on() || f.fn_mtp_ref.is_some() {
        rows * w * 2 + u64::from(super::worker::MAX_SEQS + 1) * w * 2 + w * 2
    } else {
        0
    };
    let mb = super::dev::ar_mb(f.max_chunk());
    let ar = super::dev::ar_ledger_bytes(mb);
    let buffers = (8 << 20)
        + rope_bytes(rope_pos(max_ctx))
        + 2 * rows * h * 2
        + u64::from(LAYERS) * rows * 40
        + 256
        + mtp
        + ar;
    let mk = |n: &str, b: u64| Class {
        name: n.to_string(),
        bytes: b,
    };
    let mut kvc = Vec::new();
    if let Some(g) = kv {
        // FN45b2: the tier's VRAM (the hot slab + pad page, the device page table); its pinned
        // RAM slab (`kv.ram`) is host memory, refused at boot by the plan (the admission law)
        // FN68d: both tiers' (the MTP layer's twin beside the main)
        kvc.push(mk(saber_mem::kv_tier::KV_HOT, g.hot_alloc_all()));
        kvc.push(mk(saber_mem::kv_tier::KV_INDEX, g.table_bytes_all()));
    }
    Ok(vec![
        mk("fn.pools", pools),
        mk("fn.arena.prefill", arena),
        mk("fn.buffers", buffers),
        mk("xtier.xptr", geometry::XPTR_BYTES),
        mk("fn.moe.sync", moe_sync_class(f, r)),
        mk("fn.glue120", glue_class(f, glue)),
        mk("fn.df120", dataflow_class(f)),
        mk("fn.qsa124", qsa124_class(f)),
        mk("fn.draft.int2", draft_int2_class(f)),
        mk("fn.pick", super::worker::pick_class_bytes(f.device_pick())),
        mk("xtier.hsa.words", hsa_words_class(f)),
        mk("fn.kv.desc", kv_desc_class(f)),
    ]
    .into_iter()
    .chain(kvc)
    .collect())
}

/// FN100: the `fn.buffers` class's census (`fnbufcensus` lines): every member with its bytes,
/// the rows it is sized for, its writers, the graphs baking its address and the lendable tail
/// past the heads of `--xtier-lend-arena-top-rows` rows; the members sum to the class.
/// FN134: `stride` = the boot's expert record stride (the tier's; whole under EP, half under tp2).
pub fn buffers_census(
    f: &FnFlags,
    prompts: &[Prompt],
    stride: u64,
) -> anyhow::Result<(Vec<String>, u64)> {
    let (max_ctx, _) = super::worker::pools_geometry(f, prompts)?;
    let rows = f.max_chunk();
    let mtp_on = f.fn_mtp.on() || f.fn_mtp_ref.is_some();
    let mb = super::dev::ar_mb(rows);
    let ar = super::dev::ar_ledger_bytes(mb);
    let w = super::exec::W;
    let small = if mtp_on {
        u64::from(super::worker::MAX_SEQS + 1) * w * 2 + w * 2
    } else {
        0
    };
    let c = super::bufs::census(
        rows,
        f.xtier.xtier_lend_arena_top_rows,
        mtp_on,
        super::exec::H,
        w,
        u64::from(LAYERS),
        rope_bytes(rope_pos(max_ctx)),
        ar,
        small,
        stride,
    );
    let mib = |x: u64| x as f64 / 1048576.0;
    let mut v = Vec::new();
    let (mut tot, mut slots) = (256u64, 0u64);
    for m in &c {
        v.push(format!(
            "fnbufcensus: {:<14} {:>8.1} MiB | rows {:>5} | tail {:>7.1} MiB = {:>3} slots | \
             writers: {} | baked: {}",
            m.name,
            mib(m.bytes),
            m.rows,
            mib(m.tail),
            m.slots,
            m.writers,
            m.baked
        ));
        tot += m.bytes;
        slots += m.slots;
    }
    v.push(format!(
        "fnbufcensus: total {:.1} MiB (+ the 256 B barrier) = the class; lendable {slots} slots \
         of {} B past heads of {} rows (reclaimed by a write of more rows: 0 copies, records \
         dropped)",
        mib(tot),
        stride,
        f.xtier.xtier_lend_arena_top_rows
    ));
    Ok((v, tot))
}

/// FN63b: the HSA lane's mark words (`layers + 1` u32, one fine-grained allocation, declared at
/// a page) under `--xtier-mover hsa-lane`; 0 otherwise (no allocation: the control's ledger is
/// unchanged).
#[must_use]
pub fn hsa_words_class(f: &FnFlags) -> u64 {
    if f.xtier.hsa_lane() {
        4096
    } else {
        0
    }
}

/// FN68b: the fp8 pages' descale table as allocated (`kvscale::KvScales::table`, 49 x 2 x 128
/// f32 = 50,176 B a rank) under `--fn-kv-dtype fp8` with a scale law other than `unit`; 0
/// otherwise (bf16 / FN68's unit law: no allocation, the null descale pointers).
#[must_use]
pub fn kv_desc_class(f: &FnFlags) -> u64 {
    if kv_desc_on(f) {
        (super::kvscale::TABLE_LAYERS * 2 * super::kvscale::DESC_N * 4) as u64
    } else {
        0
    }
}

/// FN68b: whether the boot allocates the descale table (fp8 pages, a law other than `unit`).
#[must_use]
pub fn kv_desc_on(f: &FnFlags) -> bool {
    f.kv_dtype() == saber_mem::kv_tier::KvDtype::Fp8 && f.fn_kv_scales.as_deref() != Some("unit")
}

/// FN30: rank `r`'s fused-MoE `sync` region as allocated (`super::exec::moe_sync_bytes`: 240 KiB
/// at TP=2 under `--fn-moe-train fused`, the default since FN30d; one 256 B stub under `legacy`).
#[must_use]
pub fn moe_sync_class(f: &FnFlags, r: u32) -> u64 {
    if f.fn_moe_train == super::exec::MoeTrain::Fused {
        // FN80b-2: the TP-split form routes 512 local experts (the slot's tile bound doubles)
        let dm = super::exec::Dims::new(2, r);
        super::exec::moe_sync_bytes(if f.experts_tp2() {
            dm.experts_tp2().eloc
        } else {
            dm.eloc
        })
    } else {
        256
    }
}

/// FN33: rank `r`'s int2 draft head as allocated (`saber_kernels::fn33::pack_bytes` of the
/// rank's vocabulary half, 85.3 MiB) when the device chain runs `--fn-draft-head int2`; one 256 B
/// stub otherwise (the exact ledger's remainder is the expert slots: the pack costs ~34 of them).
#[must_use]
pub fn draft_int2_class(f: &FnFlags) -> u64 {
    if draft_int2_on(f) {
        saber_kernels::fn33::pack_bytes(super::exec::VOCAB / 2)
    } else {
        256
    }
}

/// FN33: whether the boot builds the int2 draft head (MTP through the verify step, the device
/// chain, `--fn-draft-head int2`).
#[must_use]
pub fn draft_int2_on(f: &FnFlags) -> bool {
    f.fn_draft_head == saber_kernels::fn33::DraftHead::Int2
        && super::worker::mtp_verify(f)
        && !f.no_fn_mtp_device_chain
}

/// The pools' context (every sequence slot's KV / indexer capacity): `--fn-max-ctx` under
/// `--fn-serve` (rounded to 256), else every request + its decode (+ a second turn), at least the
/// 2,048 band. The worker's pools and the ledger's KV class are both this.
#[must_use]
pub fn pools_ctx(f: &FnFlags, prompts: &[Prompt]) -> u32 {
    if f.fn_serve.is_some() || f.scripted() {
        return f.max_ctx().div_ceil(256) * 256;
    }
    let turns_extra = if f.fn_turns > 1 {
        2 * f.fn_decode + 128
    } else {
        0
    };
    (prompts
        .iter()
        .map(|p| p.ids.len() as u32)
        .max()
        .unwrap_or(1)
        + f.fn_decode
        + turns_extra)
        .max(2048)
        .max(f.fn_pools_ctx.unwrap_or(0))
        .div_ceil(256)
        * 256
}

/// FN51: a harness sequence's admission under admission by reservation: the longest request +
/// its decode (+ a second turn) -- [`pools_ctx`]'s harness law without `--fn-pools-ctx` (which
/// sizes the pools, e.g. a short gate at a long boot's split law, not what a sequence holds).
/// FN51c: + [`super::kvwire::ADMIT_EXTRA`] (the step's open pages, as the served reservation)
/// and, under MTP, the decode at the verify cap's width: a harness group steps until its SLOWEST
/// sequence holds `--fn-decode` tokens, and every step advances a finished sequence by up to the
/// step's width (FN51b's `sequence 3: 164 tokens past its admitted 163` at MTP auto, b8).
#[must_use]
pub fn harness_tokens(f: &FnFlags, prompts: &[Prompt]) -> u32 {
    let turns_extra = if f.fn_turns > 1 {
        2 * f.fn_decode + 128
    } else {
        0
    };
    let width = if f.fn_mtp_arg.unwrap_or(f.fn_mtp).on() {
        super::exec::WMAX
    } else {
        1
    };
    prompts
        .iter()
        .map(|p| p.ids.len() as u32)
        .max()
        .unwrap_or(1)
        .saturating_add((f.fn_decode + turns_extra).saturating_mul(width))
        .saturating_add(super::kvwire::ADMIT_EXTRA)
}

/// FN134: the slab4 `mtp.experts` class rank `r` bills: its own file's bytes under EP (today's
/// ledger byte for byte); under `--fn-experts tp2` the LARGER of the two ranks' (the mirrored
/// placement needs equal slot counts, and rank 0's slab carries the MTP shared expert).
#[must_use]
pub fn mtp_slab_bill(bytes: [u64; 2], r: usize, tp2: bool) -> u64 {
    if tp2 {
        bytes[0].max(bytes[1])
    } else {
        bytes[r]
    }
}

/// FN110: the MTP expert forms the boot's arena-top law sizes: FN50's two (byte for byte today's)
/// + the slab4 arm's when it is booted.
fn mtp_forms(f: &FnFlags) -> Vec<super::exec::MtpX> {
    use super::exec::MtpX;
    let mut v = vec![MtpX::Bf16, MtpX::Fp8];
    if f.mtp_experts() == "slab4" {
        // FN138: the half slab's pass under `--fn-mtp-split tp2`
        v.push(f.mtp_x().unwrap_or(MtpX::Slab4));
    }
    v
}

/// FN76b: the narrow arena's scratch a pass of at most `rows` rows can take, from the host's own
/// emission (the boot's dry runs): every decode-form pass (the target's verify forms b 1/2/4/8 x
/// w 1..WMAX at the pools' context, the drafter's decode passes at the same rows) and a prefill
/// chunk of `rows` rows (at position 0 and ending at the pools' context: the sparse pipeline's
/// selection buffers) with the drafter's prefill-form pass over it. `(decode, prefill, mtp
/// prefill)` scratch high-water marks in bytes, each the max over its forms.
pub fn arena_top_need(
    decls: &Decls,
    rules: &FlashNextRules,
    f: &FnFlags,
    pctx: u32,
    rows: u32,
) -> anyhow::Result<(u64, u64, u64)> {
    let (t8, head8, t8p) = (f.trunk8(), f.trunk8_head(), f.trunk8_prefill());
    let mut dec = 0u64;
    for b in [1u32, 2, 4, 8] {
        for w in 1..=super::exec::WMAX {
            let d = plan::dry_run_verify_t8(decls, rules, b, w, pctx, t8, head8)?;
            dec = dec.max(d.sizes.scratch);
            for x in mtp_forms(f) {
                for ctx in [8192, pctx] {
                    let (_, s) = plan::dry_run_mtp_h8(decls, rules, b * w, true, ctx, w, x, head8)?;
                    dec = dec.max(s.scratch);
                }
            }
        }
    }
    let mut pf = 0u64;
    let mut p0s = vec![0u32];
    if f.fn_qsa == saber_kernels::fn19::QsaMode::Sparse
        && pctx > saber_kernels::fn19::DENSE_MAX_POS + 1
    {
        p0s.push(pctx.saturating_sub(rows));
    }
    for p0 in p0s {
        let d = plan::dry_run_prefill_at_t8(decls, rules, 2, rows, p0, t8p, head8)?;
        pf = pf.max(d.sizes.scratch);
    }
    let mut mp = 0u64;
    for x in mtp_forms(f) {
        for ctx in [8192, pctx] {
            let (_, s) =
                plan::dry_run_mtp_h8(decls, rules, rows.min(512), false, ctx, 1, x, head8)?;
            mp = mp.max(s.scratch);
        }
    }
    Ok((dec, pf, mp))
}

/// The rules every emitter of a TP=2 boot reads (both forms, the dry runs): the prod rules + the
/// flags' arms.
///
/// # Errors
/// An unknown arm.
pub fn boot_rules(f: &FnFlags) -> anyhow::Result<FlashNextRules> {
    let mut rules = FlashNextRules::prod()?;
    // FN37: the producer-rr law rides the rules every emitter of this boot reads (both forms)
    rules.hc.rr_producer = f.hc_rr_producer();
    // FN85: the prologue-reduce arm rides the rules too (every emitter of the boot, dry runs)
    rules.hc.ar_prologue = ar_prologue(f);
    // FN86b: the launch-fusion families ride the rules too (every emitter of the boot, dry runs)
    rules.hc.fuse86 = fuse86(f)?;
    // FN111: the re-tiles ride the rules too
    rules.hc.tiles111 = tiles111(f)?;
    rules.hc.sel128 = sel128(f)?;
    // FN131: the phase-split decode downs ride the rules too
    rules.hc.down131 = down131(f)?;
    // FN141: the drained-ring twins ride the rules too
    rules.hc.k141 = k141(f)?;
    // FN163: the route twins ride the rules too
    rules.hc.route163 = route163(f)?;
    // FN122b: the prefill tiles ride the rules too
    rules.hc.prefill122 = prefill122(f)?;
    Ok(rules)
}

/// Plan the TP=2 boot on the host (no HIP): the arena from the widest chunk, then each rank's
/// ledger (weights from `plan.json`, the runtime classes with THIS arena, the budget).
/// FN107 (X1): the P2P pool's slots a rank planned EXCLUSIVE (`--xtier-pool-exclusive`): the pool
/// (`pool_mib`, the card's `p2pmem/size`) less `--xtier-pool-slots-reserve-mib`, in whole strides,
/// when the pool becomes slots (`--xtier-pool-slots auto|on`, the worker form, not the valve's
/// bounce, FN8's law); 0 otherwise (`off` = FN76's inclusive pool, the control). `on` refuses a
/// boot that cannot make them exclusive.
/// # Errors
/// `--xtier-pool-exclusive on` without exclusive pool slots.
pub fn pool_exclusive_slots(f: &FnFlags, pool_mib: u64, stride: u64) -> anyhow::Result<u64> {
    let (on, must) = f.xtier.pool_exclusive();
    if !on {
        return Ok(0);
    }
    let why = if !f.xtier.pool_slots().0 {
        Some("--xtier-pool-slots off")
    } else if f.xtier.nvme_p2p() {
        Some("--xtier-nvme-p2p on (the pool is the valve's bounce)")
    } else if f.fn_one_process {
        Some("the one-process form (no pool slots)")
    } else if pool_mib == 0 {
        Some("no P2PDMA pool on this box")
    } else {
        None
    };
    if let Some(why) = why {
        ensure!(!must, "--xtier-pool-exclusive on: refused under {why}");
        return Ok(0);
    }
    let n = ((pool_mib << 20).saturating_sub(f.xtier.xtier_pool_slots_reserve_mib << 20)) / stride;
    if n > 0 {
        println!(
            "fnpoolx107: THE POOL SLOTS EXCLUSIVE (X1) -- {n} slots a rank (pool {pool_mib} MiB, \
             reserve {} MiB) count as STATIC for the RAM L2's complement: their records keep no \
             RAM copy (-{:.3} GiB pinned a rank), a victim is written back to RAM \
             (--xtier-pool-exclusive off = FN76's inclusive pool)",
            f.xtier.xtier_pool_slots_reserve_mib,
            (n * stride) as f64 / (1u64 << 30) as f64
        );
    }
    Ok(n)
}

pub fn plan_boot(
    pack: &Path,
    prebuilt: &Path,
    prompts: &[Prompt],
    f: &FnFlags,
) -> anyhow::Result<Boot> {
    let b = plan_boot_at(pack, prebuilt, prompts, f, None)?;
    // FN150: an NVFP4 boot prints its ledger (the 676-page slots) and then refuses by name every
    // piece with no NVFP4 body yet -- a half-wired boot cannot start
    let no = super::xtp2::nvfp4_refusals(f, 2);
    ensure!(
        no.is_empty(),
        "--fn-experts-fmt nvfp4: the expert cells are wired (FN150) but this boot needs pieces \
         with no NVFP4 body yet: {}",
        no.join("; ")
    );
    Ok(b)
}

/// [`plan_boot`] at a given expert geometry (`None` = the flags' and the pack's, [`XGeom::of`];
/// FN150's host test plans the Quark ledger at the NVFP4 stride) and without the NVFP4 refusals.
pub fn plan_boot_at(
    pack: &Path,
    prebuilt: &Path,
    prompts: &[Prompt],
    f: &FnFlags,
    xg_at: Option<super::xtp2::XGeom>,
) -> anyhow::Result<Boot> {
    f.kv_tier_check(2)?;
    // FN80b-2: the expert tier's geometry (`--fn-experts`), from the split marker of the header
    super::xtp2::check(f, 2)?;
    super::shx114::check(f, 2)?;
    f.mtp_x()?;
    let xg = match xg_at {
        Some(g) => g,
        None => super::xtp2::XGeom::of(f, pack)?,
    };
    let decls = Decls::load_for_ar(prebuilt, f.fn_ar == "host")?;
    let rules = boot_rules(f)?;
    let pctx = pools_ctx(f, prompts);
    ensure!(
        pctx <= ROPE_POS_MAX,
        "the pools' context {pctx} past the checkpoint's {ROPE_POS_MAX} positions"
    );
    // FN42b: the arena at a chunk cap -- the boot's widest (FN38), and under the borrow also the
    // narrow rung's (every pass of <= MAX_ROWS rows: decode, the drafter, a narrow chunk)
    let size = |cap: u32| -> anyhow::Result<plan::DryRun> {
        let m_max = prompts
            .iter()
            .map(|p| p.ids.len() as u32)
            .max()
            .unwrap_or(1)
            .max(f.fn_chunk.min(cap))
            // FN13: a second turn's one-prompt rendering is longer than any prompt -- the widest
            // chunk
            .max(if f.fn_turns > 1 || f.fn_serve.is_some() || f.scripted() {
                cap
            } else {
                0
            })
            // FN17 / FN19: a prompt past the widest chunk runs in chunks of at most it (FN38:
            // `--fn-max-chunk`, 2,048 by default)
            .min(cap);
        // FN61f: under `w8` / `w8-head` the prefill chunk runs on the W8 records -> size the
        // arena from THAT train (its activation quants), not the bf16 one
        let (t8, head8) = (f.trunk8_prefill(), f.trunk8_head());
        let mut dr = plan::dry_run_cap_x(&decls, &rules, 2, m_max, f.fn_dump, t8, head8, xg.tp2)?;
        if f.fn_qsa == saber_kernels::fn19::QsaMode::Sparse
            && pctx > saber_kernels::fn19::DENSE_MAX_POS + 1
        {
            // FN19: the arena also holds a sparse chunk's selection buffers at the pools' context
            // (FN38: the widest chunk ending at the pools' context)
            let wide = m_max.max(MAX_ROWS).min(pctx);
            let s = plan::dry_run_prefill_at_x(
                &decls,
                &rules,
                2,
                wide,
                pctx - wide,
                t8,
                head8,
                xg.tp2,
            )?
            .sizes;
            dr.sizes.scratch = dr.sizes.scratch.max(s.scratch);
            dr.sizes.zero = dr.sizes.zero.max(s.zero);
            dr.sizes.ones = dr.sizes.ones.max(s.ones);
            dr.sizes.carry = dr.sizes.carry.max(s.carry);
        }
        if f.fn_mtp.on() || f.fn_mtp_ref.is_some() {
            // FN15: the arena also holds the drafter's widest pass (a prompt piece of <= 512 rows,
            // or a decode pass at 8 rows)
            // (FN15b: + the verify-width pass 1 at 8 sequences x 5 rows)
            for (m, d, w) in [(m_max.min(512), false, 1), (8, true, 1), (40, true, 5)] {
                // FN61g: under `w8-head` the drafter's head is the W8 head (+ its act quant)
                // FN110: the slab4 arm's pass (the trunk's train, a Q8 mix) is sized too; the
                // default arms keep the bf16 pass's sizes byte for byte
                let mut s = plan::dry_run_mtp_h8(
                    &decls,
                    &rules,
                    m,
                    d,
                    8192,
                    w,
                    super::exec::MtpX::Bf16,
                    head8,
                )?
                .1;
                if f.mtp_experts() == "slab4" {
                    let (_, x) =
                        plan::dry_run_mtp_h8(&decls, &rules, m, d, 8192, w, f.mtp_x()?, head8)?;
                    s.scratch = s.scratch.max(x.scratch);
                    s.zero = s.zero.max(x.zero);
                    s.ones = s.ones.max(x.ones);
                    s.carry = s.carry.max(x.carry);
                }
                dr.sizes.scratch = dr.sizes.scratch.max(s.scratch);
                dr.sizes.zero = dr.sizes.zero.max(s.zero);
                dr.sizes.ones = dr.sizes.ones.max(s.ones);
                dr.sizes.carry = dr.sizes.carry.max(s.carry);
            }
        }
        Ok(dr)
    };
    let dr = size(f.max_chunk())?;
    // FN120e: the glue-fold region from the dry run over every served form (a `seeds` / `fold`
    // boot; `off` allocates none)
    let glue = if glue120(f)? == Glue120::Off {
        GlueCensus {
            forms: Vec::new(),
            widest_zero: 0,
            widest_ones: 0,
            region: super::exec::GlueRegion { zero: 0, ones: 0 },
        }
    } else {
        let mut c = glue_census(&decls, &rules, f, prompts, xg.tp2)?;
        // FN120e: the gate control -- a forced zero part (the census still runs and is printed)
        if let Some(z) = f.fn_glue_region_mib {
            c.region.zero = z << 20;
        }
        c
    };
    let narrow = if f.arena_borrow() {
        Some(size(MAX_ROWS)?.sizes)
    } else {
        None
    };
    ensure!(
        !f.arena_borrow()
            || (!f.fn_one_process && f.fn_poison.is_none() && f.fn_poison_step.is_none()),
        "--fn-arena-borrow on runs in the worker form without the poison arms (they write the \
         whole arena, the lent tail included)"
    );
    let pj = std::fs::read_to_string(pack.join("plan.json"))?;
    // FN19: the ledger's KV class is the pools' (8 slots x the pools' context) whenever that is
    // more than the budget's default 131,072 tokens: a longer context is paid in expert slots
    let rt = RuntimeParams {
        prefill_arena: dr.sizes.total(),
        kv_tokens: RuntimeParams::default()
            .kv_tokens
            .max(u64::from(super::worker::MAX_SEQS) * u64::from(pctx)),
        ..RuntimeParams::default()
    };
    let exact = f.ledger_exact();
    let mut budgets = Vec::new();
    // FN99: the embedding half's bytes a rank moved to pinned RAM (`--fn-embed-placement ram`)
    let mut embed_ram = 0u64;
    for r in 0..2u32 {
        let mut classes = if exact {
            exact_runtime(f, prompts, r, dr.sizes.total(), glue.region)?
        } else {
            let mut c = rt.classes();
            c.push(saber_mem::flashnext_budget::Class {
                name: "fn.moe.sync".to_string(),
                bytes: moe_sync_class(f, r),
            });
            c.push(saber_mem::flashnext_budget::Class {
                name: "fn.glue120".to_string(),
                bytes: glue_class(f, glue.region),
            });
            c.push(saber_mem::flashnext_budget::Class {
                name: "fn.df120".to_string(),
                bytes: dataflow_class(f),
            });
            c.push(saber_mem::flashnext_budget::Class {
                name: "fn.qsa124".to_string(),
                bytes: qsa124_class(f),
            });
            c.push(saber_mem::flashnext_budget::Class {
                name: "fn.draft.int2".to_string(),
                bytes: draft_int2_class(f),
            });
            c.push(saber_mem::flashnext_budget::Class {
                name: "fn.pick".to_string(),
                bytes: super::worker::pick_class_bytes(f.device_pick()),
            });
            c.push(saber_mem::flashnext_budget::Class {
                name: "xtier.hsa.words".to_string(),
                bytes: hsa_words_class(f),
            });
            c.push(saber_mem::flashnext_budget::Class {
                name: "fn.kv.desc".to_string(),
                bytes: kv_desc_class(f),
            });
            c
        };
        // FN55b: the tower (its weight window + its arena) is billed BEFORE the expert slots are
        // sized (the slots are the remainder); a text-only boot's ledger is unchanged
        // FN55c: under `lazy` the class is the REGION the tower will be uploaded into (the
        // upload plan's weight arena + the tower's arena), LENT to the expert tier at boot
        let vmode = f.vision_mode(pack);
        if vmode != VisionMode::Off {
            ensure!(
                !f.fn_one_process,
                "--fn-vision runs in the worker form (the tower is booted on each rank process)"
            );
            let (w, a) = if vmode == VisionMode::Lazy {
                super::vision::region_bytes(pack, r, f.vision_patches())?
            } else {
                super::vision::demand(pack, r, f.vision_patches())?
            };
            println!(
                "fnbudget rank{r}: fn.vision ({}) {:.3} GiB = weights {:.3} + arena {:.3} \
                 ({} patches){}",
                if vmode == VisionMode::Lazy {
                    "FN55c lazy: the region, LENT to the expert tier until an image"
                } else {
                    "FN55b"
                },
                (w + a) as f64 / (1u64 << 30) as f64,
                w as f64 / (1u64 << 30) as f64,
                a as f64 / (1u64 << 30) as f64,
                f.vision_patches(),
                if vmode == VisionMode::Lazy {
                    format!(" = {} lendable slots", w / xg.stride + a / xg.stride)
                } else {
                    String::new()
                }
            );
            classes.push(saber_mem::flashnext_budget::Class {
                name: super::vision::CLASS.to_string(),
                bytes: w + a,
            });
        }
        let mut weights = weights_from_plan(&pj, r)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .0;
        // FN99: `--fn-embed-placement ram` -- the embedding half leaves the VRAM ledger (its bytes
        // become expert slots) and is billed in the host's pinned law (`fnhost`, below)
        if f.embed_in_ram() {
            let key = format!("fn.w.rank{r}.embed");
            let i = weights
                .iter()
                .position(|c| c.name == key)
                .with_context(|| format!("{key}: not in the plan's ledger"))?;
            let c = weights.remove(i);
            ensure!(
                embed_ram == 0 || embed_ram == c.bytes,
                "FN99: the embedding halves differ ({embed_ram} vs {} B)",
                c.bytes
            );
            embed_ram = c.bytes;
            println!(
                "fnbudget rank{r}: {key} {:.3} GiB ({} B) -> PINNED RAM (FN99, --fn-embed-placement \
                 ram: read in place over UVA by the drafter chain) -> {} more strides for the slots",
                c.bytes as f64 / (1u64 << 30) as f64,
                c.bytes,
                c.bytes / xg.stride
            );
        }
        // FN50: the fp8 arm's `mtp.experts` class is the additional file's bytes (the bf16
        // records are not uploaded)
        if f.mtp_experts() == "fp8" {
            let key = format!("fn.w.rank{r}.mtp.experts");
            let c = weights
                .iter_mut()
                .find(|c| c.name == key)
                .with_context(|| format!("{key}: not in the plan's ledger"))?;
            let b8 = super::mtp8::rank_bytes(pack, r)?;
            println!(
                "fnbudget rank{r}: mtp.experts fp8 (FN50) {:.3} GiB (bf16 {:.3})",
                b8 as f64 / (1u64 << 30) as f64,
                c.bytes as f64 / (1u64 << 30) as f64
            );
            c.bytes = b8;
        }
        // FN110: the slab4 arm's `mtp.experts` class is the MXFP4 file's bytes (the MTP layer's
        // grouped tables are EP's 257 entries; FN134: under `--fn-experts tp2` too -- the MTP
        // layer runs at `Dims::mtp_form`, the EP split + whole shape, whatever the target's form)
        if f.mtp_experts() == "slab4" {
            let key = format!("fn.w.rank{r}.mtp.experts");
            let c = weights
                .iter_mut()
                .find(|c| c.name == key)
                .with_context(|| format!("{key}: not in the plan's ledger"))?;
            // FN134: under `--fn-experts tp2` both ranks bill the LARGER rank's slab (rank 0's
            // holds the MTP shared expert: +3 MiB = 2 slots), so the ranks' slot counts -- and
            // the mirrored placement FN80b-2 checks -- stay equal (h1's smoke: 17,494 vs 17,496
            // slots diverged the placement at the first prompt)
            let b4 = if f.mtp_x()? == super::exec::MtpX::Slab4Tp2 {
                // FN138: the half slab (513 half records, equal on both ranks by construction);
                // both files must exist (each rank slices both)
                super::mtpx4::rank_bytes(pack, 0)?;
                super::mtpx4::rank_bytes(pack, 1)?;
                super::mtpx4::half_slab_bytes()
            } else {
                mtp_slab_bill(
                    [
                        super::mtpx4::rank_bytes(pack, 0)?,
                        super::mtpx4::rank_bytes(pack, 1)?,
                    ],
                    r as usize,
                    f.experts_tp2(),
                )
            };
            println!(
                "fnbudget rank{r}: mtp.experts slab4 (FN110, MXFP4 records) {:.3} GiB (bf16 {:.3}) -> fnmtpx110 slab4",
                b4 as f64 / (1u64 << 30) as f64,
                c.bytes as f64 / (1u64 << 30) as f64
            );
            c.bytes = b4;
        }
        // FN61c: the W8 trunk's records are ADDED weight bytes (the bf16 records stay for prefill)
        if f.trunk8() {
            ensure!(
                !f.fn_one_process,
                "--fn-trunk {} runs in the worker form (the -trunk8 pack's rank halves)",
                f.fn_trunk
            );
            let b8 = super::trunk8::rank_bytes(pack, r, f.trunk8_head())?;
            println!(
                "fnbudget rank{r}: trunk8 (FN61c, --fn-trunk {}) {:.3} GiB added",
                f.fn_trunk,
                b8 as f64 / (1u64 << 30) as f64
            );
            weights.push(saber_mem::flashnext_budget::Class {
                name: format!("fn.w.rank{r}.trunk8"),
                bytes: b8,
            });
            // FN61e: the dropped bf16 records leave their plan classes (gdn / attn.qo / attn.kv)
            if f.trunk8_prefill() {
                let mut tot = 0u64;
                for (class, b) in super::trunk8::dropped_bf16(pack, r)? {
                    let key = format!("fn.w.rank{r}.{class}");
                    let c = weights
                        .iter_mut()
                        .find(|c| c.name == key)
                        .with_context(|| format!("{key}: not in the plan's ledger"))?;
                    c.bytes = c
                        .bytes
                        .checked_sub(b)
                        .with_context(|| format!("{key}: drops {b} of {}", c.bytes))?;
                    tot += b;
                }
                println!(
                    "fnbudget rank{r}: trunk bf16 records DROPPED (FN61e, --fn-trunk {}) {:.3} GiB",
                    f.fn_trunk,
                    tot as f64 / (1u64 << 30) as f64
                );
            }
            // FN108: `w8-hc8[-head]` -- the fp8 mixing matrices are ADDED (`fn.w.shared.hc8`) and
            // the bf16 target mixers leave `fn.w.shared.hc` (never uploaded)
            if f.hc8() {
                let b8 = super::hc8::shared_bytes(pack)?;
                let drop = super::hc8::dropped_bf16(pack)?;
                let key = format!("fn.w.shared.{}", super::hc8::BF16_CLASS);
                let c = weights
                    .iter_mut()
                    .find(|c| c.name == key)
                    .with_context(|| format!("{key}: not in the plan's ledger"))?;
                c.bytes = c
                    .bytes
                    .checked_sub(drop)
                    .with_context(|| format!("{key}: drops {drop} of {}", c.bytes))?;
                weights.push(saber_mem::flashnext_budget::Class {
                    name: format!("fn.w.shared.{}", super::hc8::CLASS),
                    bytes: b8,
                });
                println!(
                    "fnbudget rank{r}: HC mixing matrices at fp8 (FN108, --fn-trunk {}) {:.3} GiB \
                     added, the bf16 target mixers {:.3} GiB DROPPED (net {:.3} GiB to the slots)",
                    f.fn_trunk,
                    b8 as f64 / (1u64 << 30) as f64,
                    drop as f64 / (1u64 << 30) as f64,
                    (drop as f64 - b8 as f64) / (1u64 << 30) as f64
                );
            }
            // FN61g: `w8-head` never uploads the bf16 head (its class leaves the ledger)
            if f.trunk8_head() {
                let (class, b) = super::trunk8::head_bf16(pack, r)?;
                let key = format!("fn.w.rank{r}.{class}");
                let c = weights
                    .iter_mut()
                    .find(|c| c.name == key)
                    .with_context(|| format!("{key}: not in the plan's ledger"))?;
                c.bytes = c
                    .bytes
                    .checked_sub(b)
                    .with_context(|| format!("{key}: drops {b} of {}", c.bytes))?;
                println!(
                    "fnbudget rank{r}: bf16 head NOT resident (FN61g, --fn-trunk w8-head) {:.3} GiB \
                     dropped; the W8 head serves every post + the drafter's rerank",
                    b as f64 / (1u64 << 30) as f64
                );
            }
        }
        // FN64d: a billed graph exec budget -- the plan grows by today's unbilled execs and the
        // class takes the budget (the difference becomes expert slots); unbilled = unchanged.
        // FN64e: the default CHARGE -- the class takes the execs the boot holds and the plan grows
        // by the same bytes (slots unchanged; the ledger counts the kernarg pools)
        let (graph_grow, graph_class) = f.graph_exec_ledger()?;
        if graph_class > 0 {
            let who = match f.graph_exec_bill()? {
                ExecBill::Charge => "FN64e charge",
                ExecBill::One => "FN98 one graph per form",
                _ => "FN64d",
            };
            println!(
                "fnbudget rank{r}: fn.graph.exec ({who}) {} MiB billed ({} execs x {} MiB); plan \
                 budget +{} MiB ({}) -> {} MiB to the slots",
                graph_class >> 20,
                (graph_class >> 20) / GRAPH_EXEC_POOL_MIB,
                GRAPH_EXEC_POOL_MIB,
                graph_grow >> 20,
                if f.graph_exec_bill()? == ExecBill::One {
                    "the class + the VRAM main's lazy boot left to its graph pools".to_string()
                } else {
                    format!("FN64d's {FN64D_UNBILLED_EXECS} measured lazy execs")
                },
                (graph_grow - graph_class) >> 20
            );
            classes.push(saber_mem::flashnext_budget::Class {
                name: "fn.graph.exec".to_string(),
                bytes: graph_class,
            });
        }
        // FN71b: the kernel's P2PDMA pool (`amdgpu.p2pmem_mb`) is pinned VRAM no allocator sees;
        // the fence (`--fn-budget-mib`, a pool-free card's bytes) bills it as a named class
        // (host tests plan against a pool-free card: their slot counts must not depend on the
        // box's boot line; `ledger_law`'s fake-sysfs test covers the read)
        let pool_mib = if cfg!(test) {
            0
        } else {
            crate::ledger_law::box_p2pmem_mib()
        };
        if pool_mib > 0 {
            println!(
                "fnbudget rank{r}: {} {pool_mib} MiB billed (the kernel's P2PDMA pool, pinned at \
                 boot; FN71b) -> {pool_mib} MiB fewer for the slots",
                crate::ledger_law::P2PMEM_CLASS
            );
            classes.push(saber_mem::flashnext_budget::Class {
                name: crate::ledger_law::P2PMEM_CLASS.to_string(),
                bytes: pool_mib << 20,
            });
        }
        // FN80b-2: the TP-split form's xptr tables (513 entries a GEMM) and the shared expert's
        // HALF on both ranks (one half-layout buffer a layer, as uploaded)
        if xg.tp2 {
            let c = classes
                .iter_mut()
                .find(|c| c.name == "xtier.xptr")
                .context("xtier.xptr: not in the runtime classes")?;
            c.bytes = xg.xptr_bytes();
            let c = weights
                .iter_mut()
                .find(|c| c.name == "fn.w.shared.shared_expert")
                .context("fn.w.shared.shared_expert: not in the plan's ledger")?;
            let half = u64::from(LAYERS) * super::xsplit::half_layout().1;
            println!(
                "fnbudget rank{r}: shared_expert (FN80b-2, the split's half on both ranks) {:.3} GiB (whole {:.3}); xtier.xptr {} B",
                half as f64 / (1u64 << 30) as f64,
                c.bytes as f64 / (1u64 << 30) as f64,
                xg.xptr_bytes()
            );
            c.bytes = half;
        }
        // FN114c: `--fn-shared-expert split` -- rank 0 keeps the whole shared expert (prefill and
        // the drafter) and ADDS its half; rank 1's class (billed whole, uploaded nothing today)
        // holds its half; both add the half tables
        if f.shared_split() {
            let c = weights
                .iter_mut()
                .find(|c| c.name == "fn.w.shared.shared_expert")
                .context("fn.w.shared.shared_expert: not in the plan's ledger")?;
            let (whole, add) = (c.bytes, super::shx114::rank_extra_bytes(r, c.bytes));
            c.bytes += add;
            println!(
                "fnbudget rank{r}: shared_expert (FN114c split) {:.3} GiB billed (whole {:.3}; {})",
                c.bytes as f64 / (1u64 << 30) as f64,
                whole as f64 / (1u64 << 30) as f64,
                if r == 0 {
                    "+ the half + the tables: rank 0 keeps the whole for prefill / the drafter"
                } else {
                    "the half + the tables inside the whole's bill"
                }
            );
        }
        let b = Ep2Budget::plan(
            r,
            (f.fn_budget_mib << 20) + graph_grow,
            weights,
            classes,
            xg.stride,
            xg.records,
        );
        match b {
            Ok(b) => {
                if xg.fmt != saber_mem::xtier::XtierFormat::Mxfp4E8m0 {
                    let rem = b.budget - b.weight_bytes() - b.runtime_bytes();
                    println!("{}", xg.slots_line(r, b.slots, rem));
                }
                let mut led = Ledger::new();
                b.declare(&mut led, 0)
                    .map_err(|e| anyhow::anyhow!("rank {r}: ledger: {e}"))?;
                budgets.push(b);
            }
            Err(e) => {
                println!("fnbudget rank{r}: REFUSED at {} MiB: {e}", f.fn_budget_mib);
                bail!("the byte ledger refuses the boot: {e}");
            }
        }
    }
    let mut host = HostBudget::this_box(mem_total()?);
    // FN51: the Flash-Next plane pins no park staging (the 27B's 4 GiB `serve.pinned`): its own
    // pinned bytes outside the classes are the warm/verify stage (64 slots), the xptr mirrors and
    // the tier's table mirror -- declared at FN_PLANE_PINNED
    host.serve_pinned = FN_PLANE_PINNED;
    if f.fn_ar == "host" {
        // Both workers map both mailboxes; each physical file is pinned only once.
        let region = 2 * super::host_ar::slot_bytes(f.max_chunk()) + 4 * super::host_ar::BLOCKS;
        let per_rank = region.div_ceil(4096) * 4096;
        host.serve_pinned += 2 * per_rank;
    }
    // FN51e: the PLE row cache counts at its flag's bytes (was 1 GiB a rank whatever the flag)
    host.ple_cache_per_rank = f.xtier.ple_cache_mib << 20;
    // FN99: the embedding half a rank in pinned RAM (0 at `vram`)
    host = host.with_embed(embed_ram);
    let mut laws = Vec::new();
    let mut l2_capped = false;
    // FN107 (X1): the pool's slots EXCLUSIVE -- never reclaimed, so they count as static for the
    // RAM L2's complement (their records keep no RAM copy)
    let pool_excl = pool_exclusive_slots(
        f,
        if cfg!(test) {
            0
        } else {
            crate::ledger_law::box_p2pmem_mib()
        },
        xg.stride,
    )?;
    for b in &mut budgets {
        // FN43: the staging ring comes out of the ledger's slots (line `xtier.stage_ring`) and
        // is LENT back to the tier while no staged pass runs: the L2 covers every record outside
        // the STATIC slots (a lent slot is inclusive: its record keeps its RAM shadow)
        let ring = f.xtier.stage_ring_slots(f.ring_default_ok(), f.max_chunk());
        if ring > 0 {
            *b = b
                .clone()
                .with_stage_ring(u64::from(ring))
                .map_err(|e| anyhow::anyhow!("--xtier-stage: {e}"))?;
        }
        let l2 = host.default_l2_slots(b.complement().saturating_sub(pool_excl), xg.stride);
        laws.push(
            f.xtier
                .resolve(
                    (b.slots + b.stage_slots) as u32,
                    l2 as u32,
                    f.ring_default_ok(),
                    f.max_chunk(),
                )
                .map_err(anyhow::Error::msg)?,
        );
    }
    // FN45b4: the tier's pinned KV slab counts in the host law; a boot past the pinnable bytes
    // is refused (the expert RAM cache is not shrunk for it) unless `--fn-kv-ram-overcommit`
    if let Some(g) = f.kv_geom(pctx) {
        // FN45c3: + the prompt cache's pinned staging buffer (declared, refused past it at open)
        let pc = if f.pcache_gib().is_some() {
            super::worker::pcache_staging_bound(pctx, f.max_chunk())
        } else {
            0
        };
        // FN113: + the tree's pinned RAM tier (refused past the pin law like the slab)
        let pc = pc + f.prefix_tree_ram_bytes();
        host = host.with_kv_ram(g.ram_alloc_all() + pc);
        let mut l2: Vec<u64> = laws.iter().map(|l| u64::from(l.tier.ram_slots)).collect();
        // FN55c: CAP THE LEND first -- with the arena's patches at their default, a lazy region
        // that does not pin is planned smaller (the largest arena whose region keeps every
        // record in VRAM or RAM); only when none does is the L2 capped (below)
        if f.vision_mode(pack) == VisionMode::Lazy
            && f.fn_vision_max_patches.is_none()
            && f.xtier.xtier_ram_slots.is_none()
            && host.check(&l2, xg.stride).is_err()
        {
            for p in [12_288u32, 8_192, 4_096] {
                let mut g = f.clone();
                g.fn_vision_max_patches = Some(p);
                let b = plan_boot_at(pack, prebuilt, prompts, &g, xg_at)?;
                if !b.l2_capped {
                    println!(
                        "fnhost: FN55c lazy vision at {pctx} tokens: the region at {} patches \
                         does not pin beside the KV slab -> the lend is CAPPED at {p} patches \
                         (images up to {} image tokens); every expert record stays in VRAM or RAM",
                        DEFAULT_FN_VISION_MAX_PATCHES,
                        p / 4
                    );
                    return Ok(Boot {
                        vision_patches: Some(p),
                        ..b
                    });
                }
            }
        }
        // FN55c: a LAZY tower's region is lent (inclusive: its records keep RAM shadows), so the
        // L2 covers the complement of the static slots; where that does not pin beside the KV
        // slab the L2 is CAPPED at what pins (the boot is never refused for the lend): the
        // records past it stay NVMe-only (demand reads), FN55b's 262,144 form
        // (only the lend's own shortfall is capped: a boot whose L2 less the region's slots
        // does not pin either is refused below, as without the tower)
        let region: Vec<u64> = budgets.iter().map(lazy_slots).collect();
        let without: Vec<u64> = l2
            .iter()
            .zip(&region)
            .map(|(l, r)| l.saturating_sub(*r))
            .collect();
        if f.vision_mode(pack) == VisionMode::Lazy
            && f.xtier.xtier_ram_slots.is_none()
            && host.check(&l2, xg.stride).is_err()
            && host.check(&without, xg.stride).is_ok()
        {
            // the boot's pins with no L2 slot (the spares counted), then what is left a rank
            let zeros = vec![0u64; l2.len()];
            let base = host.pinned(&zeros, xg.stride);
            let per = host.pinnable().saturating_sub(base) / l2.len().max(1) as u64;
            let cap = per / xg.stride;
            l2_capped = true;
            for (i, l) in laws.iter_mut().enumerate() {
                let want = u64::from(l.tier.ram_slots);
                let got = want.min(cap);
                let stat = u64::from(l.tier.vram_slots);
                println!(
                    "fnhost: FN55c lazy vision at {pctx} tokens: rank{i} RAM L2 CAPPED {want} -> \
                     {got} slots (the complement of {stat} static slots does not pin beside the \
                     KV slab): {} records a rank stay NVMe-only (demand reads); \
                     --fn-max-ctx 253952 or a smaller --fn-vision-max-patches pins the whole set",
                    xg.records.saturating_sub(stat + got)
                );
                l.tier.ram_slots = got as u32;
                l2[i] = got;
            }
        }
        if let Err(e) = host.check(&l2, xg.stride) {
            let ceil = host.kv_ceiling(
                &l2,
                xg.stride,
                g.token_bytes_all(),
                super::kvwire::page_bytes(f.kv_dtype()),
                2048,
            );
            if f.fn_kv_ram_overcommit {
                println!("fnhost: KV RAM OVERCOMMIT (gate arm): {e}");
            } else if let Some(t) = f.fn_kv_admit_slab_tokens {
                // LR-K: the arm's own refusal, the largest slab that pins named (beside the same
                // L2; a bigger slab's paged index also moves expert slots into the L2, so the
                // named value is an upper bound -- step down by a few k if it is refused too)
                let base = host.clone().with_kv_ram(pc).pinned(&l2, xg.stride);
                let left = host.pinnable().saturating_sub(base) / l2.len().max(1) as u64;
                let max = g.admit_slab_tokens_in(left);
                bail!(
                    "--fn-kv-admit-slab-tokens {t}: {e}; the largest admission slab this host pins \
                     beside the expert RAM cache and the prefix tree is ~{max} tokens a rank (the \
                     default is {})",
                    super::kvwire::admit_slab_tokens(pctx)
                );
            } else {
                bail!(
                    "--fn-kv-tier at {pctx} tokens: {e}; the largest context this host pins \
                     beside the expert RAM cache is {ceil} (--fn-max-ctx), or --fn-kv-tier off \
                     (the flat pools), or --fn-kv-ram-overcommit (gate arms)"
                );
            }
        }
    }
    // FN99: the pinned law of the arm -- `ram` is PINNED-NEUTRAL only while the RAM L2 still
    // covers every record outside the static slots; a boot where it does not is REFUSED by name
    if f.embed_in_ram() {
        let mut why = None;
        for (b, l) in budgets.iter().zip(&laws) {
            let (have, want) = (
                u64::from(l.tier.ram_slots),
                b.complement().saturating_sub(pool_excl),
            );
            if have < want || l2_capped {
                why = Some(format!(
                    "--fn-embed-placement ram: rank{} RAM L2 {have} slots < the complement {want} \
                     of the {} static{} slots (the embedding's {:.3} GiB pinned leaves the L2 \
                     incomplete): refused (FN99's law; --fn-embed-placement vram)",
                    b.rank,
                    b.slots,
                    if pool_excl > 0 {
                        format!(" + {pool_excl} exclusive pool")
                    } else {
                        String::new()
                    },
                    embed_ram as f64 / (1u64 << 30) as f64
                ));
                break;
            }
        }
        let l2: Vec<u64> = laws.iter().map(|l| u64::from(l.tier.ram_slots)).collect();
        if why.is_none() {
            if let Err(e) = host.check(&l2, xg.stride) {
                if f.fn_kv_ram_overcommit {
                    println!("fnembed99: KV RAM OVERCOMMIT (gate arm): {e}");
                } else {
                    why = Some(format!(
                        "--fn-embed-placement ram: {e}: refused (FN99's law; --fn-embed-placement \
                         vram)"
                    ));
                }
            }
        }
        if let Some(why) = why {
            // FN107: THE STAND-DOWN -- only an explicit `ram` is refused; a plain boot (the
            // default placement) plans again at `vram` and says so
            ensure!(!f.embed_explicit(), "{why}");
            println!(
                "fnembed99: STANDS DOWN to vram -- the default placement ram would leave the RAM \
                 L2 incomplete ({why}); re-planned at vram (the fnbudget / fnhost lines below \
                 are the boot's)"
            );
            let mut g = f.clone();
            g.fn_embed_placement = Some("vram".to_string());
            return plan_boot_at(pack, prebuilt, prompts, &g, xg_at);
        }
        println!(
            "fnembed99: placement ram ({}) -- {:.3} GiB a rank pinned ({} B), {} static slots a \
             rank, RAM L2 {} slots a rank = the complement (complete); host pins {:.3} of {:.3} GiB",
            if f.embed_explicit() {
                "explicit"
            } else {
                "the default, FN107"
            },
            embed_ram as f64 / (1u64 << 30) as f64,
            embed_ram,
            budgets.first().map_or(0, |b| b.slots),
            laws.first().map_or(0, |l| l.tier.ram_slots),
            host.pinned(&l2, xg.stride) as f64 / (1u64 << 30) as f64,
            host.pinnable() as f64 / (1u64 << 30) as f64
        );
    }
    if laws.iter().any(|l| l.stage.is_some()) {
        ensure!(
            !f.fn_one_process
                && f.fn_poison.is_none()
                && f.fn_poison_step.is_none()
                && !f.fn_diag_alias_ram,
            "--xtier-stage (FN43's ring) runs in the worker form without the poison arms and the \
             FN30b alias (they write / re-point the ring's lent slots)"
        );
        ensure!(
            laws.iter()
                .all(|l| l.tier.defer_spares == 0 && l.tier.refill_spares == 0),
            "--xtier-stage lends the ring to the tier: FN8's law only (no --xtier-defer, \
             --xtier-demote none)"
        );
    }
    Ok(Boot {
        decls,
        rules,
        dr,
        narrow,
        budgets,
        host,
        laws,
        vision_patches: None,
        l2_capped,
        pool_excl,
        xg,
        glue,
    })
}

#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]
fn engine_tp2(
    hips: [&HipThread; 2],
    boot: Boot,
    pack: &Path,
    ckpt: &Path,
    cfg: &Qwen4ExpConfig,
    prebuilt: &Path,
    prompts: &[Prompt],
    out: &Path,
    f: &FnFlags,
) -> anyhow::Result<()> {
    let t_boot = Instant::now();
    let Boot {
        decls,
        rules,
        dr,
        narrow: _,
        budgets,
        host,
        laws,
        vision_patches: _,
        l2_capped: _,
        pool_excl: _,
        xg: _,
        glue: _,
    } = boot;
    for (r, h) in hips.iter().enumerate() {
        h.set_device(r as i32)?;
        // FN106d: the page form before the first pinned allocation
        h.set_pinned_pages(f.pinned_pages()?);
    }
    hips[0].enable_peer_access(1)?;
    hips[1].enable_peer_access(0)?;
    let prof_path = f.profile_path(pack);
    let prof = if f.xtier.no_xtier_profile {
        None
    } else {
        Some(
            RouteProfile::from_json(&std::fs::read_to_string(&prof_path).with_context(|| {
                format!(
                    "the warm-start profile {} (--fn-profile user needs `saber flashnext-profile rebuild` first)",
                    prof_path.display()
                )
            })?)
            .map_err(|e| anyhow::anyhow!("{}: {e}", prof_path.display()))?,
        )
    };
    // ---- weights (pack records, sha256-checked), per rank
    let recs = PackRecs::open(pack, ckpt)?;
    let lc = LoadCounters::new();
    let ps0 = PluginSet::load_named(hips[0], prebuilt, &PLUGINS, &lc)?;
    let ps1 = PluginSet::load_named(hips[1], prebuilt, &PLUGINS, &lc)?;
    let tables = [ps0.functions(hips[0])?, ps1.functions(hips[1])?];
    // FN51: the table covers the pools' context (the worker form's law)
    let cs = rope_table(rope_pos(pools_ctx(f, prompts)));
    let mut devs: Vec<RankDev<'_>> = (0..2)
        .map(|r| RankDev::new(hips[r], r as u32, dr.sizes, &cs))
        .collect::<anyhow::Result<_>>()?;
    let arules = ArRules::prod()?;
    let mb = super::dev::ar_mb(f.max_chunk());
    let st0 = devs[0].ar_alloc(&arules, mb)?;
    let st1 = devs[1].ar_alloc(&arules, mb)?;
    let mut pair = ArPair {
        host: false,
        rules: arules,
        st: [st0, st1],
        mb,
        calls_exact: 0,
        calls_wht6: 0,
        decode_exact: f.fn_decode_ar == "exact",
        calls_decode_exact: 0,
        ll: ar_ll(f),
        calls_ll: 0,
        gp: ar_gp(f),
        calls_gp: 0,
        far: moe_ar_push(f),
        calls_far: 0,
        far_stood_down: 0,
        mix: mixer_ar_push(f),
        calls_mix: 0,
        mix_stood_down: 0,
    };
    let t_w = Instant::now();
    let mut owned = [Owned::new(), Owned::new()];
    let mut lws: [Vec<LayerW>; 2] = [Vec::new(), Vec::new()];
    let mut fms = [HcW::default(); 2];
    let mut heads = [0u64; 2];
    let mut shared: Vec<[u64; 6]> = Vec::new();
    for r in 0..2 {
        let src = Src::Pack(&recs, r as u32);
        for l in 0..LAYERS {
            lws[r].push(model::dense_layer(
                &src,
                &mut owned[r],
                hips[r],
                &devs[r].stream,
                l,
                false,
                false,
            )?);
        }
        fms[r] = model::final_mixer(&src, &mut owned[r], hips[r], &devs[r].stream, false)?;
        heads[r] = model::head(&src, &mut owned[r], hips[r], &devs[r].stream)?;
        if r == 0 {
            for l in 0..LAYERS {
                shared.push(model::shared_expert(
                    &recs,
                    &mut owned[0],
                    hips[0],
                    &devs[0].stream,
                    l,
                )?);
            }
        }
    }
    println!(
        "fnweights: rank0 {:.3} GiB + rank1 {:.3} GiB of dense records on the cards in {:.1} s, every record == its sha256",
        owned[0].bytes as f64 / (1u64 << 30) as f64,
        owned[1].bytes as f64 / (1u64 << 30) as f64,
        t_w.elapsed().as_secs_f64()
    );
    // ---- the expert tiers
    ensure!(
        f.fn_sdma_lanes == "shared",
        "--fn-sdma-lanes {}: the worker form only (the one-process form keeps the pooled queues)",
        f.fn_sdma_lanes
    );
    let xts: Vec<XtierFile> = (0..2)
        .map(|r| {
            XtierFile::open(
                &pack.join(format!("rank{r}/experts.xtier")),
                OpenMode::Direct,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut tiers: Vec<TierRank<'_, '_>> = Vec::new();
    for r in 0..2 {
        let h = xts[r].header().clone();
        h.require_mxfp4().map_err(|e| anyhow::anyhow!(e))?;
        let tp = laws[r].tier.clone();
        let q = HipCopyQueue::new(
            hips[r],
            h.stride,
            tp.vram_slots,
            saber_mem::expert_tier::ExpertTier::slab_slots(&tp) - 1,
            h.layers,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut q = q;
        if f.xtier.two_queue() {
            q.enable_two_queue().map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        // FN63b: the HSA lane (this process holds both ranks: one lane per rank's HIP thread)
        if f.xtier.hsa_lane() {
            let desc = q
                .enable_hsa_lane(
                    crate::fn63_hsa::hsa_bdfid(hips[r])?,
                    f.xtier.hsa_engines()?,
                    h.layers,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("xtiermover rank{r}: hsa-lane {desc}");
        }
        // FN72b: the landed-mark law
        q.set_marks(f.xtier.mover_marks()?, h.layers)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if f.xtier.xtier_mover_census {
            q.set_census().map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        q.set_submit(f.xtier.plan_submit());
        let (vb, rb) = (q.vram_base(), q.ram_device_addr);
        let mut tier = ExpertTier::new(&h, tp.clone(), prof.as_ref(), vb, rb)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        tier.set_fast_plan(f.xtier.plan_fast());
        let xptr = hips[r].malloc(u64::from(LAYERS) * 2 * XBYTES)?;
        let xhost = hips[r].host_malloc(u64::from(LAYERS) * 2 * XBYTES)?;
        let alias = fn30b_alias(
            f.fn_diag_alias_ram,
            &q,
            h.stride,
            saber_mem::expert_tier::ExpertTier::slab_slots(&tp) - 1,
        );
        tiers.push(TierRank {
            tier,
            mover: DeviceMover::new(&xts[r], q, 32).with_defer(tp.defer_spares > 0),
            xptr,
            xhost,
            shared: if r == 0 { shared.clone() } else { Vec::new() },
            xn: XN,
            alias,
        });
    }
    for r in 0..2 {
        println!("{}", budgets[r].boot_line());
        println!(
            "{}",
            host.boot_line(
                u64::from(laws[r].tier.ram_slots),
                geometry::EXPERT_RECORDS - u64::from(laws[r].tier.vram_slots),
                geometry::EXPERT_STRIDE
            )
        );
        println!("{}", tiers[r].tier.boot_line());
        println!("{}", laws[r].boot_line(r as u32));
    }
    let t_warm = Instant::now();
    let mut stage = hips[0].host_malloc(64 * geometry::EXPERT_STRIDE)?;
    ensure!(
        !f.xtier.nvme_p2p(),
        "--xtier-nvme-p2p on serves the valve's bounce: the worker form only"
    );
    for r in 0..2 {
        let mut nv = boot_pool_reader(hips[r], &xts[r], &f.xtier, r)?;
        let fill = match nv.as_mut() {
            Some(nv) => WarmFill::Pool {
                nv,
                check: if f.fn_verify_slots {
                    u32::MAX
                } else {
                    POOL_FILL_CHECK
                },
            },
            None => WarmFill::Host,
        };
        let (v, ram, bad) = warm(hips[r], &xts[r], &mut tiers[r], &mut stage, fill, r)?;
        drop(nv);
        println!(
            "fnwarm rank{r}: {v} VRAM + {ram} RAM records from NVMe, {bad} differ from the xtier sha256 table ({:.1} s so far)",
            t_warm.elapsed().as_secs_f64()
        );
        ensure!(
            bad == 0,
            "rank {r}: {bad} expert records differ from their sha256"
        );
    }
    // ---- the PLE tier + hash
    let pre = format!("{P}layers.1.ple.ple_embedding.");
    let ck = Ckpt::open(ckpt)?;
    let ph = PleHash::from_buffers(
        &cfg.ple,
        cfg.eos_token_id,
        &ck.raw(&format!("{pre}layer_multipliers"))?,
        &ck.raw(&format!("{pre}ngram_heads_vocab_sizes"))?,
        &ck.raw(&format!("{pre}ngram_heads_offsets"))?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let idx = PleIndex::from_json(
        &std::fs::read_to_string(pack.join("ple/ple_index.json"))?,
        PleSourceKind::CkptBf16,
        PleGate::Default,
        None,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ple = PleTier::new(
        idx,
        OpenMode::Direct,
        f.xtier.ple_qd_prefill,
        ((f.xtier.ple_cache_mib << 20) / 320) as u32,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{}", ple.boot_line());
    // ---- per-rank buffers: sequence state, embed rows, PLE rows (double-buffered), route-id log
    let max_ctx = prompts
        .iter()
        .map(|p| p.ids.len() as u32)
        .max()
        .unwrap_or(1);
    let seqs: Vec<Seq> = (0..2)
        .map(|r| {
            Seq::new(
                hips[r],
                &Dims::new(2, r as u32),
                max_ctx,
                f.fn_gdn_state_f32,
            )
        })
        .collect::<anyhow::Result<_>>()?;
    let emb: Vec<DevicePtr> = (0..2)
        .map(|r| hips[r].malloc(u64::from(max_ctx) * H * 2))
        .collect::<Result<_, _>>()?;
    let rowsd: Vec<DevicePtr> = (0..2)
        .map(|r| hips[r].malloc(u64::from(max_ctx) * H * 2))
        .collect::<Result<_, _>>()?;
    let mut staged = [
        hips[0].host_malloc(u64::from(max_ctx) * 5120)?,
        hips[0].host_malloc(u64::from(max_ctx) * 5120)?,
    ];
    let idlog = hips[0].malloc(u64::from(LAYERS * f.max_chunk()) * 40)?;
    let ev: Vec<(Event, Event)> = (0..2)
        .map(|r| {
            Ok((
                hips[r].create_event(EventFlags::Default)?,
                hips[r].create_event(EventFlags::Default)?,
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut ems: Vec<Emitter<'_>> = (0..2)
        .map(|r| {
            Emitter::new(
                &decls,
                &rules,
                Dims::new(2, r as u32),
                Arena::carve(devs[r].base(), dr.sizes),
            )
        })
        .collect();
    for d in &mut devs {
        d.poison = f.fn_poison;
        d.op_timing = f.fn_op_timing;
        d.gate_time = f.fn_gate_time;
    }
    // FN30: the fused MoE train's sync region per rank, zeroed ONCE here (outside the arena and
    // its poison; the ledger class `fn.moe.sync`)
    let moe_sync: Vec<DevicePtr> = (0..2)
        .map(|r| -> anyhow::Result<DevicePtr> {
            let p = hips[r].malloc(moe_sync_class(f, r as u32))?;
            hips[r].memset(&p, 0, &devs[r].stream)?;
            Ok(p)
        })
        .collect::<anyhow::Result<_>>()?;
    for (r, e) in ems.iter_mut().enumerate() {
        e.qsa = f.fn_qsa;
        e.vclass = f.fn_verify_class;
        e.glue_v1 = f.fn_glue_control;
        e.kv8 = f.kv_dtype() == saber_mem::kv_tier::KvDtype::Fp8;
        e.kv8_mtp = e.kv8;
        ensure!(
            !kv_desc_on(f),
            "--fn-kv-dtype fp8's scale law (FN68b) runs in the worker form; the one-process \
             control takes --fn-kv-scales unit (FN68's 1.0)"
        );
        e.moe_train = f.fn_moe_train;
        e.route_form = route_form(f);
        e.miss_order = miss_order(f);
        e.ar_prologue = ar_prologue(f);
        e.moe_sync = moe_sync[r].addr().as_u64();
    }
    if f.fn_control.as_deref() == Some("swap-arg") {
        for e in &mut ems {
            e.ctl = Ctl::SwapArg;
        }
    }
    for e in &mut ems {
        e.state_f32 = f.fn_gdn_state_f32;
    }
    let (sz, _) = (dr.sizes, 0);
    println!(
        "fnboot: TP=2 / EP=2 one process two cards; arena {:.1} MiB/rank (scratch {:.1} zero {:.2} ones {:.2} carry {:.1}) sized by emitting the widest chunk; \
         launches/chunk {} over {} symbols (GDN {} / PLE {} / QSA {} per layer + post {}); AR band exact < {} B, wht6 above; boot {:.1} s",
        sz.total() as f64 / 1048576.0,
        sz.scratch as f64 / 1048576.0,
        sz.zero as f64 / 1048576.0,
        sz.ones as f64 / 1048576.0,
        sz.carry as f64 / 1048576.0,
        dr.per_layer.iter().sum::<u32>() + dr.post,
        dr.by_symbol.len(),
        dr.per_layer[0],
        dr.per_layer[1],
        dr.per_layer[3],
        dr.post,
        arules.quant_min_bytes,
        t_boot.elapsed().as_secs_f64()
    );
    if f.fn_evict_all {
        for t in &mut tiers {
            t.tier.evict_all();
        }
        println!("fnevict: evict_all on both ranks -- every placement is Fetch; the demand path serves the next request");
    }

    // ---- the request loop
    let mut all_stats: BTreeMap<String, Vec<ReqStat>> = BTreeMap::new();
    let mut cap_hashes = serde_json::Map::new();
    let mut times: Vec<(Group, i32, String, f64)> = Vec::new();
    let mut prefetched: Option<Prefetched> = None;
    let caps_dir = out.join("capture");
    let order: Vec<usize> = (0..prompts.len()).collect();
    for &conc in &f.fn_conc {
        for pass in 0..f.fn_repeat {
            let key = format!("conc{conc}_pass{pass}");
            let mut stats = Vec::new();
            let mut group_t0 = Instant::now();
            for (k, &pi) in order.iter().enumerate() {
                if k as u32 % conc.max(1) == 0 {
                    group_t0 = Instant::now();
                }
                let p = &prompts[pi];
                let t_req = Instant::now();
                let m_all = p.ids.len() as u32;
                // PLE rows: gathered ahead (by the previous request), or now
                let slot = k % 2;
                let (pc, ple_ms, hidden) = match prefetched.take() {
                    Some((i, _, c, ms)) if i == k => (c, ms, true),
                    _ => {
                        let ids = ph.ids(PleContext::start(cfg.eos_token_id), &p.ids);
                        let g0 = Instant::now();
                        let c = ple
                            .gather(&ids, &mut staged[slot].as_mut_slice()[..ids.len() * 320])
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        (
                            (c.distinct, c.cache_hits, c.nvme_reads, c.read_s),
                            g0.elapsed().as_secs_f64() * 1e3,
                            false,
                        )
                    }
                };
                let e = embed_rows(&recs, &p.ids)?;
                for r in 0..2 {
                    hips[r].memcpy_h2d_blocking(&emb[r], &e, &devs[r].stream)?;
                    hips[r].memcpy_h2d_blocking(
                        &rowsd[r],
                        &staged[slot].as_slice()[..(m_all as usize) * 5120],
                        &devs[r].stream,
                    )?;
                    seqs[r].zero(hips[r], &devs[r].stream)?;
                }
                let cuts: Vec<(u32, u32)> = if f.fn_chunk > 0 && f.fn_chunk < m_all {
                    let mut v = Vec::new();
                    let mut a = 0;
                    while a < m_all {
                        let b = (a + f.fn_chunk).min(m_all);
                        v.push((a, b));
                        a = b;
                    }
                    v
                } else {
                    vec![(0, m_all)]
                };
                let mut cap = f.fn_dump.then(|| Cap::new(m_all as usize));
                let mut rs = ReqStat {
                    id: p.id.clone(),
                    tokens: m_all,
                    chunks: cuts.len() as u32,
                    ple: pc,
                    ple_hidden: hidden,
                    ..ReqStat::default()
                };
                let launches0 = devs[0].launches;
                let mut host_ms = 0.0;
                let mut wall_ms = 0.0;
                let mut logits = Vec::new();
                for (ci, &(a, b)) in cuts.iter().enumerate() {
                    let m = b - a;
                    let last = ci + 1 == cuts.len();
                    let h0 = Instant::now();
                    let cc: Vec<ChunkConsts> = (0..2)
                        .map(|r| consts(&devs[r], 0, m, a, ci as u32))
                        .collect::<anyhow::Result<_>>()?;
                    for t in &mut tiers {
                        t.tier.begin_step();
                    }
                    let complete = tiers.iter().all(|t| t.tier.complete());
                    if complete {
                        for (r, t) in tiers.iter_mut().enumerate() {
                            for l in 0..LAYERS {
                                let (gu, dn, fetch) = t.tables(l, r as u32);
                                ensure!(fetch == 0, "a complete tier has a Fetch placement");
                                t.put(l, &gu, &dn);
                            }
                            t.upload(hips[r], &devs[r].stream, 0, LAYERS)?;
                        }
                    }
                    for r in 0..2 {
                        hips[r].record_event(&ev[r].0, &devs[r].stream)?;
                    }
                    let mut ins: Vec<LayerIn> = (0..2)
                        .map(|r| LayerIn::Embed(emb[r].addr().as_u64() + u64::from(a) * H * 2))
                        .collect();
                    let rows: Vec<u64> = (0..2)
                        .map(|r| rowsd[r].addr().as_u64() + u64::from(a) * H * 2)
                        .collect();
                    for l in 0..LAYERS {
                        for r in 0..2 {
                            tiers[r]
                                .mover
                                .gate(l, &devs[r].stream)
                                .map_err(|e| anyhow::anyhow!("{e}"))?;
                        }
                        let mut lw = [lws[0][l as usize], lws[1][l as usize]];
                        if complete {
                            for r in 0..2 {
                                (lw[r].xgu, lw[r].xdn) = tiers[r].addr(l);
                            }
                        }
                        let sts = [seqs[0].st[l as usize], seqs[1].st[l as usize]];
                        let outs = {
                            let mut ar: Option<&mut dyn Reduce> = Some(&mut pair);
                            if complete {
                                let mut hook = Resident(&lw);
                                emit_layer(
                                    &mut ems, &lw, &sts, &ins, &rows, &cc, &mut ar, &mut hook,
                                )?
                            } else {
                                let mut hook = Demand {
                                    devs: &mut devs,
                                    tables: &tables,
                                    tiers: &mut tiers,
                                    counts: vec![PathCounts::default(); 2],
                                    nvme_distinct: vec![0; 2],
                                    unrouted_fetch: vec![0; 2],
                                };
                                let o = emit_layer(
                                    &mut ems, &lw, &sts, &ins, &rows, &cc, &mut ar, &mut hook,
                                )?;
                                for r in 0..2 {
                                    rs.tier[r].resident += hook.counts[r].resident;
                                    rs.tier[r].ram += hook.counts[r].ram;
                                    rs.tier[r].nvme += hook.counts[r].nvme;
                                    rs.distinct[r] += hook.nvme_distinct[r];
                                }
                                o
                            }
                        };
                        let want = plan::declared_form_qsa(
                            &rules,
                            l,
                            2,
                            m,
                            saber_model::qwen4_exp::layer_graph::Form::Prefill,
                            ems[0].qsa_sparse(&cc[0]),
                            ems[0].moe_train,
                        );
                        ensure!(
                            outs[0].launches == want && outs[1].launches == want,
                            "L{l}: launches {}/{} vs declared {want}",
                            outs[0].launches,
                            outs[1].launches
                        );
                        let mut rsd = [0u64; 2];
                        if cap.is_some() {
                            for r in 0..2 {
                                rsd[r] = ems[r]
                                    .arena
                                    .take(u64::from(m) * W * 2, super::kern::Seed::None)?;
                                ems[r].materialize(&outs[r].pend, u64::from(m), rsd[r])?;
                            }
                        }
                        for r in 0..2 {
                            devs[r].exec(&tables[r], &mut ems[r])?;
                        }
                        if complete {
                            // the route ids ride to the host for the tier's counters after the
                            // chunk: an async D2D into the log (the ids buffer is the next
                            // layer's scratch), no sync inside the chunk
                            let base = devs[0].base();
                            hips[0].run_batch(&[Job::MemcpyD2d {
                                dst: &idlog,
                                dst_offset: u64::from(l) * u64::from(f.max_chunk()) * 40,
                                src: &devs[0].arena,
                                src_offset: outs[0].ids - base,
                                bytes: u64::from(m) * 40,
                                stream: &devs[0].stream,
                            }])?;
                        }
                        if let Some(c) = cap.as_mut() {
                            let mm = u64::from(m);
                            let a0 = a as usize;
                            let li = l as usize;
                            let g = |r: usize, x: u64, n: u64| {
                                d2h_in(&devs[r], &devs[r].arena, x, n as usize)
                            };
                            let r0 = (
                                g(0, rsd[0], mm * W * 2)?,
                                g(0, outs[0].att, mm * H * 2)?,
                                g(0, outs[0].pend.mo, mm * H * 2)?,
                                g(0, outs[0].ids, mm * 40)?,
                            );
                            let r1 = (
                                g(1, rsd[1], mm * W * 2)?,
                                g(1, outs[1].att, mm * H * 2)?,
                                g(1, outs[1].pend.mo, mm * H * 2)?,
                                g(1, outs[1].ids, mm * 40)?,
                            );
                            ensure!(
                                r0 == r1,
                                "L{l} {}: the ranks' resid/attn/mlp/ids bytes differ",
                                p.id
                            );
                            Cap::put(&mut c.resid, c.t, li, a0, W as usize * 2, &r0.0);
                            Cap::put(&mut c.att, c.t, li, a0, H as usize * 2, &r0.1);
                            Cap::put(&mut c.mlp, c.t, li, a0, H as usize * 2, &r0.2);
                            Cap::put(&mut c.ids, c.t, li, a0, 40, &r0.3);
                            Cap::put(
                                &mut c.rlog,
                                c.t,
                                li,
                                a0,
                                1024,
                                &g(0, outs[0].logits, mm * 1024)?,
                            );
                            if let Some(po) = outs[0].ple {
                                let b = g(0, po, mm * W * 2)?;
                                c.ple[a0 * W as usize * 2..a0 * W as usize * 2 + b.len()]
                                    .copy_from_slice(&b);
                            }
                        }
                        ins = outs.iter().map(|o| LayerIn::Pending(o.pend)).collect();
                    }
                    let pend: Vec<Pending> = ins
                        .iter()
                        .map(|i| match i {
                            LayerIn::Pending(p) => *p,
                            LayerIn::Embed(_) | LayerIn::Stream(_) => unreachable!(),
                        })
                        .collect();
                    let mut post = None;
                    if last || cap.is_some() {
                        let mut v = Vec::new();
                        for r in 0..2 {
                            v.push(emit_post(
                                &mut ems[r],
                                &fms[r],
                                heads[r],
                                &pend[r],
                                u64::from(m),
                                false,
                            )?);
                            devs[r].exec(&tables[r], &mut ems[r])?;
                        }
                        post = Some(v);
                    }
                    for r in 0..2 {
                        hips[r].record_event(&ev[r].1, &devs[r].stream)?;
                    }
                    host_ms += h0.elapsed().as_secs_f64() * 1e3;
                    // the next request's PLE rows, gathered while this chunk runs
                    if last && k + 1 < order.len() && prefetched.is_none() {
                        let np = &prompts[order[k + 1]];
                        let ids = ph.ids(PleContext::start(cfg.eos_token_id), &np.ids);
                        let g0 = Instant::now();
                        let c = ple
                            .gather(
                                &ids,
                                &mut staged[(k + 1) % 2].as_mut_slice()[..ids.len() * 320],
                            )
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        let ms = g0.elapsed().as_secs_f64() * 1e3;
                        prefetched = Some((
                            k + 1,
                            ids,
                            (c.distinct, c.cache_hits, c.nvme_reads, c.read_s),
                            ms,
                        ));
                    }
                    for r in 0..2 {
                        hips[r].sync_event(&ev[r].1)?;
                    }
                    wall_ms += f64::from(hips[0].elapsed_ms(&ev[0].0, &ev[0].1)?);
                    if let Some(v) = post {
                        if last {
                            let mut lg = Vec::new();
                            for r in 0..2 {
                                lg.extend(bf16_to_f32(&d2h_in(
                                    &devs[r],
                                    &devs[r].arena,
                                    v[r].1,
                                    (VOCAB as usize / 2) * 2,
                                )?));
                            }
                            logits = lg;
                        }
                        if let Some(c) = cap.as_mut() {
                            let fb = d2h_in(
                                &devs[0],
                                &devs[0].arena,
                                v[0].0,
                                (u64::from(m) * H * 2) as usize,
                            )?;
                            let o = a as usize * H as usize * 2;
                            c.fin[o..o + fb.len()].copy_from_slice(&fb);
                        }
                    }
                    // the tier's counters from the routes, then its LRU on the copy stream
                    for r in 0..2 {
                        tiers[r]
                            .mover
                            .after(&devs[r].stream)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    }
                    if complete {
                        let log = d2h_in(
                            &devs[0],
                            &idlog,
                            idlog.addr().as_u64(),
                            (LAYERS * f.max_chunk() * 40) as usize,
                        )?;
                        for l in 0..LAYERS as usize {
                            let at = l * f.max_chunk() as usize * 40;
                            let b = &log[at..at + m as usize * 40];
                            let routed: Vec<u32> = b
                                .chunks_exact(4)
                                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            for r in 0..2 {
                                let t = &mut tiers[r];
                                let c = t
                                    .tier
                                    .resolve(l as u32, &routed, &mut t.mover)
                                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                                ensure!(c.nvme == 0, "the complete tier read NVMe");
                                rs.tier[r].resident += c.resident;
                                rs.tier[r].ram += c.ram;
                            }
                        }
                    }
                    for t in &mut tiers {
                        t.tier
                            .end_step(&mut t.mover)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    }
                    if f.fn_op_timing {
                        for d in &mut devs {
                            let v = d.take_times()?;
                            if d.rank == 0 {
                                for (g, l, _, ms) in &v {
                                    let kind = if *l < 0 {
                                        "post".to_string()
                                    } else if *l as u32 == PLE_LAYER {
                                        "Gdn+PLE".to_string()
                                    } else {
                                        format!("{:?}", plan::kind(*l as u32))
                                    };
                                    *rs.groups
                                        .entry(format!("{kind}|{}", g.name()))
                                        .or_insert(0.0) += ms;
                                    *rs.groups.entry(format!("all|{}", g.name())).or_insert(0.0) +=
                                        ms;
                                    rs.busy_ms += ms;
                                }
                                times.extend(v);
                            }
                        }
                    }
                }
                let (top, v1, v2) = argmax(&logits);
                rs.top1 = top;
                rs.top1_logit = v1;
                rs.gap12 = v1 - v2;
                rs.wall_ms = wall_ms;
                rs.host_ms = host_ms;
                rs.ple_hidden = hidden;
                let _ = (t_req, ple_ms);
                rs.ttft_ms = group_t0.elapsed().as_secs_f64() * 1e3;
                rs.launches = devs[0].launches - launches0;
                if let Some(mut c) = cap {
                    c.logits = logits.clone();
                    cap_hashes.insert(p.id.clone(), c.hashes());
                    c.write(
                        &caps_dir
                            .join(format!(
                                "chunk{}{}",
                                f.fn_chunk,
                                if f.fn_gdn_state_f32 { "_f32" } else { "" }
                            ))
                            .join(&p.id),
                        p,
                        "saber engine FN12 (TP=2 / EP=2)",
                    )?;
                }
                println!(
                    "fnreq {key} {}: {} tok {} chunk(s) wall {:.2} ms host {:.2} ms ttft {:.2} ms | top1 {} (gap {:.3}) | tier r0 res {} ram {} nvme {} / r1 res {} ram {} nvme {} | ple distinct {} hits {} nvme {} ({}) | launches {}",
                    p.id,
                    rs.tokens,
                    rs.chunks,
                    rs.wall_ms,
                    rs.host_ms,
                    rs.ttft_ms,
                    rs.top1,
                    rs.gap12,
                    rs.tier[0].resident,
                    rs.tier[0].ram,
                    rs.tier[0].nvme,
                    rs.tier[1].resident,
                    rs.tier[1].ram,
                    rs.tier[1].nvme,
                    rs.ple.0,
                    rs.ple.1,
                    rs.ple.2,
                    if rs.ple_hidden { "gathered ahead" } else { "gathered inline" },
                    rs.launches
                );
                stats.push(rs);
            }
            all_stats.insert(key, stats);
        }
    }
    // ---- results
    let mut doc = serde_json::Map::new();
    for (k, v) in &all_stats {
        let rows: Vec<Value> = v
            .iter()
            .map(|s| {
                json!({"id": s.id, "tokens": s.tokens, "chunks": s.chunks, "wall_ms": s.wall_ms, "host_ms": s.host_ms,
                       "ttft_ms": s.ttft_ms, "top1": s.top1, "top1_logit": s.top1_logit, "gap12": s.gap12,
                       "tier": s.tier.iter().map(|c| json!({"resident": c.resident, "ram": c.ram, "nvme": c.nvme})).collect::<Vec<_>>(),
                       "ple": {"distinct": s.ple.0, "hits": s.ple.1, "nvme": s.ple.2, "read_s": s.ple.3, "ahead": s.ple_hidden},
                       "launches": s.launches, "demand_distinct": s.distinct, "busy_ms": s.busy_ms, "groups": s.groups})
            })
            .collect();
        doc.insert(k.clone(), Value::Array(rows));
    }
    let mut tsum: BTreeMap<String, f64> = BTreeMap::new();
    for (g, l, s, ms) in &times {
        let kind = if *l < 0 {
            "post".to_string()
        } else {
            format!(
                "{:?}{}",
                plan::kind(*l as u32),
                if *l as u32 == PLE_LAYER { "+PLE" } else { "" }
            )
        };
        *tsum.entry(format!("{}|{}", kind, g.name())).or_insert(0.0) += ms;
        *tsum.entry(format!("sym|{s}")).or_insert(0.0) += ms;
    }
    let (res_t, mov_t) = (tiers[0].tier.totals(), tiers[1].tier.totals());
    let full = json!({
        "mode": "engine", "tp": 2, "chunk": f.fn_chunk, "evict_all": f.fn_evict_all, "poison": f.fn_poison,
        "control": f.fn_control, "arena_bytes": dr.sizes.total(), "launches_per_chunk_2048": dr.per_layer.iter().sum::<u32>() + dr.post,
        "ar_calls": {"exact": pair.calls_exact, "wht6": pair.calls_wht6, "decode_exact": pair.calls_decode_exact},
        "runs": doc, "op_times_ms": tsum, "hashes": cap_hashes, "form": "one-process",
        "tier_totals": [format!("{:?}", res_t), format!("{:?}", mov_t)],
        "ple_totals": format!("{:?}", ple.totals()),
        "metrics": format!("{}{}{}", tiers[0].tier.render_prometheus(), tiers[1].tier.render_prometheus(), ple.render_prometheus()),
    });
    let name = format!(
        "engine_chunk{}{}{}{}{}{}.json",
        f.fn_chunk,
        if f.fn_gdn_state_f32 { "_f32" } else { "" },
        if f.fn_dump { "_cap" } else { "" },
        if f.fn_evict_all { "_evict" } else { "" },
        if f.fn_op_timing { "_optime" } else { "" },
        f.fn_control
            .as_deref()
            .map(|c| format!("_{c}"))
            .unwrap_or_default()
    );
    std::fs::write(out.join(&name), serde_json::to_string_pretty(&full)?)?;
    println!("fnresult: wrote {}", out.join(&name).display());
    println!(
        "{}",
        tiers[0]
            .tier
            .render_prometheus()
            .lines()
            .filter(|l| !l.contains("layer"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    drop(ems);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FN80d: the forced-miss count is over the rank's OWN expert range -- EP's half, all 512
    /// under `--fn-experts tp2` (the dims the emitters carry); the pre-FN80d range (`256 * rank`)
    /// is the control that undercounts tp2.
    #[test]
    fn fn80d_the_forced_miss_count_is_over_the_ranks_own_experts() {
        // one layer's route ids at 3 rows x 10: duplicates, both halves
        let routed: Vec<u32> = (0..30)
            .map(|i| (i * 37) % 512)
            .chain([0, 37, 300])
            .collect();
        let all = {
            let mut v = routed.clone();
            v.sort_unstable();
            v.dedup();
            v.len() as u64
        };
        let ep: Vec<u64> = (0..2)
            .map(|r| {
                let d = Dims::new(2, r);
                demand_distinct(&routed, d.e_lo, d.eloc)
            })
            .collect();
        assert_eq!(
            ep[0] + ep[1],
            all,
            "EP: the two ranks partition the distinct records"
        );
        assert!(ep[0] > 0 && ep[1] > 0);
        for r in 0..2 {
            let d = Dims::new(2, r).experts_tp2();
            assert_eq!(
                demand_distinct(&routed, d.e_lo, d.eloc),
                all,
                "tp2 rank {r}: every record"
            );
            // control: the pre-FN80d range counts only EP's half under tp2
            assert_ne!(demand_distinct(&routed, 256 * r, 256), all);
        }
        let t1 = Dims::new(1, 0);
        assert_eq!(
            demand_distinct(&routed, t1.e_lo, t1.eloc),
            all,
            "TP=1: all 512 local"
        );
    }

    #[test]
    fn fn57_block_and_the_device_pick_are_the_default_and_spin_host_the_control() {
        assert_eq!(DEFAULT_FN_HOST_WAIT, saber_exec::HostWait::Block);
        assert_eq!(host_wait_fn(None).unwrap(), saber_exec::HostWait::Block);
        assert_eq!(
            host_wait_fn(Some("spin")).unwrap(),
            saber_exec::HostWait::Spin
        );
        assert!(host_wait_fn(Some("poll")).is_err());
        assert_eq!(DEFAULT_FN_PICK, "device");
        assert_eq!(DEFAULT_FN_PIN, "off");
        assert_eq!(DEFAULT_FN_HOST_LEVERS, "pick");
        assert_eq!(
            crate::engine::DEFAULT_HOST_WAIT_27B,
            saber_exec::HostWait::Aimed
        );
    }

    #[test]
    fn the_host_levers_parse_by_name_and_refuse_an_unknown_one() {
        let all: HostLevers = "all".parse().unwrap();
        assert!(all.pick && all.ple && all.defer && all.begin);
        assert_eq!("none".parse::<HostLevers>().unwrap(), HostLevers::default());
        let two: HostLevers = "pick, defer".parse().unwrap();
        assert!(two.pick && !two.ple && two.defer && !two.begin);
        assert!("pick,fast".parse::<HostLevers>().is_err());
        // FN93: the default is the threaded pick alone; `none` is the control
        let d: HostLevers = DEFAULT_FN_HOST_LEVERS.parse().unwrap();
        assert!(d.pick && !d.ple && !d.defer && !d.begin);
        let c: HostLevers = "none".parse().unwrap();
        assert!(
            !c.pick && c != d,
            "the control differs from the default by the pick alone"
        );
    }

    #[test]
    fn the_rope_table_is_the_harness_table_below_2048_positions() {
        let t = rope_table(2048);
        assert_eq!(
            sha256_hex(&t),
            "a189d0427cd0cede4521ca9c00443a52e0c83bb5a7c5a9f5d3f6ab104b84ba23",
            "== fn6_ref.cos_sin_cache(65536)[:2048] (numpy, bf16 bits)"
        );
    }

    #[test]
    fn npy_header_is_64_aligned_and_parses() {
        // Under `target/`, removed when `_scratch` drops -- never `/tmp` (FN51h).
        let _scratch = saber_test_scratch::ScratchDir::new("fn12-npy");
        let d = _scratch.join("fn12-npy.npy");
        npy(&d, "<i4", &[2, 3], &[0u8; 24]).unwrap();
        let b = std::fs::read(&d).unwrap();
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        assert_eq!((10 + hl) % 64, 0);
        assert!(std::str::from_utf8(&b[10..10 + hl])
            .unwrap()
            .contains("'shape': (2, 3)"));
        std::fs::remove_file(&d).unwrap();
    }

    /// Parse `saber engine`-shaped Flash-Next flags (the real clap derive).
    #[test]
    fn the_slab4_mtp_experts_are_the_default_and_fp8_is_the_control() {
        // FN110b: the 4-bit slab is the default (the user's word 2026-10-12); fp8 = the control,
        // bf16 = FN50's control
        assert_eq!(fnx(&[]).mtp_experts(), "slab4");
        assert_eq!(DEFAULT_FN_MTP_EXPERTS, "slab4");
        assert_eq!(fnx(&["--fn-mtp-experts", "fp8"]).mtp_experts(), "fp8");
        assert_eq!(fnx(&["--fn-mtp-experts", "bf16"]).mtp_experts(), "bf16");
        assert_eq!(fnx(&["--fn-mtp-experts", "slab4"]).mtp_experts(), "slab4");
        // FN80d: `auto` resolves by the target's expert form; FN134: slab4 under tp2 too
        // (FN80d's fp8 = the control by name); explicit wins
        assert_eq!(fnx(&["--fn-experts", "tp2"]).mtp_experts(), "slab4");
        assert_eq!(DEFAULT_FN_MTP_EXPERTS_TP2, "slab4");
        assert_eq!(
            fnx(&["--fn-experts", "tp2", "--fn-mtp-experts", "fp8"]).mtp_experts(),
            "fp8"
        );
        assert_eq!(
            fnx(&["--fn-experts", "tp2", "--fn-mtp-experts", "slab4"]).mtp_experts(),
            "slab4"
        );
        assert_eq!(fnx(&["--fn-experts", "ep"]).mtp_experts(), "slab4");
        // FN138: `--fn-mtp-split` (ep = the default and the control; tp2 = the half slab)
        use super::super::exec::MtpX;
        assert_eq!(fnx(&[]).fn_mtp_split, "ep");
        assert_eq!(fnx(&[]).mtp_x().unwrap(), MtpX::Slab4);
        assert_eq!(fnx(&["--fn-experts", "tp2"]).mtp_x().unwrap(), MtpX::Slab4);
        assert_eq!(
            fnx(&["--fn-experts", "tp2", "--fn-mtp-split", "tp2"])
                .mtp_x()
                .unwrap(),
            MtpX::Slab4Tp2
        );
        assert!(fnx(&["--fn-mtp-split", "tp2"])
            .mtp_x()
            .unwrap_err()
            .to_string()
            .contains("--fn-experts tp2"));
        assert!(fnx(&[
            "--fn-experts",
            "tp2",
            "--fn-mtp-split",
            "tp2",
            "--fn-mtp-experts",
            "fp8"
        ])
        .mtp_x()
        .unwrap_err()
        .to_string()
        .contains("slab4"));
        assert!(fnx(&["--fn-mtp-split", "tp2"]).refuse_on_27b().is_err());
    }

    /// FN134: the slab4 class under tp2 bills the larger rank's slab on BOTH ranks (equal slot
    /// counts = the mirrored placement's premise); EP bills each rank's own (today's ledger). The
    /// control: the per-rank bill differs across the ranks (h1's 17,494 vs 17,496 slots).
    #[test]
    fn fn134_the_tp2_slab_bill_is_equal_on_both_ranks() {
        let b = [672_137_216u64, 668_991_488];
        assert_eq!(mtp_slab_bill(b, 0, false), b[0]);
        assert_eq!(mtp_slab_bill(b, 1, false), b[1]);
        assert_ne!(mtp_slab_bill(b, 0, false), mtp_slab_bill(b, 1, false));
        assert_eq!(mtp_slab_bill(b, 0, true), b[0]);
        assert_eq!(mtp_slab_bill(b, 1, true), b[0]);
        assert_eq!(mtp_slab_bill([1, 2], 0, true), 2);
    }

    /// FN69: prompt lookup is OFF by default (the control); `on` = n-grams 3..4, bar 3.
    #[test]
    fn spec_lookup_is_off_by_default_and_on_parses_the_law() {
        assert_eq!(DEFAULT_SPEC_LOOKUP, "off");
        assert_eq!(fnx(&[]).lookup().unwrap(), None);
        assert_eq!(
            fnx(&["--spec-lookup", "on"]).lookup().unwrap(),
            Some(saber_engine::LookupConfig::default())
        );
        let c = fnx(&[
            "--spec-lookup",
            "on",
            "--spec-lookup-ngram",
            "2..6",
            "--spec-lookup-min-match",
            "8",
        ])
        .lookup()
        .unwrap()
        .unwrap();
        assert_eq!((c.nmin, c.nmax, c.min_match), (2, 6, 8));
        assert!(fnx(&["--spec-lookup", "on", "--spec-lookup-ngram", "5..2"])
            .lookup()
            .is_err());
    }

    #[test]
    fn the_ple_overlap_defaults_to_layer1_inline_and_off_is_the_control() {
        assert_eq!(
            fnx(&[]).ple_overlap().unwrap(),
            Some(PleOverlap {
                pre: true,
                side: false
            })
        );
        assert_eq!(
            fnx(&["--fn-ple-overlap", "off"]).ple_overlap().unwrap(),
            None
        );
        let m = |v: &str| {
            fnx(&["--fn-ple-overlap", v])
                .ple_overlap()
                .unwrap()
                .unwrap()
        };
        assert_eq!(
            m("layer1"),
            PleOverlap {
                pre: true,
                side: true
            }
        );
        assert_eq!(
            m("layer1-l0only"),
            PleOverlap {
                pre: false,
                side: true
            }
        );
        assert_eq!(
            m("layer1-inline"),
            PleOverlap {
                pre: true,
                side: false
            }
        );
        assert_eq!(
            m("layer1-inline-l0only"),
            PleOverlap {
                pre: false,
                side: false
            }
        );
        assert!(fnx(&["--fn-ple-overlap", "layer0"]).ple_overlap().is_err());
        // FN64: the graph split
        assert_eq!(
            fnx(&[]).graph_split().unwrap(),
            GraphSplit::Lazy {
                q: 4,
                cache: DEFAULT_FN_GRAPH_LAZY_CACHE,
                pre: LazyPre::Recorded
            }
        );
        assert_eq!(
            fnx(&["--fn-graph-split", "layer"]).graph_split().unwrap(),
            GraphSplit::Layer
        );
        assert_eq!(
            fnx(&["--fn-graph-split", "step"]).graph_split().unwrap(),
            GraphSplit::Step
        );
        assert_eq!(
            fnx(&["--fn-graph-split", "lazy"]).graph_split().unwrap(),
            GraphSplit::Lazy {
                q: DEFAULT_FN_GRAPH_LAZY_Q,
                cache: DEFAULT_FN_GRAPH_LAZY_CACHE,
                pre: DEFAULT_FN_GRAPH_LAZY_PRE
            }
        );
        assert_eq!(
            fnx(&["--fn-graph-split", "lazy", "--fn-graph-lazy-q", "1"])
                .graph_split()
                .unwrap(),
            GraphSplit::Lazy {
                q: 1,
                cache: DEFAULT_FN_GRAPH_LAZY_CACHE,
                pre: DEFAULT_FN_GRAPH_LAZY_PRE
            }
        );
        assert_eq!(
            fnx(&["--fn-graph-split", "lazy", "--fn-graph-lazy-pre", "on"])
                .graph_split()
                .unwrap(),
            GraphSplit::Lazy {
                q: DEFAULT_FN_GRAPH_LAZY_Q,
                cache: DEFAULT_FN_GRAPH_LAZY_CACHE,
                pre: LazyPre::FirstBand
            }
        );
        assert!(
            fnx(&["--fn-graph-split", "lazy", "--fn-graph-lazy-pre", "x"])
                .graph_split()
                .is_err()
        );
        assert!(
            fnx(&["--fn-graph-split", "layer", "--fn-graph-lazy-pre", "on"])
                .graph_split()
                .is_err()
        );
        // FN64e: unbilled by default; the CHARGE arm -- the class and the plan's growth are the
        // same 1,556 MiB (704 pre-captured + 32 reserve + 42 drafter execs x 2 MiB: slots
        // unchanged), the LRU holds 736 runs, the pre-capture 704; `0` = unbilled explicitly
        let lz = fnx(&["--fn-graph-form", "lazy"]);
        assert_eq!(lz.graph_exec_bill().unwrap(), ExecBill::Unbilled);
        assert_eq!(lz.graph_exec_ledger().unwrap(), (0, 0));
        // FN98: `one` is the default form since h3 (`lazy` = the control): it bills its execs
        // (48 form + 16 drafter execs = 128 MiB) and grows the plan by that + the 600 MiB
        // measured free at the lazy boot; a lazy-only arm without the flag keeps `lazy`
        assert_eq!(DEFAULT_FN_GRAPH_FORM, "one");
        assert!(fnx(&[]).graph_form_one().unwrap());
        assert_eq!(fnx(&[]).graph_exec_bill().unwrap(), ExecBill::One);
        assert!(!lz.graph_form_one().unwrap());
        assert!(!fnx(&["--fn-graph-split", "layer"])
            .graph_form_one()
            .unwrap());
        let one = fnx(&["--fn-graph-form", "one"]);
        assert!(one.graph_form_one().unwrap());
        assert_eq!(one.graph_exec_bill().unwrap(), ExecBill::One);
        assert_eq!(one.graph_exec_ledger().unwrap(), (728 << 20, 128 << 20));
        assert_eq!(ExecBill::One.run_cap(), None);
        assert!(fnx(&["--fn-graph-form", "x"]).graph_form_one().is_err());
        let open = fnx(&["--fn-graph-form", "one-open"]);
        assert!(open.graph_form_one().unwrap() && open.graph_form_open());
        assert!(!one.graph_form_open());
        assert!(
            fnx(&["--fn-graph-form", "one", "--fn-graph-split", "layer"])
                .graph_form_one()
                .is_err()
        );
        assert!(fnx(&["--fn-graph-form", "one", "--fn-graph-exec-charge"])
            .graph_exec_bill()
            .is_err());
        assert!(fnx(&[
            "--fn-graph-form",
            "one",
            "--fn-graph-exec-budget-mib",
            "600"
        ])
        .graph_exec_bill()
        .is_err());
        assert!(
            fnx(&["--fn-graph-form", "one", "--fn-graph-lazy-record", "/tmp/x"])
                .graph_form_one()
                .is_err()
        );
        assert_eq!(FN64E_CHARGE_MIB, 1556);
        let c = fnx(&["--fn-graph-exec-charge"]);
        assert_eq!(c.graph_exec_bill().unwrap(), ExecBill::Charge);
        assert_eq!(c.graph_exec_ledger().unwrap(), (1556 << 20, 1556 << 20));
        assert_eq!(
            c.graph_split().unwrap(),
            GraphSplit::Lazy {
                q: DEFAULT_FN_GRAPH_LAZY_Q,
                cache: 736,
                pre: LazyPre::Recorded
            }
        );
        assert_eq!(ExecBill::Charge.run_cap(), Some(736));
        assert_eq!(ExecBill::Charge.pre_cap(), Some(704));
        assert!(fnx(&[
            "--fn-graph-exec-charge",
            "--fn-graph-exec-budget-mib",
            "600"
        ])
        .graph_exec_bill()
        .is_err());
        assert!(
            fnx(&["--fn-graph-split", "layer", "--fn-graph-exec-charge"])
                .graph_exec_bill()
                .is_err()
        );
        let u = fnx(&["--fn-graph-exec-budget-mib", "0"]);
        assert_eq!(u.graph_exec_bill().unwrap(), ExecBill::Unbilled);
        assert_eq!(u.graph_exec_ledger().unwrap(), (0, 0));
        assert_eq!(
            u.graph_split().unwrap(),
            GraphSplit::Lazy {
                q: DEFAULT_FN_GRAPH_LAZY_Q,
                cache: DEFAULT_FN_GRAPH_LAZY_CACHE,
                pre: LazyPre::Recorded
            }
        );
        // not lazy: unbilled, whatever the default
        assert_eq!(
            fnx(&["--fn-graph-split", "layer"])
                .graph_exec_ledger()
                .unwrap(),
            (0, 0)
        );
        // FN64d: the billed exec budget -- the cap is the budget's 2 MiB execs, the plan grows by
        // today's 704 unbilled execs, bounds refused
        let b = fnx(&["--fn-graph-exec-budget-mib", "600"]);
        assert_eq!(b.graph_exec_ledger().unwrap(), (1408 << 20, 600 << 20));
        assert_eq!(
            b.graph_split().unwrap(),
            GraphSplit::Lazy {
                q: DEFAULT_FN_GRAPH_LAZY_Q,
                cache: 300,
                pre: LazyPre::Recorded
            }
        );
        assert!(fnx(&["--fn-graph-exec-budget-mib", "1410"])
            .graph_exec_ledger()
            .is_err());
        assert!(fnx(&["--fn-graph-exec-budget-mib", "2"])
            .graph_exec_ledger()
            .is_err());
        assert!(fnx(&[
            "--fn-graph-split",
            "layer",
            "--fn-graph-exec-budget-mib",
            "600"
        ])
        .graph_exec_ledger()
        .is_err());
        // FN64c: the recorded set
        assert_eq!(
            fnx(&[
                "--fn-graph-split",
                "lazy",
                "--fn-graph-lazy-pre",
                "recorded"
            ])
            .graph_split()
            .unwrap(),
            GraphSplit::Lazy {
                q: DEFAULT_FN_GRAPH_LAZY_Q,
                cache: DEFAULT_FN_GRAPH_LAZY_CACHE,
                pre: LazyPre::Recorded
            }
        );
        assert!(
            fnx(&["--fn-graph-split", "layer", "--fn-graph-lazy-record", "/x"])
                .graph_split()
                .is_err()
        );
        assert!(
            fnx(&["--fn-graph-lazy-pre", "off", "--fn-graph-lazy-set", "/x"])
                .graph_split()
                .is_err()
        );
        let rec = fnx(&[
            "--fn-graph-split",
            "lazy",
            "--fn-graph-lazy-pre",
            "recorded",
            "--fn-graph-lazy-set",
            "/x",
            "--fn-graph-lazy-record",
            "/y",
        ]);
        assert!(rec.graph_split().is_ok());
        assert_eq!(rec.lazy_set_dir(Path::new("/p")), Path::new("/x"));
        assert_eq!(
            fnx(&["--fn-graph-split", "lazy"]).lazy_set_dir(Path::new("/p")),
            Path::new("/p/profile")
        );
        assert_eq!(
            lazy_set_file(Path::new("/p/profile"), 1),
            Path::new("/p/profile/lazy_runs.rank1.json")
        );
        assert!(fnx(&["--fn-graph-split", "lazy", "--fn-graph-lazy-q", "0"])
            .graph_split()
            .is_err());
        assert!(fnx(&["--fn-graph-split", "block"]).graph_split().is_err());
        assert!(
            fnx(&["--fn-graph-split", "layer", "--fn-graph-lazy-q", "2"])
                .graph_split()
                .is_err()
        );
        assert!(fnx(&["--fn-graph-split", "lazy"]).refuse_on_27b().is_err());
        assert!(fnx(&["--fn-ple-overlap", "layer1"])
            .refuse_on_27b()
            .is_err());
        assert!(fnx(&["--fn-ple-overlap", "layer1-inline"])
            .refuse_on_27b()
            .is_err());
    }

    /// FN55c: `--fn-vision` takes `on` (also the bare flag, FN55b's form), `lazy`, `off`; an
    /// absent flag is [`DEFAULT_FN_VISION`] where it can stand (the worker form, no poison, FN8's
    /// tier law, a pack with the member) and `off` elsewhere; the idle clock defaults to 300 s.
    #[test]
    fn fn55c_the_vision_forms_and_the_default_law() {
        let none = Path::new("/nonexistent-pack");
        assert_eq!(fnx(&["--fn-vision"]).vision_mode(none), VisionMode::On);
        assert_eq!(
            fnx(&["--fn-vision", "on"]).vision_mode(none),
            VisionMode::On
        );
        assert_eq!(
            fnx(&["--fn-vision", "lazy"]).vision_mode(none),
            VisionMode::Lazy
        );
        assert_eq!(
            fnx(&["--fn-vision", "off"]).vision_mode(none),
            VisionMode::Off
        );
        assert!(
            <FnFlagsProbe as clap::Parser>::try_parse_from(["saber", "--fn-vision", "maybe"])
                .is_err()
        );
        // absent: the default where it can stand, else off (no member in a missing pack)
        assert_eq!(fnx(&[]).vision_mode(none), VisionMode::Off);
        let scratch = saber_test_scratch::ScratchDir::new("saber-fn55c-vmode");
        let pack = scratch.path();
        std::fs::create_dir_all(pack.join(super::super::vision::DIR)).unwrap();
        std::fs::write(
            pack.join(super::super::vision::DIR).join("rank0.pack"),
            b"x",
        )
        .unwrap();
        let want = match DEFAULT_FN_VISION {
            "lazy" => VisionMode::Lazy,
            "on" => VisionMode::On,
            _ => VisionMode::Off,
        };
        assert_eq!(fnx(&[]).vision_mode(pack), want);
        assert_eq!(
            fnx(&["--fn-one-process"]).vision_mode(pack),
            VisionMode::Off
        );
        assert_eq!(fnx(&["--xtier-defer"]).vision_mode(pack), VisionMode::Off);
        assert_eq!(
            fnx(&["--fn-poison", "165"]).vision_mode(pack),
            VisionMode::Off
        );
        assert_eq!(fnx(&[]).fn_vision_idle_s, 300);
    }

    #[derive(clap::Parser)]
    struct FnFlagsProbe {
        #[command(flatten)]
        f: FnFlags,
    }

    #[test]
    fn fn53b_the_mem_valve_is_off_by_default_and_an_arm() {
        let d = fnx(&[]);
        assert_eq!(d.fn_mem_valve, "off");
        assert_eq!(d.fn_mem_valve_max_tail, DEFAULT_FN_MEM_VALVE_MAX_TAIL);
        assert_eq!(DEFAULT_FN_MEM_VALVE_MAX_TAIL, 1024);
        let on = fnx(&["--fn-mem-valve", "on", "--fn-mem-valve-max-tail", "512"]);
        assert_eq!(
            (on.fn_mem_valve.as_str(), on.fn_mem_valve_max_tail),
            ("on", 512)
        );
        assert_eq!(FN_MEM_VALVE_BOUNCE % 2, 0);
    }

    /// FN114b: `--fn-ar oneshot` is the control (FN114d made `gp` the default), `ll` the
    /// flag-in-data arm; a 27B boot refuses a non-default form; the form's window and counters sit
    /// clear of every other channel.
    #[test]
    fn fn114b_ar_oneshot_is_the_control_and_ll_is_the_arm() {
        use super::super::dev::{
            hcrr_scratch_off, ll_scratch_off, rp_scratch_off, LL_MAX_BLOCKS, LL_PAYLOAD, LL_SEQ0,
            LL_TAIL_OFF,
        };
        assert_eq!(DEFAULT_FN_AR, "gp");
        assert!(!ar_ll(&fnx(&[])));
        assert!(ar_ll(&fnx(&["--fn-ar", "ll"])));
        assert!(!ar_ll(&fnx(&["--fn-ar", "oneshot"])));
        assert!(fnx(&[]).refuse_on_27b().is_ok());
        assert!(fnx(&["--fn-ar", "ll"]).refuse_on_27b().is_err());
        assert!(fnx(&["--fn-ar", "oneshot"]).refuse_on_27b().is_err());
        for mb in [16u64 << 20, 32 << 20, 64 << 20] {
            // FN85's window ends 320 KiB above its start; the form's payload + tail inside the slot
            assert!(ll_scratch_off(mb) >= rp_scratch_off(mb) + (320 << 10));
            assert!(rp_scratch_off(mb) > hcrr_scratch_off(mb));
            assert!(ll_scratch_off(mb) + LL_TAIL_OFF + (8 << 10) <= mb);
        }
        // 32 counters from word 448 inside the 512-word seq region (FN85's ctl is 384..386)
        const { assert!(LL_SEQ0 > 386 && LL_SEQ0 + LL_MAX_BLOCKS <= 512) };
        // every exact-band reduce fits: < 128 KiB, and a decode form of <= 64 rows
        const { assert!(LL_PAYLOAD >= 128 << 10) };
    }

    /// FN116: `--fn-first-step kv` (the K-side front + a steady round) IS THE DEFAULT since h5 + h6,
    /// `pass` (the m-row prefill-form pass) the control; a 27B boot refuses the control flag.
    #[test]
    fn fn116_first_step_kv_is_the_default_and_pass_the_control() {
        // FN119f: `dev` (a `kv` form) IS THE DEFAULT; `kv` its control
        assert_eq!(DEFAULT_FN_FIRST_STEP, "dev");
        assert!(first_step_kv(&fnx(&[])) && first_step_dev(&fnx(&[])));
        assert!(!first_step_dev(&fnx(&["--fn-first-step", "kv"])));
        assert!(first_step_kv(&fnx(&["--fn-first-step", "kv"])));
        assert!(!first_step_kv(&fnx(&["--fn-first-step", "pass"])));
        // FN119e: `kvnext` = `kv` + the first round's ids pending
        assert!(first_step_kv(&fnx(&["--fn-first-step", "kvnext"])));
        assert!(first_step_defer(&fnx(&["--fn-first-step", "kvnext"])));
        assert!(!first_step_defer(&fnx(&["--fn-first-step", "kv"])));
        let c = fnx(&["--fn-first-step", "chunk"]);
        assert!(first_step_kv(&c) && first_step_defer(&c) && first_step_front(&c));
        assert!(!first_step_front(&fnx(&["--fn-first-step", "kvnext"])));
        let d = fnx(&["--fn-first-step", "dev"]);
        assert!(first_step_kv(&d) && first_step_defer(&d) && first_step_front(&d));
        assert!(first_step_dev(&d) && !first_step_dev(&c));
        // FN119g: `under` = `dev` + the front under the chunk's head
        let u = fnx(&["--fn-first-step", "under"]);
        assert!(first_step_kv(&u) && first_step_defer(&u) && first_step_front(&u));
        assert!(first_step_dev(&u) && first_step_under(&u) && !first_step_under(&d));
        assert!(fnx(&["--fn-first-step", "pass"]).refuse_on_27b().is_err());
        assert!(fnx(&[]).refuse_on_27b().is_ok());
    }

    /// FN114d: `--fn-ar gp` (the global-parity twins) IS THE DEFAULT and `oneshot` the control; its
    /// counter / flag words sit clear of every other channel's and cover the widest grid either
    /// form takes.
    #[test]
    fn fn114d_ar_gp_is_the_default_and_its_words_are_its_own() {
        use super::super::dev::{
            fake_pair, ArRules, GP_EXACT_SYMBOL, GP_NC, GP_WHT6_SYMBOL, GP_WORD0, HCRR_FLAG0,
            LL_SEQ0, RP_FLAG0,
        };
        assert_eq!(DEFAULT_FN_AR, "gp");
        assert!(ar_gp(&fnx(&[])));
        assert!(ar_gp(&fnx(&["--fn-ar", "gp"])));
        assert!(!ar_ll(&fnx(&["--fn-ar", "gp"])));
        assert!(!ar_gp(&fnx(&["--fn-ar", "oneshot"])));
        assert!(!ar_gp(&fnx(&["--fn-ar", "ll"])));
        // the shipped one-shot's words are 0..96 (exact <= 24 blocks, wht6 <= 96); FN37's from
        // 256, FN85's from 384, FN114b's seq from 448: the gp words are 128..224
        let r = ArRules::prod().unwrap();
        const { assert!(GP_WORD0 >= 96 && GP_WORD0 + GP_NC <= HCRR_FLAG0) };
        const { assert!(HCRR_FLAG0 < RP_FLAG0 && RP_FLAG0 < LL_SEQ0) };
        assert!(r.max_nb <= GP_NC && r.max_nb_exact <= GP_NC);
        assert!(
            GP_WORD0 + GP_NC <= r.max_blocks,
            "inside the flag / seq allocations"
        );
        assert!(GP_EXACT_SYMBOL.starts_with("saber_ar_oneshot_2rank_exact_bf16"));
        assert!(GP_WHT6_SYMBOL.starts_with("saber_ar_oneshot_2rank_wht6_bf16"));
        assert!(
            !fake_pair().unwrap().gp,
            "the launch-count tests keep the shipped symbols"
        );
    }

    /// FN137 / FN145: `--fn-moe-ar-push on` is the default (flipped by FN145's served holds),
    /// `off` (the standalone MoE-out reduce) the control; a 27B boot refuses the flag; the fused channel's flag / counter /
    /// diagnostic words and its window sit clear of every other channel and inside the
    /// allocations; the kernel's by-value `FarArgs` is the ABI's 72-byte struct, field for field.
    #[test]
    fn fn137_moe_ar_push_is_an_arm_and_its_channel_is_its_own() {
        use super::super::dev::{
            far_scratch_off, far_symbol, ll_scratch_off, ArRules, FAR_CTR0, FAR_DIAG0, FAR_FLAG0,
            FAR_PAYLOAD, FAR_SLICES, LL_MAX_BLOCKS, LL_SEQ0, LL_TAIL_OFF, RP_FLAG0,
        };
        use super::super::exec::FarChan;
        use saber_kernels::abi::{pack_fields_at, FAR_ARGS_BYTES, STRUCT_FAR_ARGS};
        assert_eq!(DEFAULT_FN_MOE_AR_PUSH, "on");
        assert!(moe_ar_push(&fnx(&[])));
        assert!(moe_ar_push(&fnx(&["--fn-moe-ar-push", "on"])));
        assert!(!moe_ar_push(&fnx(&["--fn-moe-ar-push", "off"])));
        assert!(fnx(&["--fn-moe-ar-push", "off"]).refuse_on_27b().is_err());
        // flags 448..468 of ar_flags (FN85's are 384..392; FN114b keeps its flags in its window),
        // counter + diagnostics 480..485 of ar_seq (FN114b's seq is 448..480)
        let r = ArRules::prod().unwrap();
        const { assert!(FAR_FLAG0 >= RP_FLAG0 + 8 && FAR_FLAG0 + FAR_SLICES <= FAR_CTR0) };
        const { assert!(FAR_CTR0 >= LL_SEQ0 + LL_MAX_BLOCKS && FAR_DIAG0 == FAR_CTR0 + 2) };
        assert!(
            FAR_DIAG0 + 3 <= r.max_blocks,
            "inside the 512-word allocations"
        );
        assert_eq!(FAR_SLICES * 128, super::super::exec::H);
        for mb in [16u64 << 20, 32 << 20, 64 << 20] {
            assert!(far_scratch_off(mb) >= ll_scratch_off(mb) + LL_TAIL_OFF + (8 << 10));
            assert!(far_scratch_off(mb) + FAR_PAYLOAD <= mb, "inside the slot");
            // FN137b h1: clear of FN119's issue-ahead seam window (the served device step)
            let (w0, w1) = (
                super::super::dstep::win_off(mb),
                super::super::dstep::win_off(mb) + super::super::dstep::WIN_BYTES,
            );
            assert!(far_scratch_off(mb) >= w1 || far_scratch_off(mb) + FAR_PAYLOAD <= w0);
            // control: the h1 offset (FN114b's window + 512 KiB) IS FN119's window
            assert_eq!(ll_scratch_off(mb) + (512 << 10), w0);
        }
        assert_eq!(far_symbol(4, 2), "fn137_moe_back_h2_far_gu4_dn2_p1");
        let ch = FarChan {
            peer: 1,
            my: 2,
            pflags: 3,
            mflags: 4,
            ctr: 5,
            diag: 7,
            stride16: 1 << 20,
        };
        let b = ch.blob(6);
        assert_eq!(b.len() as u32, FAR_ARGS_BYTES);
        let (offs, end) = pack_fields_at(STRUCT_FAR_ARGS, 0);
        assert_eq!(end, FAR_ARGS_BYTES);
        let v = |o: u32, n: u32| -> u64 {
            let mut x = [0u8; 8];
            x[..n as usize].copy_from_slice(&b[o as usize..(o + n) as usize]);
            u64::from_le_bytes(x)
        };
        let got: Vec<u64> = offs.iter().map(|&(o, n)| v(o, n)).collect();
        assert_eq!(got, vec![1, 2, 3, 4, 5, 6, 7, 0, 1 << 20, 0]);
    }

    /// FN164: `--fn-mixer-ar-push on` is the default (FN164 h1's flip), `off` (the standalone
    /// mixer-out reduce) the control, `b1` an arm; a 27B boot refuses a non-default value; the fused mixer channel's flag /
    /// counter / diagnostic words and its window sit clear of EVERY other channel (the shipped
    /// one-shot 0..96, FN114d 128..224, FN37 256..320, FN85 384..392, FN114b's seq 448..480,
    /// FN137's flags 448..468 + counter / diagnostics 480..485, FN119's seam window, FN137's
    /// window) and inside the allocations; every fused symbol takes the ABI's T8 + `FarArgs`
    /// signature and is in the shipped object's manifest at the parent's kernarg + 72 bytes.
    #[test]
    fn fn164_mixer_ar_push_is_an_arm_and_its_channel_is_its_own() {
        use super::super::dev::{
            far_scratch_off, hcrr_scratch_off, ll_scratch_off, mix_scratch_off, mix_symbol,
            rp_scratch_off, ArRules, MixPush, FAR_CTR0, FAR_DIAG0, FAR_FLAG0, FAR_PAYLOAD,
            FAR_SLICES, GP_NC, GP_WORD0, HCRR_FLAG0, HCRR_MAX_ROWS, LL_MAX_BLOCKS, LL_SEQ0,
            MIX_CTR0, MIX_DIAG0, MIX_FLAG0, MIX_PARENTS, MIX_PAYLOAD, MIX_WGS, RP_FLAG0,
        };
        use saber_kernels::abi::{signature_for, SIG_FN164_T8_FAR, SIG_FN_T8_A8W8_LDC};
        assert_eq!(DEFAULT_FN_MIXER_AR_PUSH, "on");
        assert_eq!(mixer_ar_push(&fnx(&[])), MixPush::On);
        assert_eq!(
            mixer_ar_push(&fnx(&["--fn-mixer-ar-push", "on"])),
            MixPush::On
        );
        assert_eq!(
            mixer_ar_push(&fnx(&["--fn-mixer-ar-push", "b1"])),
            MixPush::B1
        );
        assert_eq!(
            mixer_ar_push(&fnx(&["--fn-mixer-ar-push", "off"])),
            MixPush::Off
        );
        assert!(fnx(&["--fn-mixer-ar-push", "off"]).refuse_on_27b().is_err());
        assert!(fnx(&["--fn-mixer-ar-push", "b1"]).refuse_on_27b().is_err());
        // ar_flags: [0, 96) shipped, [128, 224) gp, [256, 320) hcrr, [384, 392) rp, [448, 468) far
        let flags = [
            (0u64, 96u64),
            (GP_WORD0, GP_WORD0 + GP_NC),
            (HCRR_FLAG0, HCRR_FLAG0 + HCRR_MAX_ROWS),
            (RP_FLAG0, RP_FLAG0 + 8),
            (FAR_FLAG0, FAR_FLAG0 + FAR_SLICES),
        ];
        // ar_seq: the same + ll [448, 480), far counter + diagnostics [480, 486)
        let seqs = [
            (0u64, 96u64),
            (GP_WORD0, GP_WORD0 + GP_NC),
            (HCRR_FLAG0, HCRR_FLAG0 + HCRR_MAX_ROWS),
            (RP_FLAG0, RP_FLAG0 + 2),
            (LL_SEQ0, LL_SEQ0 + LL_MAX_BLOCKS),
            (FAR_CTR0, FAR_DIAG0 + 4),
        ];
        let clear =
            |a: u64, b: u64, used: &[(u64, u64)]| used.iter().all(|&(x, y)| b <= x || a >= y);
        let r = ArRules::prod().unwrap();
        assert!(
            clear(MIX_FLAG0, MIX_FLAG0 + MIX_WGS, &flags),
            "the flags' words"
        );
        assert!(
            clear(MIX_CTR0, MIX_CTR0 + MIX_WGS, &seqs),
            "the counters' words"
        );
        assert!(
            clear(MIX_DIAG0, MIX_DIAG0 + 4, &seqs),
            "the diagnostics' words"
        );
        assert!(
            !clear(FAR_FLAG0 + FAR_SLICES - 1, FAR_FLAG0 + FAR_SLICES, &flags),
            "control: FN137's last flag is used"
        );
        assert!(
            MIX_FLAG0 + MIX_WGS <= r.max_blocks,
            "the flags inside the 512-word allocation"
        );
        assert!(
            MIX_DIAG0 + 4 <= r.max_blocks,
            "the diagnostics inside the 512-word allocation"
        );
        assert!(clear(
            MIX_CTR0,
            MIX_CTR0 + MIX_WGS,
            &[(MIX_DIAG0, MIX_DIAG0 + 4)]
        ));
        assert_eq!(MIX_WGS * 64, super::super::exec::H);
        for mb in [16u64 << 20, 32 << 20, 64 << 20] {
            let w = mix_scratch_off(mb);
            let win = [
                (
                    hcrr_scratch_off(mb),
                    hcrr_scratch_off(mb) + HCRR_MAX_ROWS * super::super::exec::H * 2,
                ),
                (rp_scratch_off(mb), rp_scratch_off(mb) + (320 << 10)),
                (ll_scratch_off(mb), ll_scratch_off(mb) + (420 << 10)),
                (
                    super::super::dstep::win_off(mb),
                    super::super::dstep::win_off(mb) + super::super::dstep::WIN_BYTES,
                ),
                (far_scratch_off(mb), far_scratch_off(mb) + FAR_PAYLOAD),
            ];
            assert!(
                clear(w, w + MIX_PAYLOAD, &win),
                "mb {mb}: the window is its own"
            );
            assert!(w + MIX_PAYLOAD <= mb, "mb {mb}: inside the slot");
            // control: FN137's window is not clear of itself
            assert!(!clear(far_scratch_off(mb), far_scratch_off(mb) + 1, &win));
        }
        let man = saber_kernels::manifest::Manifest::load(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../prebuilt/gfx1201/flashnext_t8far.manifest.toml"),
        )
        .unwrap();
        for parent in MIX_PARENTS {
            assert_eq!(signature_for(parent), Some(SIG_FN_T8_A8W8_LDC), "{parent}");
            let far = mix_symbol(parent).unwrap();
            for suf in ["", "_ctl", "_chk"] {
                let s = format!("{far}{suf}");
                assert_eq!(signature_for(&s), Some(SIG_FN164_T8_FAR), "{s}");
                let k = man
                    .kernel
                    .iter()
                    .find(|k| k.name == s)
                    .unwrap_or_else(|| panic!("{s} in the manifest"));
                assert_eq!(
                    k.kernarg_segment_size,
                    56 + 72,
                    "{s}: the parent's 56 B + FarArgs"
                );
                assert_eq!(k.private_segment_fixed_size, 0, "{s}: no scratch");
            }
        }
        assert!(
            mix_symbol("dflash_gemm_a8w8_bs128_m16n128k128").is_none(),
            "the m16 cell has no twin"
        );
        assert!(signature_for("fn164_delay").is_some());
    }

    /// FN128: `--fn-sel128 all` (top-k + union on `flashnext_qsa128`) is the default, `default` (the
    /// `flashnext_qsa` parents) the control, a list picks.
    #[test]
    fn fn128_sel128_all_is_the_default_and_default_is_the_control() {
        use saber_kernels::flashnext::sel128 as s128;
        assert_eq!(sel128(&fnx(&[])).unwrap(), s128::ALL);
        assert_eq!(s128::ALL, s128::TOPK | s128::UNION);
        assert_eq!(sel128(&fnx(&["--fn-sel128", "default"])).unwrap(), 0);
        assert_eq!(sel128(&fnx(&["--fn-sel128", "topk"])).unwrap(), s128::TOPK);
        assert!(sel128(&fnx(&["--fn-sel128", "x"])).is_err());
    }

    /// FN131: `--fn-hc-down split` is the default (flipped by the served hold h1); `none` = the
    /// shipped downs (the control).
    #[test]
    fn fn131_hc_down_split_is_the_default_and_none_the_control() {
        use saber_kernels::flashnext::down131 as d131;
        assert_eq!(down131(&fnx(&[])).unwrap(), d131::SPLIT);
        assert_eq!(down131(&fnx(&["--fn-hc-down", "none"])).unwrap(), 0);
        assert_eq!(
            down131(&fnx(&["--fn-hc-down", "split"])).unwrap(),
            d131::SPLIT
        );
        assert_eq!(down131(&fnx(&["--fn-hc-down", "all"])).unwrap(), d131::ALL);
        assert!(down131(&fnx(&["--fn-hc-down", "t1"])).is_err());
    }

    /// FN141: `--fn-k141 all` is the default (flipped by the served hold h1); `none` = the shipped
    /// symbols (the control); a list picks families.
    #[test]
    fn fn141_k141_all_is_the_default_and_none_the_control() {
        use saber_kernels::flashnext::k141 as k;
        assert_eq!(k141(&fnx(&[])).unwrap(), k::ALL);
        assert_eq!(k141(&fnx(&["--fn-k141", "none"])).unwrap(), 0);
        assert_eq!(k141(&fnx(&["--fn-k141", "all"])).unwrap(), k::ALL);
        assert_eq!(k141(&fnx(&["--fn-k141", "up"])).unwrap(), k::UP);
        assert_eq!(
            k141(&fnx(&["--fn-k141", "t8,router"])).unwrap(),
            k::T8 | k::ROUTER
        );
        assert!(k141(&fnx(&["--fn-k141", "sfx"])).is_err());
        assert_eq!(
            boot_rules(&fnx(&["--fn-k141", "all"])).unwrap().hc.k141,
            k::ALL
        );
    }

    /// FN163: `--fn-route163 none` is the default (the control) until the served hold flips it.
    #[test]
    fn fn163_route163_none_is_the_default_and_all_the_arm() {
        use saber_kernels::flashnext::route163 as r;
        assert_eq!(route163(&fnx(&[])).unwrap(), 0);
        assert_eq!(route163(&fnx(&["--fn-route163", "all"])).unwrap(), r::ALL);
        assert_eq!(route163(&fnx(&["--fn-route163", "b8"])).unwrap(), r::B8);
        assert_eq!(route163(&fnx(&["--fn-route163", "b1"])).unwrap(), r::B1);
        assert!(route163(&fnx(&["--fn-route163", "b4"])).is_err());
    }

    /// f6/f7 flip: the default `--fn-fuse` runs ATTN-OPT-3's f6 + f7; the old default (FN86d's
    /// list) and `none` leave both off.
    #[test]
    fn attn3_f6f7_in_the_default_fuse_set() {
        use saber_kernels::flashnext::fuse86;
        let d = fuse86(&fnx(&[])).unwrap();
        assert_eq!(d & fuse86::F6, fuse86::F6);
        assert_eq!(d & fuse86::F7, fuse86::F7);
        assert_eq!(fuse86::names(d), "f1,f2,f4,f5,f6,f7,f10");
    }

    /// f6/f7 flip control: `none` runs no family, and the previous default's explicit list runs
    /// neither f6 nor f7 (the flip hold's arm A).
    #[test]
    fn attn3_f6f7_none_and_old_list_are_the_controls() {
        use saber_kernels::flashnext::fuse86;
        assert_eq!(fuse86(&fnx(&["--fn-fuse", "none"])).unwrap(), 0);
        let a = fuse86(&fnx(&["--fn-fuse", "f1,f2,f4,f5,f10"])).unwrap();
        assert_eq!(a & (fuse86::F6 | fuse86::F7), 0);
        assert_eq!(a, fuse86::ALL & !(fuse86::F6 | fuse86::F7));
    }

    /// FN86b: `--fn-fuse all` is the default (every built family since the f6/f7 flip), `none` the
    /// control, a list picks.
    #[test]
    fn fn86b_fuse_all_is_the_default_and_none_is_the_control() {
        use saber_kernels::flashnext::fuse86;
        assert_eq!(fuse86(&fnx(&[])).unwrap(), fuse86::ALL);
        assert_eq!(fuse86(&fnx(&["--fn-fuse", "none"])).unwrap(), 0);
        assert_eq!(fuse86(&fnx(&["--fn-fuse", "f1"])).unwrap(), fuse86::F1);
        assert!(fuse86(&fnx(&["--fn-fuse", "f7"])).is_err());
        // FN86d: `all` includes family 10; FN86c's set is the control arm
        assert_eq!(
            fuse86(&fnx(&["--fn-fuse", "f1,f2,f4,f5"])).unwrap(),
            fuse86::ALL_FN86C
        );
        assert_eq!(fuse86::ALL_FN86D, fuse86::ALL_FN86C | fuse86::F10);
        // f6/f7 flip: the default runs ATTN-OPT-3's f6 + f7 (every built family); the previous
        // default is the explicit list `f1,f2,f4,f5,f10`, and a list still picks
        assert_eq!(fuse86::ALL, fuse86::ALL_FN86D | fuse86::F6 | fuse86::F7);
        assert_eq!(fuse86::ALL, fuse86::BUILT);
        assert_eq!(
            fuse86(&fnx(&[])).unwrap() & (fuse86::F6 | fuse86::F7),
            fuse86::F6 | fuse86::F7
        );
        assert_eq!(
            fuse86(&fnx(&["--fn-fuse", "f1,f2,f4,f5,f10"])).unwrap(),
            fuse86::ALL_FN86D
        );
        assert_eq!(
            fuse86(&fnx(&["--fn-fuse", "all,f6,f7"])).unwrap(),
            fuse86::ALL
        );
        assert_eq!(fuse86(&fnx(&["--fn-fuse", "f6"])).unwrap(), fuse86::F6);
        // FN111: `--fn-tiles all` is the default (t1 + t3), `default` = the shipped cells
        use saber_kernels::flashnext::tiles111;
        assert_eq!(tiles111(&fnx(&[])).unwrap(), tiles111::ALL);
        assert_eq!(tiles111(&fnx(&["--fn-tiles", "default"])).unwrap(), 0);
        assert_eq!(
            tiles111(&fnx(&["--fn-tiles", "all"])).unwrap(),
            tiles111::ALL
        );
        assert_eq!(
            tiles111(&fnx(&["--fn-tiles", "t1,fn111_moe_route_lds_dpp_nt512"])).unwrap(),
            tiles111::T1 | tiles111::T3
        );
        assert!(tiles111(&fnx(&["--fn-tiles", "t9"])).is_err());
        // FN122b: `--fn-prefill-tiles hc` is the default, `default` = the shipped cells (control)
        use saber_kernels::flashnext::prefill122 as p122;
        assert_eq!(prefill122(&fnx(&[])).unwrap(), p122::HC | p122::DQ);
        assert_eq!(p122::dq_min_rows(p122::HC | p122::DQ), 1024);
        assert_eq!(p122::dq_min_rows(p122::ALL), 513);
        assert_eq!(
            prefill122(&fnx(&["--fn-prefill-tiles", "hc"])).unwrap(),
            p122::HC
        );
        assert_eq!(
            prefill122(&fnx(&["--fn-prefill-tiles", "default"])).unwrap(),
            0
        );
        assert_eq!(
            prefill122(&fnx(&["--fn-prefill-tiles", "hc"])).unwrap(),
            p122::HC
        );
        assert_eq!(
            prefill122(&fnx(&["--fn-prefill-tiles", "all"])).unwrap(),
            p122::ALL
        );
        assert!(prefill122(&fnx(&["--fn-prefill-tiles", "w8"])).is_err());
        assert_eq!(
            prefill122(&fnx(&["--fn-prefill-tiles", "hc,dq,dqb"])).unwrap(),
            p122::HC | p122::DQ | p122::DQB
        );
        // FN124: `qsa` joins `all`; `qsatwin` is a diagnostic outside it
        assert_eq!(
            prefill122(&fnx(&["--fn-prefill-tiles", "hc,dq,dqb,qsa"])).unwrap(),
            p122::ALL
        );
        assert_eq!(p122::ALL & p122::QSATWIN, 0);
        assert_eq!(qsa124_class(&fnx(&[])), 0);
        assert_eq!(
            qsa124_class(&fnx(&["--fn-prefill-tiles", "hc,dq,qsatwin"])),
            QSA124_TWIN_BYTES
        );
        assert_eq!(p122::hc_cfg(125), None);
        assert!(p122::hc_cfg(2048).is_some() && p122::hc_cfg(8192).is_some());
        assert!(p122::hc_cfg(1023).is_none());
    }

    #[test]
    fn the_glue_fold_is_the_default_and_off_is_the_control() {
        // FN120b: `--fn-glue fold` is the default; `off` (per-layer fills + copies) the control
        // (FN120e h2 reverted it on a conc-3 deep-band cell, which is outside the byte law in
        // every arm; restored)
        assert_eq!(glue120(&fnx(&[])).unwrap(), Glue120::Fold);
        assert_eq!(glue120(&fnx(&["--fn-glue", "off"])).unwrap(), Glue120::Off);
        assert_eq!(
            glue_class(
                &fnx(&["--fn-glue", "off"]),
                super::super::exec::GlueRegion::fixed()
            ),
            0
        );
        assert_eq!(
            glue120(&fnx(&["--fn-glue", "seeds"])).unwrap(),
            Glue120::Seeds
        );
        assert_eq!(
            glue120(&fnx(&["--fn-glue", "fold"])).unwrap(),
            Glue120::Fold
        );
        let fixed = super::super::exec::GlueRegion::fixed();
        assert_eq!(
            glue_class(&fnx(&["--fn-glue", "fold"]), fixed),
            super::super::exec::GLUE_ZERO_BYTES + super::super::exec::GLUE_ONES_BYTES
        );
        // FN120e: `fold-strict` is `fold` with the loud refusal; a served boot spills instead
        assert_eq!(
            glue120(&fnx(&["--fn-glue", "fold-strict"])).unwrap(),
            Glue120::Fold
        );
        assert!(glue_strict(&fnx(&[
            "--fn-serve",
            "8099",
            "--fn-glue",
            "fold-strict"
        ])));
        assert!(!glue_strict(&fnx(&["--fn-serve", "8099"])));
        assert!(glue_strict(&fnx(&[])), "the harness / gates refuse loudly");
        // the gate control: absent = the census; a value forces the zero part (MiB)
        assert_eq!(fnx(&[]).fn_glue_region_mib, None);
        assert_eq!(
            fnx(&["--fn-glue-region-mib", "6"]).fn_glue_region_mib,
            Some(6)
        );
        assert!(glue120(&fnx(&["--fn-glue", "all"])).is_err());
    }

    #[test]
    fn a_glue_run_takes_distinct_seeds_and_restores_its_extent() {
        use super::super::exec::GlueSeeds;
        let mut g = GlueSeeds::new(1 << 40, true);
        g.begin(4).unwrap();
        // fold: the ids block (48 x 4 x 40 B) sits at the base; seeds follow it
        assert_eq!(g.ids_stride, 160);
        let z0 = g.take(100, false).unwrap().unwrap();
        let z1 = g.take(100, false).unwrap().unwrap();
        let o0 = g.take(64, true).unwrap().unwrap();
        assert!(
            z0 >= (1 << 40) + 48 * 160 && z1 > z0,
            "distinct zero seeds past the ids block"
        );
        assert_eq!(o0, (1 << 40) + super::super::exec::GLUE_ZERO_BYTES);
        let (zb, ob) = g.end().unwrap();
        assert!(zb >= z1 + 100 - (1 << 40) && zb % 256 == 0);
        assert_eq!(ob, 256);
        assert!(g.end().is_err(), "a run closes once");
        // a new run restarts both parts (the previous run restored them)
        g.begin(4).unwrap();
        assert_eq!(g.take(100, false).unwrap(), Some(z0));
        // seeds: no ids block
        let mut s = GlueSeeds::new(1 << 40, false);
        s.begin(4).unwrap();
        assert_eq!(s.ids_stride, 0);
        assert_eq!(s.take(8, false).unwrap(), Some(1 << 40));
    }

    /// FN120e: THE GLUE CENSUS of the served boot -- every decode / verify form (buckets 1/2/4/8
    /// x widths 1..5 x short / sparse x the 262,144 boot's bands) emitted on the host as ONE
    /// glue-fold run; prints each form's high-water, names the forms past FN120b's 6 + 2 MiB,
    /// and asserts every form fits the derived region. NEGATIVE CONTROL: FN120b's fixed region
    /// does NOT hold the served forms (the 10-14 13:35 crash: a b4 sparse form at ~55k).
    #[test]
    fn fn120e_the_glue_census_sizes_the_region_over_every_served_form() {
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let decls = Decls::load(&prebuilt).unwrap();
        let f = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-trunk",
            "w8-hc8-head",
            "--fn-glue",
            "fold",
        ])
        .resolved();
        let rules = boot_rules(&f).unwrap();
        let t = Instant::now();
        let c = glue_census(&decls, &rules, &f, &[], false).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let fixed = super::super::exec::GlueRegion::fixed();
        println!("fn120e census: {} forms in {ms:.0} ms", c.forms.len());
        for x in &c.forms {
            println!(
                "fn120e form {:<28} zero {:>10} ones {:>8}{}",
                x.name(),
                x.zero,
                x.ones,
                if x.zero > fixed.zero || x.ones > fixed.ones {
                    format!(
                        "  PAST FN120b's region (zero by {} B)",
                        x.zero.saturating_sub(fixed.zero)
                    )
                } else {
                    String::new()
                }
            );
        }
        println!("fn120e {}", c.line());
        let (max_ctx, pool_w) = super::super::worker::pools_geometry(&f, &[]).unwrap();
        assert_eq!(max_ctx, 262_144, "the served context");
        // b 1/2/4/8 x w 1..=pool_w x (short at the first band + sparse at the 5 bands)
        assert_eq!(c.forms.len(), 4 * pool_w as usize * 6);
        for x in &c.forms {
            assert!(
                x.zero <= c.region.zero && x.ones <= c.region.ones,
                "{} passes the derived region",
                x.name()
            );
        }
        // the law: the widest + one 2 MiB page, whole pages
        let (wz, wo) = (c.forms[c.widest_zero].zero, c.forms[c.widest_ones].ones);
        assert!(c.region.zero >= wz + super::super::exec::GLUE_MARGIN_BYTES);
        assert!(c.region.zero < wz + 2 * super::super::exec::GLUE_MARGIN_BYTES);
        assert!(c.region.ones >= wo + super::super::exec::GLUE_MARGIN_BYTES);
        assert_eq!(c.region.zero % (2 << 20), 0);
        // the widest form is the deepest band's sparse b8
        let w = &c.forms[c.widest_zero];
        assert!(!w.short && w.band == 262_144 && w.b == 8, "{}", w.name());
        // NEGATIVE CONTROL: FN120b's fixed 6 MiB zero part does not hold the served forms --
        // replayed in the fixed STRICT region, each such form refuses with the engine's message
        // (the 10-14 crash's text: `arena region over-committed: 6295968 B of 6291456 B`)
        let mut refused = Vec::new();
        for x in &c.forms {
            let cc = super::super::exec::ChunkConsts {
                m: x.b * x.w,
                b: x.b,
                w: x.w,
                decode: true,
                nb: x.band.div_ceil(16) + 1,
                max_ctx: x.band,
                qsa_short: x.short,
                ..super::super::exec::ChunkConsts::default()
            };
            let r = plan::dry_run_glue_at(
                &decls,
                &rules,
                cc,
                f.fn_verify_class,
                f.fn_moe_train,
                f.trunk8(),
                f.trunk8_head(),
                false,
                fixed,
            );
            assert_eq!(r.is_err(), x.zero > fixed.zero || x.ones > fixed.ones);
            if let Err(e) = r {
                println!("fn120e fixed-region refusal {:<28} {e:#}", x.name());
                refused.push((x.name(), format!("{e:#}")));
            }
        }
        assert!(
            !refused.is_empty(),
            "the fixed region would have held every form"
        );
        // the crash's own text, byte for byte: the b4 w3 sparse form at the 262,144 band
        let crash: Vec<&String> = refused
            .iter()
            .filter(|(_, e)| e.contains("over-committed: 6295968 B of 6291456 B"))
            .map(|(n, _)| n)
            .collect();
        assert_eq!(crash, ["b4 w3 sparse band 262144"]);
        // the ledger class follows the derived region
        let cls = exact_runtime(&f, &[], 0, 0, c.region).unwrap();
        let g = cls.iter().find(|x| x.name == "fn.glue120").unwrap().bytes;
        assert_eq!(g, c.region.total());
        // a 10,240 boot (one band) and SPEC 0 (width 1) size their own, smaller regions
        let s = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-max-ctx",
            "10240",
            "--fn-mtp",
            "0",
        ])
        .resolved();
        let cs = glue_census(&decls, &boot_rules(&s).unwrap(), &s, &[], false).unwrap();
        assert_eq!(cs.forms.len(), 4 * 2);
        assert!(cs.region.zero < c.region.zero);
    }

    /// FN120e: THE FALLBACK LAW -- a served (non-strict) region spills the run that would pass
    /// it (that seed and every later one go to the arena's per-layer fills), counted; the seeds
    /// taken before stay in the region and the restore covers them; the next run starts clean.
    /// Strict (`fold-strict`, the gates) refuses loudly. An ids block past the region runs the
    /// run as `seeds`.
    #[test]
    fn fn120e_a_run_past_the_region_spills_to_the_per_layer_fills_and_strict_refuses() {
        use super::super::exec::{GlueRegion, GlueSeeds};
        let r = GlueRegion {
            zero: 64 << 10,
            ones: 4 << 10,
        };
        let mut g = GlueSeeds::sized(1 << 40, r, true, false);
        g.begin(8).unwrap();
        let ids = 48 * 8 * 40;
        let a = g.take(1000, false).unwrap().unwrap();
        assert!(a >= (1 << 40) + ids && g.holds(a));
        // a take past the zero part spills; every later take too (both parts)
        assert!(g.take(64 << 10, false).unwrap().is_none());
        let sp = g.spill.unwrap();
        assert!(!sp.ones && sp.cap == r.zero && sp.want > r.zero);
        assert!(g.take(8, false).unwrap().is_none());
        assert!(g.take(8, true).unwrap().is_none());
        assert_eq!((g.spills, g.spilled_seeds), (1, 3));
        // the restore covers what the run took in the region (ids + the first seed)
        let (zb, ob) = g.end().unwrap();
        assert_eq!(zb, (a - (1 << 40) + 1000).div_ceil(256) * 256);
        assert_eq!(ob, 0);
        // the next run starts clean
        g.begin(8).unwrap();
        assert!(g.spill.is_none());
        assert_eq!(g.take(1000, false).unwrap(), Some(a));
        assert!(
            g.take(5000, true).unwrap().is_none(),
            "the ones part spills too"
        );
        assert!(g.spill.unwrap().ones);
        g.end().unwrap();
        // an ids block past the region: the run is `seeds` (ids are ordinary seeds)
        g.begin(1000).unwrap();
        assert_eq!(g.ids_stride, 0);
        assert_eq!(g.take(8, false).unwrap(), Some(1 << 40));
        g.end().unwrap();
        // strict: FN120b's loud refusal
        let mut s = GlueSeeds::sized(1 << 40, r, true, true);
        s.begin(8).unwrap();
        let e = s.take(64 << 10, false).unwrap_err().to_string();
        assert!(e.contains("over-committed"), "{e}");
        assert!(s.spill.is_none());
        let mut s = GlueSeeds::sized(1 << 40, r, true, true);
        assert!(s.begin(1000).is_err(), "strict: the ids block refuses");
    }

    /// FN120e: the fallback through the EMITTER -- the crash form (b4 w3 sparse at the 262,144
    /// band) in FN120b's fixed region on a served (non-strict) boot emits the whole step (no
    /// refusal), with the SAME launches as the run that fits, its spilled seeds in the arena's
    /// per-layer seeded part (which the fitting run never touches): `off`'s law for the rest of
    /// the run.
    #[test]
    fn fn120e_the_crash_form_spills_through_the_emitter_with_the_same_launches() {
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let decls = Decls::load(&prebuilt).unwrap();
        let f = fnx(&["--fn-serve", "8099", "--fn-trunk", "w8-hc8-head"]).resolved();
        let rules = boot_rules(&f).unwrap();
        let cc = super::super::exec::ChunkConsts {
            m: 12,
            b: 4,
            w: 3,
            decode: true,
            nb: 262_144 / 16 + 1,
            max_ctx: 262_144,
            ..super::super::exec::ChunkConsts::default()
        };
        let fits = super::super::exec::GlueRegion::of_need(32 << 20, 1 << 20);
        let fixed = super::super::exec::GlueRegion::fixed();
        let a = plan::dry_run_glue_spill(&decls, &rules, cc, true, true, fits).unwrap();
        let b = plan::dry_run_glue_spill(&decls, &rules, cc, true, true, fixed).unwrap();
        assert_eq!(a.glue_spills, 0);
        assert_eq!(b.glue_spills, 2, "one spilled run a rank");
        assert_eq!(a.per_layer, b.per_layer, "the launches a layer");
        assert_eq!(a.by_cell, b.by_cell, "every cell");
        assert_eq!(a.sizes.zero, 0, "a fitting run takes no arena seed");
        assert!(b.sizes.zero > 0, "the spilled seeds are the arena's");
        assert!(b.glue.0 <= fixed.zero && a.glue.0 > fixed.zero);
    }

    #[test]
    fn the_dataflow_train_is_the_default_and_gdn_is_the_arm() {
        // FN120c: `off` (the launch train) is the default; `gdn` the arm, `gdn-nodep` its control
        assert_eq!(dataflow120(&fnx(&[])).unwrap(), Dataflow120::Off);
        assert_eq!(dataflow_class(&fnx(&[])), 0);
        assert_eq!(
            dataflow120(&fnx(&["--fn-dataflow", "gdn"])).unwrap(),
            Dataflow120::Gdn
        );
        assert_eq!(
            dataflow120(&fnx(&["--fn-dataflow", "gdn-nodep"])).unwrap(),
            Dataflow120::GdnNoDep
        );
        assert_eq!(
            dataflow_class(&fnx(&["--fn-dataflow", "gdn"])),
            super::super::exec::DF120_BYTES
        );
        assert!(dataflow120(&fnx(&["--fn-dataflow", "all"])).is_err());
    }

    #[test]
    fn a_dataflow_site_is_found_by_content_and_the_region_refuses_when_full() {
        use super::super::exec::{DfTables, DF120_BYTES, DF120_SITE_BYTES};
        let mut d = DfTables::new(1 << 40, false);
        let a = d.site(vec![1; DF120_SITE_BYTES as usize]).unwrap();
        let b = d.site(vec![2; DF120_SITE_BYTES as usize]).unwrap();
        assert_eq!(a, 1 << 40);
        assert_eq!(b, a + DF120_SITE_BYTES);
        // a re-emission finds its site and queues nothing
        assert_eq!(d.site(vec![1; DF120_SITE_BYTES as usize]).unwrap(), a);
        assert_eq!(d.pending.len(), 2);
        assert_eq!(d.used(), 2);
        for i in 2..DF120_BYTES / DF120_SITE_BYTES {
            let mut v = vec![0u8; DF120_SITE_BYTES as usize];
            v[..8].copy_from_slice(&i.to_le_bytes());
            d.site(v).unwrap();
        }
        assert!(d.site(vec![3; DF120_SITE_BYTES as usize]).is_err());
    }

    #[test]
    fn host_ar_excludes_peer_writing_paths_but_preserves_the_default() {
        let original = fnx(&[]).resolved();
        assert_eq!(original.fn_ar, DEFAULT_FN_AR);
        let host = fnx(&["--fn-serve", "8099", "--fn-ar", "host"]).resolved();
        assert!(host.fn_ar_exact);
        assert!(!host.fn_mtp.on() && !host.spec_couple().unwrap());
        assert_eq!(host.device_step().unwrap(), super::super::dstep::DeviceStep::Off);
        assert!(!host.hc_rr_producer() && !ar_prologue(&host) && !moe_ar_push(&host));
        assert_eq!(mixer_ar_push(&host), super::super::dev::MixPush::Off);
        assert!(!first_step_dev(&host));
        let mtp = fnx(&["--fn-serve", "8099", "--fn-ar", "host", "--fn-mtp", "2"]).resolved();
        assert_eq!(mtp.fn_mtp, FnMtp::K(2));
        assert!(mtp.no_fn_mtp_device_chain);
        assert!(!mtp.spec_couple().unwrap() && !draft_int2_on(&mtp));
    }

    fn fnx(args: &[&str]) -> FnFlags {
        #[derive(clap::Parser)]
        struct C {
            #[command(flatten)]
            f: FnFlags,
        }
        let mut v = vec!["saber"];
        v.extend_from_slice(args);
        <C as clap::Parser>::try_parse_from(v).unwrap().f
    }

    #[test]
    fn fn100_the_buffers_census_sums_to_the_class_and_lends_97_slots() {
        // the served boot's `fn.buffers` (MTP auto, 8,192-row chunks, 262k rope): the census's
        // members + the 256 B barrier ARE the exact ledger's class; the tails = 97 slots a rank
        let f = fnx(&["--fn-serve", "8099"]).resolved();
        let (lines, tot) = buffers_census(&f, &[], geometry::EXPERT_STRIDE).unwrap();
        for l in &lines {
            println!("{l}");
        }
        let c = exact_runtime(&f, &[], 0, 0, super::super::exec::GlueRegion::fixed()).unwrap();
        let b = c.iter().find(|x| x.name == "fn.buffers").unwrap().bytes;
        assert_eq!(tot, b, "the census (+ the 256 B barrier) is the class");
        assert!(lines.last().unwrap().contains("lendable 97 slots"));
        // the default is the layout without the lend; `separate` is the control
        assert_eq!(f.fn_buffers_layout, DEFAULT_FN_BUFFERS_LAYOUT);
        assert_eq!(f.xtier.lend_buffers_top(), (false, false));
        let s = fnx(&["--fn-serve", "8099", "--fn-buffers-layout", "separate"]);
        assert_eq!(s.fn_buffers_layout, "separate");
    }

    /// FN45b4: [`fnx`] for the VRAM-ledger arms (FN38 / FN42b / FN44b): their larger expert RAM
    /// cache leaves no room for the default KV slab, so they plan under the overcommit arm.
    fn fnx_ov(args: &[&str]) -> FnFlags {
        let mut v = args.to_vec();
        v.push("--fn-kv-ram-overcommit");
        fnx(&v)
    }

    /// FN45c3: the prompt / conversation cache default switch (off until the returning-load stall is fixed) at 16 GiB a rank under the
    /// tiered KV; `--no-fn-kv-pcache` = the control; the harness, the flat pools and an explicit
    /// budget keep their own law; the 27B refuses every cache flag.
    #[test]
    fn the_drive_tier_is_refused_at_boot_naming_fn45c3() {
        let e = fnx(&["--fn-serve", "8099", "--fn-kv-ssd-gib", "8"])
            .kv_tier_check(2)
            .unwrap_err()
            .to_string();
        assert!(SSD_TIER_REFUSED && e.contains("FN45c3") && e.contains("fn_draft_wgather"));
        assert!(fnx(&["--fn-serve", "8099"]).kv_tier_check(2).is_ok());
        assert_eq!(
            fnx(&[]).fn_kv_pcache_align,
            DEFAULT_FN_KV_PCACHE_ALIGN,
            "the cut is unaligned by default (FN45c4)"
        );
    }

    /// FN51f-b: `--fn-vram-elastic` defaults to [`DEFAULT_FN_VRAM_ELASTIC`]; `on` grows the hot
    /// pool by exactly `--fn-vram-elastic-kv-max-gib` (whole pages, either dtype), `off` is
    /// today's geometry, the flat pools refuse `on`, a 27B pack refuses every elastic flag.
    #[test]
    fn fn51fb_the_elastic_region_is_the_boot_window_plus_the_borrow_and_off_is_todays() {
        for dt in ["bf16", "fp8"] {
            let off = fnx(&[
                "--fn-serve",
                "8099",
                "--fn-kv-dtype",
                dt,
                "--fn-vram-elastic",
                "off",
            ]);
            let def = fnx(&["--fn-serve", "8099", "--fn-kv-dtype", dt]);
            let on = fnx(&[
                "--fn-serve",
                "8099",
                "--fn-kv-dtype",
                dt,
                "--fn-vram-elastic",
                "on",
            ]);
            let ctx = off.max_ctx();
            let (go, gd, gn) = (
                off.kv_geom(ctx).unwrap(),
                def.kv_geom(ctx).unwrap(),
                on.kv_geom(ctx).unwrap(),
            );
            assert!(!off.vram_elastic() && on.vram_elastic());
            assert_eq!(def.vram_elastic(), DEFAULT_FN_VRAM_ELASTIC);
            assert_eq!(gd == gn, DEFAULT_FN_VRAM_ELASTIC);
            let extra = (DEFAULT_FN_VRAM_ELASTIC_KV_MAX_GIB * (1u64 << 30) as f64) as u64;
            assert_eq!(gn.hot_bytes, (go.hot_bytes + extra) / go.bb() * go.bb());
            assert_eq!(gn.hot_bytes % gn.bb(), 0);
            assert_eq!(
                (gn.ram_bytes, gn.window, gn.chunk),
                (go.ram_bytes, go.window, go.chunk),
                "the RAM slab and the window are unchanged"
            );
        }
        let flat = fnx(&["--fn-kv-tier", "off", "--fn-vram-elastic", "on"]);
        assert!(!flat.vram_elastic());
        assert!(flat.kv_tier_check(2).is_err());
        assert!(fnx(&["--fn-vram-elastic-kv-max-gib", "17"])
            .kv_tier_check(2)
            .is_err());
        for a in [
            &["--fn-vram-elastic", "on"][..],
            &["--fn-vram-elastic", "off"],
            &["--fn-vram-elastic-min-slots", "1"],
            &["--fn-vram-elastic-kv-max-gib", "2"],
            &["--fn-plane-dump", "/p"],
        ] {
            assert!(fnx(a).refuse_on_27b().is_err(), "{a:?}");
        }
    }

    /// FN68d: `--fn-kv-mtp-dtype` -- absent = `DEFAULT_FN_KV_MTP_DTYPE` under fp8 pages in the
    /// worker form; bf16 splits the MTP layer into its own one-layer bf16 tier (12 + 1 layers,
    /// the same pages a layer, the elastic borrow the main tier's, the ledger's `kv.hot` /
    /// `kv.index` / pinned `kv.ram` the two tiers' sum); fp8 = FN68b's one tier; bf16 pages need
    /// no split (an explicit fp8 MTP under them is refused); the one-process control keeps one
    /// dtype (an explicit bf16 there is refused), and the drive tier refuses the split by name.
    #[test]
    fn fn68d_the_mtp_layer_dtype_flag_splits_the_tier_and_its_controls() {
        use saber_mem::kv_tier::KvDtype;
        let srv = |extra: &[&str]| {
            let mut a = vec!["--fn-serve", "8099", "--fn-kv-dtype", "fp8"];
            a.extend_from_slice(extra);
            fnx(&a).resolved()
        };
        let d = srv(&[]);
        assert_eq!(d.fn_kv_mtp_dtype, None);
        assert_eq!(d.kv_mtp_dtype().name(), DEFAULT_FN_KV_MTP_DTYPE);
        let (s, o) = (
            srv(&["--fn-kv-mtp-dtype", "bf16"]),
            srv(&["--fn-kv-mtp-dtype", "fp8"]),
        );
        assert_eq!(
            (s.kv_mtp_dtype(), o.kv_mtp_dtype()),
            (KvDtype::Bf16, KvDtype::Fp8)
        );
        assert!(s.kv_tier_check(2).is_ok() && o.kv_tier_check(2).is_ok());
        let (gs, go) = (
            s.kv_geom(s.max_ctx()).unwrap(),
            o.kv_geom(o.max_ctx()).unwrap(),
        );
        assert!(s.vram_elastic(), "the served boot lends the hot pool");
        assert_eq!((gs.layers, go.layers, go.mtp), (12, 13, None));
        let t = gs.twin().unwrap();
        assert_eq!((t.layers, t.dt), (1, KvDtype::Bf16));
        assert_eq!(gs.active_tokens(), go.active_tokens());
        assert_eq!(
            gs.ram_bytes + t.ram_bytes / 2,
            go.ram_bytes,
            "the same pages a layer"
        );
        // the elastic borrow stays the main tier's: the twin's hot = one layer of the window's
        let base = o.kv_geom(o.max_ctx()).unwrap().hot_bytes;
        let lent = base
            - super::super::kvwire::KvGeom::new_dt(
                o.max_ctx(),
                o.max_chunk(),
                None,
                None,
                KvDtype::Fp8,
            )
            .hot_bytes;
        assert!(lent > 0);
        assert_eq!(t.hot_bytes / 2 * 13, base - lent);
        // the exact ledger's KV classes are the two tiers' sum
        let cls = |f: &FnFlags| {
            exact_runtime(f, &[], 0, 0, super::super::exec::GlueRegion::fixed()).unwrap()
        };
        let get = |c: &[saber_mem::flashnext_budget::Class], n: &str| {
            c.iter().find(|x| x.name == n).map(|x| x.bytes).unwrap()
        };
        let (cs, co) = (cls(&s), cls(&o));
        assert_eq!(get(&cs, saber_mem::kv_tier::KV_HOT), gs.hot_alloc_all());
        assert_eq!(get(&cs, saber_mem::kv_tier::KV_INDEX), go.table_bytes());
        assert_eq!(get(&co, saber_mem::kv_tier::KV_HOT), go.hot_alloc());
        assert!(gs.ram_alloc_all() > go.ram_alloc());
        // bf16 pages: no split; an fp8 MTP beside them is refused
        let b = fnx(&["--fn-kv-dtype", "bf16", "--fn-kv-mtp-dtype", "bf16"]).resolved();
        assert_eq!(b.kv_mtp_dtype(), KvDtype::Bf16);
        assert!(b.kv_geom(b.max_ctx()).unwrap().mtp.is_none());
        let e = fnx(&["--fn-kv-dtype", "bf16", "--fn-kv-mtp-dtype", "fp8"])
            .kv_tier_check(2)
            .unwrap_err()
            .to_string();
        assert!(e.contains("--fn-kv-mtp-dtype"), "{e}");
        // the one-process control: one dtype, an explicit bf16 refused
        let p = fnx(&["--fn-kv-dtype", "fp8", "--fn-one-process"]).resolved();
        assert_eq!(p.kv_mtp_dtype(), KvDtype::Fp8);
        assert!(fnx(&[
            "--fn-kv-dtype",
            "fp8",
            "--fn-one-process",
            "--fn-kv-mtp-dtype",
            "bf16"
        ])
        .kv_tier_check(2)
        .is_err());
        // the flat pools take the split too (the MTP layer's pool at bf16)
        let fl = fnx(&[
            "--fn-kv-dtype",
            "fp8",
            "--fn-kv-tier",
            "off",
            "--fn-kv-mtp-dtype",
            "bf16",
        ])
        .resolved();
        assert!(fl.kv_tier_check(2).is_ok());
        assert_eq!(fl.kv_mtp_dtype(), KvDtype::Bf16);
    }

    /// FN51f-c: `--fn-vram-elastic-refill` defaults to [`DEFAULT_FN_VRAM_ELASTIC_REFILL`];
    /// `eager` and `lazy` parse (clap's value parser refuses anything else); the non-default
    /// value on a 27B pack is refused.
    #[test]
    fn fn51fc_the_refill_default_and_its_control() {
        assert_eq!(
            fnx(&["--fn-serve", "8099"]).fn_vram_elastic_refill,
            DEFAULT_FN_VRAM_ELASTIC_REFILL
        );
        for v in ["eager", "lazy"] {
            assert_eq!(
                fnx(&["--fn-vram-elastic-refill", v]).fn_vram_elastic_refill,
                v
            );
        }
        let other = if DEFAULT_FN_VRAM_ELASTIC_REFILL == "eager" {
            "lazy"
        } else {
            "eager"
        };
        assert!(fnx(&["--fn-vram-elastic-refill", other])
            .refuse_on_27b()
            .is_err());
        assert!(fnx(&[]).refuse_on_27b().is_ok());
    }

    /// FN130: `--fn-vram-elastic-demand` defaults to `on`; a served boot reclaims on demand, every
    /// non-served boot (the harness, the gates) and `strict` refuse loudly; `off` = the control; a
    /// non-default value on a 27B pack is refused.
    #[test]
    fn fn130_the_demand_law_default_strict_and_control() {
        let srv = fnx(&["--fn-serve", "8099"]);
        assert_eq!(srv.fn_vram_elastic_demand, DEFAULT_FN_VRAM_ELASTIC_DEMAND);
        assert_eq!(elastic_demand_form(&srv).unwrap(), ElasticDemandForm::On);
        assert!(
            !elastic_demand_strict(&srv),
            "a served boot reclaims on demand"
        );
        assert!(
            elastic_demand_strict(&fnx(&[])),
            "the harness / gates refuse loudly"
        );
        let st = fnx(&["--fn-serve", "8099", "--fn-vram-elastic-demand", "strict"]);
        assert_eq!(elastic_demand_form(&st).unwrap(), ElasticDemandForm::Strict);
        assert!(elastic_demand_strict(&st));
        let off = fnx(&["--fn-serve", "8099", "--fn-vram-elastic-demand", "off"]);
        assert_eq!(elastic_demand_form(&off).unwrap(), ElasticDemandForm::Off);
        assert!(fnx(&["--fn-vram-elastic-demand", "off"])
            .refuse_on_27b()
            .is_err());
    }

    /// FN130: THE BOUND behind the demand law -- at the served boot (bf16 and fp8 KV, the
    /// launcher's 0.5 GiB borrow and the default) the hot pool WITHOUT the borrow already holds
    /// every sequence slot's window + one widest chunk (the tier's `hot_floor` at max seqs), so
    /// once a demand reclaim has taken every segment back no step can lack a VRAM slot: a demand
    /// reclaim never needs a wait, and the tier's `no vram slot` can only mean the floor law broke.
    #[test]
    fn fn130_the_boot_window_holds_every_window_and_one_chunk_at_max_seqs() {
        for extra in [&[][..], &["--fn-kv-dtype", "fp8"][..]] {
            for gib in ["0.5", "1.5"] {
                let mut a = vec!["--fn-serve", "8099", "--fn-vram-elastic-kv-max-gib", gib];
                a.extend_from_slice(extra);
                let f = fnx(&a);
                let ctx = f.max_ctx();
                let mut off = a.clone();
                off.extend_from_slice(&["--fn-vram-elastic", "off"]);
                let win = fnx(&off).kv_geom(ctx).unwrap().plan().unwrap();
                let reg = f.kv_geom(ctx).unwrap().plan().unwrap();
                let wb = win.window_tokens / win.block_tokens + 1;
                let cb = win.chunk_tokens.div_ceil(win.block_tokens) + 1;
                let floor = win.layers * (win.max_seqs * win.blocks(win.max_ctx).min(wb) + cb);
                assert!(
                    floor <= win.hot_slots && win.hot_slots <= reg.hot_slots,
                    "{a:?}: floor {floor} pages vs the boot window {} / region {}",
                    win.hot_slots,
                    reg.hot_slots
                );
            }
        }
    }

    /// FN68 / FN68b: `--fn-kv-dtype` defaults to `DEFAULT_FN_KV_DTYPE` on every boot; fp8 serves
    /// with the tier (halved geometry), on the flat pools, with the prompt cache and
    /// --warm-prompts (FN68b), under the pack's scale law by default (refused when it has none).
    #[test]
    fn fn68b_the_kv_dtype_default_and_fp8_takes_the_prompt_cache_and_a_scale_law() {
        use saber_mem::kv_tier::KvDtype;
        for a in [&[][..], &["--fn-serve", "8099"][..]] {
            let f = fnx(a);
            assert_eq!(f.fn_kv_dtype, DEFAULT_FN_KV_DTYPE);
        }
        let f = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-kv-dtype",
            "fp8",
            "--fn-kv-mtp-dtype",
            "fp8",
        ])
        .resolved();
        assert_eq!(f.kv_dtype(), KvDtype::Fp8);
        assert!(f.kv_tier_check(2).is_ok());
        let g = f.kv_geom(f.max_ctx()).unwrap();
        assert_eq!(g.dt, KvDtype::Fp8);
        let b = fnx(&["--fn-serve", "8099", "--fn-kv-dtype", "bf16"]).resolved();
        assert_eq!(b.kv_dtype(), KvDtype::Bf16);
        assert_eq!(g.ram_bytes * 2, b.kv_geom(b.max_ctx()).unwrap().ram_bytes);
        assert!(fnx(&["--fn-kv-dtype", "fp8", "--fn-kv-tier", "off"])
            .kv_tier_check(2)
            .is_ok());
        // FN68b: the prompt cache and --warm-prompts serve fp8 pages (the snapshot's page unit is
        // the tier's, the key chain carries the dtype + the scale law)
        for a in [
            &[
                "--fn-serve",
                "8099",
                "--fn-kv-dtype",
                "fp8",
                "--fn-max-ctx",
                "10240",
                "--fn-kv-pcache-gib",
                "4",
            ][..],
            &[
                "--fn-serve",
                "8099",
                "--fn-kv-dtype",
                "fp8",
                "--warm-prompts",
                "/w.json",
            ][..],
        ] {
            assert!(fnx(a).kv_tier_check(2).is_ok(), "{a:?}");
        }
        // the scale law: bf16 refuses it; fp8 reads the pack's file by default and refuses a pack
        // without one; `unit` = FN68's law (no descale table); a file is read as given
        let e = fnx(&["--fn-kv-dtype", "bf16", "--fn-kv-scales", "unit"])
            .kv_tier_check(2)
            .unwrap_err()
            .to_string();
        assert!(e.contains("--fn-kv-scales"), "{e}");
        let scratch = saber_test_scratch::ScratchDir::new("saber-fn68b-kvscales");
        let dir = scratch.path().to_path_buf();
        std::fs::create_dir_all(&dir).unwrap();
        let e = f.kv_scales(&dir).unwrap_err().to_string();
        assert!(e.contains("kv_scales.json"), "{e}");
        let doc = r#"{"format": "fn68b-kv-scales/1", "law": "t", "layers": {"3": {"k": [0.1, 0.2], "v": [0.01, 0.02]}}}"#;
        std::fs::write(dir.join(super::super::kvscale::PACK_FILE), doc).unwrap();
        let k = f.kv_scales(&dir).unwrap().unwrap();
        assert_eq!(k.at(3, 1), (0.2, 0.02));
        assert!(kv_desc_on(&f));
        assert_eq!(kv_desc_class(&f), 49 * 2 * 128 * 4);
        let u = fnx(&["--fn-kv-dtype", "fp8", "--fn-kv-scales", "unit"]);
        assert!(u.kv_scales(&dir).unwrap().unwrap().is_unit());
        assert!(!kv_desc_on(&u));
        assert_eq!(kv_desc_class(&u), 0);
        assert_eq!(kv_desc_class(&b), 0);
        assert!(b.kv_scales(&dir).unwrap().is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// FN66: `--warm-prompts` turns the prompt cache on at the tied settings (16 GiB a rank, no
    /// conversation cut) and serves without admission by reservation (an absent max context =
    /// 10,240); explicit cache flags keep their word; outside the serve plane / without the
    /// cache it is refused; the 27B refuses it.
    #[test]
    fn fn66_warm_prompts_turn_the_cache_on_at_the_tied_settings() {
        let w = ["--fn-serve", "8099", "--warm-prompts", "/w.json"];
        let d = fnx(&w);
        assert_eq!(d.pcache_gib(), Some(DEFAULT_FN_KV_PCACHE_GIB));
        assert_eq!(d.pcache_min_tokens(), WARM_FN_KV_PCACHE_MIN_TOKENS);
        assert_eq!(d.max_ctx(), super::super::kvwire::DEFAULT_FN_KV_MAX_CTX);
        assert!(!d.kv_admit(d.max_ctx()));
        assert!(d.kv_tier_check(2).is_ok());
        // a longer context runs the FN45b4 law (the host pin law decides at boot), never admission
        let d = fnx(&[&w[..], &["--fn-max-ctx", "20480"]].concat());
        assert_eq!(d.max_ctx(), 20_480);
        assert!(!d.kv_admit(20_480) && d.kv_tier_check(2).is_ok());
        // explicit flags keep their word
        let d = fnx(&[
            &w[..],
            &[
                "--fn-kv-pcache-min-tokens",
                "512",
                "--fn-kv-pcache-gib",
                "4",
            ],
        ]
        .concat());
        assert_eq!((d.pcache_gib(), d.pcache_min_tokens()), (Some(4.0), 512));
        // without the flag nothing moved (the CONTROL: today's served boot)
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!(d.pcache_min_tokens(), DEFAULT_FN_KV_PCACHE_MIN_TOKENS);
        assert_eq!(d.max_ctx(), super::super::kvwire::DEFAULT_FN_SERVED_MAX_CTX);
        // refusals by name
        assert!(fnx(&["--warm-prompts", "/w.json"])
            .kv_tier_check(2)
            .is_err());
        assert!(fnx(&[&w[..], &["--no-fn-kv-pcache"]].concat())
            .kv_tier_check(2)
            .is_err());
        assert!(fnx(&["--warm-prompts", "/w.json"]).refuse_on_27b().is_err());
    }

    #[test]
    fn the_prompt_cache_default_switch_and_no_fn_kv_pcache_is_the_control() {
        // FN113b + the user's word 2026-10-13: the prefix tree (THE DEFAULT at a served boot)
        // turns the drive cache on at its own size; `--fn-prefix-tree off` is the control
        let d = fnx(&["--fn-serve", "8099", "--fn-prefix-tree", "on"]);
        assert_eq!(d.pcache_gib(), Some(DEFAULT_FN_PREFIX_TREE_DRIVE_GIB));
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!(d.pcache_gib(), Some(DEFAULT_FN_PREFIX_TREE_DRIVE_GIB));
        let d = fnx(&["--fn-serve", "8099", "--fn-prefix-tree", "off"]);
        assert_eq!(
            d.pcache_gib(),
            PCACHE_SERVED_DEFAULT.then_some(DEFAULT_FN_KV_PCACHE_GIB)
        );
        let d = fnx(&["--fn-serve", "8099", "--fn-kv-pcache-gib", "16"]);
        let (dir, b) = d.kv_pcache(1).unwrap();
        assert_eq!(dir, PathBuf::from("/mnt/nvme/saber-kv/pcache-rank1"));
        assert_eq!(b, 16 << 30);
        assert_eq!(d.pcache_min_tokens(), DEFAULT_FN_KV_PCACHE_MIN_TOKENS);
        assert!(!d.fn_kv_pcache_sync && !d.fn_kv_pcache_verify);
        assert_eq!(
            fnx(&["--fn-serve", "8099", "--no-fn-kv-pcache"]).pcache_gib(),
            None
        );
        assert_eq!(fnx(&[]).pcache_gib(), None, "the harness keeps it off");
        assert_eq!(
            fnx(&[
                "--fn-serve",
                "8099",
                "--fn-kv-tier",
                "off",
                "--fn-max-ctx",
                "8192"
            ])
            .pcache_gib(),
            None
        );
        assert_eq!(fnx(&["--fn-kv-pcache-gib", "4"]).pcache_gib(), Some(4.0));
        for a in [
            &["--no-fn-kv-pcache"][..],
            &["--fn-kv-pcache-sync"],
            &["--fn-kv-pcache-verify"],
            &["--fn-kv-pcache-min-tokens", "64"],
        ] {
            assert!(fnx(a).refuse_on_27b().is_err(), "{a:?}");
        }
    }

    #[test]
    fn fn113_the_prefix_tree_is_the_default_and_off_is_the_control() {
        // FN113b + the user's word 2026-10-13: `on` = the default at a served boot; `off` = the
        // control; the harness (no `--fn-serve`) never runs it
        assert_eq!(DEFAULT_FN_PREFIX_TREE, "on");
        let d = fnx(&["--fn-serve", "8099"]);
        assert!(d.fn_prefix_tree.is_none() && d.prefix_tree());
        assert_eq!(d.pcache_gib(), Some(DEFAULT_FN_PREFIX_TREE_DRIVE_GIB));
        d.kv_tier_check(2).unwrap();
        // a served boot where the tree cannot run stands it down without the flag
        assert!(!fnx(&["--fn-serve", "8099", "--no-fn-kv-pcache"]).prefix_tree());
        let o = fnx(&["--fn-serve", "8099", "--fn-prefix-tree", "off"]);
        assert!(!o.prefix_tree() && o.prefix_tree_opts().is_none());
        assert_eq!(o.prefix_tree_ram_bytes(), 0);
        assert_eq!(
            o.pcache_gib(),
            PCACHE_SERVED_DEFAULT.then_some(DEFAULT_FN_KV_PCACHE_GIB)
        );
        assert!(!fnx(&[]).prefix_tree());
        // an explicit `on` where the tree cannot run is refused
        for a in [
            &["--fn-kv-tier", "off", "--fn-max-ctx", "8192"][..],
            &["--no-fn-kv-pcache"],
        ] {
            assert!(
                fnx(&[&["--fn-serve", "8099", "--fn-prefix-tree", "on"][..], a].concat())
                    .kv_tier_check(2)
                    .is_err()
            );
        }
        let t = fnx(&["--fn-serve", "8099", "--fn-prefix-tree", "on"]);
        assert!(t.prefix_tree());
        assert_eq!(t.pcache_gib(), Some(DEFAULT_FN_PREFIX_TREE_DRIVE_GIB));
        assert_eq!(t.prefix_tree_ram_bytes(), 3 << 29);
        let o = t.prefix_tree_opts().unwrap();
        assert_eq!((o.min_gap, o.sys_min), (64, 512));
        // the served default boots admission by reservation: the tree is allowed there
        t.kv_tier_check(2).unwrap();
        assert!(fnx(&[
            "--fn-prefix-tree",
            "on",
            "--fn-kv-tier",
            "off",
            "--fn-max-ctx",
            "8192"
        ])
        .kv_tier_check(2)
        .is_err());
        // the 27B refuses the flag
        assert!(fnx(&["--fn-prefix-tree", "on"]).refuse_on_27b().is_err());
        assert!(fnx(&["--fn-prefix-tree-ram-gib", "2"])
            .refuse_on_27b()
            .is_err());
    }

    /// FN125: `--fn-image-cache` -- absent = the default (`on` since hold h1); `off` = FN55b's
    /// bypass, the control; `on` defers the tower only on a boot with a cache; the 27B refuses the
    /// flag.
    #[test]
    fn fn125_the_image_cache_is_the_default_and_off_is_the_control() {
        assert_eq!(DEFAULT_FN_IMAGE_CACHE, "on");
        let d = fnx(&[]);
        assert!(image_cache_on(&d) && image_cache_defers(&d, true));
        assert!(
            !image_cache_defers(&d, false),
            "no cache: the tower runs at encode"
        );
        let on = fnx(&["--fn-image-cache", "on"]);
        assert!(image_cache_on(&on));
        assert!(image_cache_defers(&on, true) && !image_cache_defers(&on, false));
        let off = fnx(&["--fn-image-cache", "off"]);
        assert!(!image_cache_on(&off));
        assert!(on.refuse_on_27b().is_err());
        assert!(d.refuse_on_27b().is_ok());
    }

    #[test]
    fn fn113b_the_end_node_is_stored_off_the_step_by_default_and_step_is_the_control() {
        let t = fnx(&["--fn-serve", "8099", "--fn-prefix-tree", "on"]);
        assert_eq!(t.fn_prefix_tree_end, "live");
        assert_eq!(DEFAULT_FN_PREFIX_TREE_END, "live");
        t.kv_tier_check(2).unwrap();
        for arm in ["side", "step", "off"] {
            let a = fnx(&[
                "--fn-serve",
                "8099",
                "--fn-prefix-tree",
                "on",
                "--fn-prefix-tree-end",
                arm,
            ]);
            assert_eq!(a.fn_prefix_tree_end, arm);
            a.kv_tier_check(2).unwrap();
            // the arm without the tree is refused by name; the 27B refuses the flag
            let n = fnx(&[
                "--fn-serve",
                "8099",
                "--fn-prefix-tree",
                "off",
                "--fn-prefix-tree-end",
                arm,
            ]);
            assert!(n.kv_tier_check(2).is_err());
            assert!(fnx(&["--fn-prefix-tree-end", arm]).refuse_on_27b().is_err());
        }
    }

    /// LR-I: `--fn-prefix-tree-cuts` -- `today` is the default (the qwen4 law); the arms parse to
    /// their rule, reach the tree's options, need the tree, and the 27B refuses them.
    #[test]
    fn lri_the_cut_rule_is_today_by_default_and_the_arms_reach_the_tree() {
        use saber_mem::prefix_tree::CutRule;
        let t = fnx(&["--fn-serve", "8099", "--fn-prefix-tree", "on"]);
        assert_eq!(t.fn_prefix_tree_cuts, "today");
        assert_eq!(t.prefix_tree_opts().unwrap().cuts, CutRule::default());
        assert!(fnx(&[]).refuse_on_27b().is_ok());
        for (arm, asst, grid) in [
            ("asst", true, 0),
            ("grid2048", false, 2048),
            ("grid2048+asst", true, 2048),
        ] {
            let a = fnx(&[
                "--fn-serve",
                "8099",
                "--fn-prefix-tree",
                "on",
                "--fn-prefix-tree-cuts",
                arm,
            ]);
            a.kv_tier_check(2).unwrap();
            let c = a.prefix_tree_opts().unwrap().cuts;
            assert_eq!((c.asst, c.grid), (asst, grid), "{arm}");
            assert_eq!(a.fn_prefix_tree_cuts, c.name());
            let n = fnx(&[
                "--fn-serve",
                "8099",
                "--fn-prefix-tree",
                "off",
                "--fn-prefix-tree-cuts",
                arm,
            ]);
            assert!(n.kv_tier_check(2).is_err());
            assert!(fnx(&["--fn-prefix-tree-cuts", arm])
                .refuse_on_27b()
                .is_err());
        }
        #[derive(clap::Parser)]
        struct C {
            #[command(flatten)]
            f: FnFlags,
        }
        for bad in ["grid100", "end", ""] {
            let r = <C as clap::Parser>::try_parse_from(["saber", "--fn-prefix-tree-cuts", bad]);
            assert!(r.is_err(), "{bad}");
        }
    }

    #[test]
    fn fn38_the_max_chunk_rungs_and_the_idle_law() {
        // FN44: the default is 8,192 rows, `always` (no cap beside decode)
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!((d.max_chunk(), d.busy_chunk()), (8192, 0));
        let w = fnx(&["--fn-serve", "8099", "--fn-wide-when", "idle"]);
        assert_eq!((w.max_chunk(), w.busy_chunk()), (8192, MAX_ROWS));
        let n = fnx(&["--fn-serve", "8099", "--fn-max-chunk", "2048"]);
        assert_eq!((n.max_chunk(), n.busy_chunk()), (2048, 0));
        let a = fnx(&["--fn-max-chunk", "4096", "--fn-wide-when", "always"]);
        assert_eq!((a.max_chunk(), a.busy_chunk()), (4096, 0));
        for bad in ["1024", "3000", "16384"] {
            #[derive(clap::Parser)]
            struct C {
                #[command(flatten)]
                f: FnFlags,
            }
            assert!(
                <C as clap::Parser>::try_parse_from(["saber", "--fn-max-chunk", bad]).is_err(),
                "{bad}"
            );
        }
        // the reduce's mailbox: 16 MiB holds 4,096 rows' wht6 payload, 8,192 rows take 32 MiB
        assert_eq!(super::super::dev::ar_mb(2048), super::super::dev::AR_MB);
        assert_eq!(super::super::dev::ar_mb(4096), super::super::dev::AR_MB);
        assert_eq!(super::super::dev::ar_mb(8192), 2 * super::super::dev::AR_MB);
    }

    /// FN150: the ledger at the NVFP4 record (676 pages): the STRIDE alone leaves 5.47 % fewer
    /// VRAM expert slots a rank than MXFP4's 639 (every other class unchanged; the 10,240-token
    /// boot reads exactly that), and the served 262,144 boot pays MORE: the 676-page RAM L2 no
    /// longer pins beside the KV slab, so the embedding stands down to VRAM (FN99) and the L2 is
    /// capped (records NVMe-only = the demand path) -- a measured planning fact for the FP4-Local
    /// served cells. `--fn-experts-fmt nvfp4` on an MXFP4 pack is refused by the header's tag.
    #[test]
    fn fn150_the_nvfp4_ledger_has_the_676_page_slots() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2");
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let nvg = Some(super::super::xtp2::XGeom::NVFP4_EP);
        let nvs = super::super::xtp2::NVFP4_EXPERT_STRIDE;
        for (args, lo, hi) in [
            (
                &["--fn-serve", "8099", "--fn-max-ctx", "10240"][..],
                0.940,
                0.947,
            ),
            (&["--fn-serve", "8099"][..], 0.88, 0.947),
        ] {
            let f = fnx(args).resolved();
            let mx = plan_boot(pack, &prebuilt, &[], &f).unwrap();
            let nv = plan_boot_at(pack, &prebuilt, &[], &f, nvg).unwrap();
            for r in 0..2 {
                let (a, b) = (mx.budgets[r].slots, nv.budgets[r].slots);
                let ratio = b as f64 / a as f64;
                let (la, lb) = (mx.laws[r].tier.ram_slots, nv.laws[r].tier.ram_slots);
                let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
                println!(
                    "fn150 {args:?} rank{r}: VRAM expert slots MXFP4 {a} -> NVFP4 {b} ({:+.2} %), \
                     RAM L2 {la} -> {lb} (NVMe-only {} -> {}), pinned L2 {:.2} -> {:.2} GiB",
                    (ratio - 1.0) * 100.0,
                    12_288u64.saturating_sub(a + u64::from(la) + mx.budgets[r].stage_slots),
                    12_288u64.saturating_sub(b + u64::from(lb) + nv.budgets[r].stage_slots),
                    gib(mx.host.pinned(&[u64::from(la)], geometry::EXPERT_STRIDE)),
                    gib(nv.host.pinned(&[u64::from(lb)], nvs)),
                );
                assert!((lo..hi).contains(&ratio), "{args:?} rank {r}: {a} -> {b}");
                assert_eq!(nv.budgets[r].stride, nvs);
            }
        }
        let g = fnx(&["--fn-serve", "8099", "--fn-experts-fmt", "nvfp4"]).resolved();
        let e = plan_boot(pack, &prebuilt, &[], &g)
            .err()
            .expect("refused")
            .to_string();
        assert!(
            e.contains("MXFP4_E8M0") && e.contains("--fn-experts-fmt"),
            "{e}"
        );
    }

    /// FN45b4: the DEFAULT served boot (tiered KV at the ceiling) pins its KV slab within the host's
    /// pinnable bytes beside the UNSHRUNK expert RAM cache; past the ceiling it is refused (the
    /// overcommit arm boots); the control (`--fn-kv-tier off`) is today's 8,192 flat pools.
    #[test]
    fn fn45b4_the_default_served_boot_fits_the_host_and_past_it_is_refused() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2");
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        // FN51b: the pre-FN51b served boot (the slab's sizing law) is now `--fn-max-ctx 10240`
        let f = fnx(&["--fn-serve", "8099", "--fn-max-ctx", "10240"]).resolved();
        assert!(f.kv_tier());
        assert_eq!(f.max_ctx(), super::super::kvwire::DEFAULT_FN_KV_MAX_CTX);
        let b = plan_boot(pack, &prebuilt, &[], &f).unwrap();
        let l2: Vec<u64> = b.laws.iter().map(|l| u64::from(l.tier.ram_slots)).collect();
        let s = geometry::EXPERT_STRIDE;
        // the ceiling at the BOOT's KV dtype (fp8 since FN68b's flip, half the bytes a token):
        // `plan_boot` sizes the slab with `token_bytes(f.kv_dtype())`, so a ceiling at the bf16
        // constants is half the real one and names a context that fits.
        let ceil = b.host.kv_ceiling(
            &l2,
            s,
            super::super::kvwire::token_bytes(f.kv_dtype()),
            super::super::kvwire::page_bytes(f.kv_dtype()),
            2048,
        );
        println!(
            "fn45b4 default: L2 {l2:?}, pinned {:.2} of {:.2} GiB, ceiling {ceil} at {:?}",
            b.host.pinned(&l2, s) as f64 / f64::from(1u32 << 30),
            b.host.pinnable() as f64 / f64::from(1u32 << 30),
            f.kv_dtype()
        );
        assert!(b.host.kv_ram_per_rank > 0);
        assert!(ceil >= u64::from(f.max_ctx()));
        // a margin, so a small ledger move (a few expert slots) does not refuse the default boot
        assert!(
            b.host.pinnable() - b.host.pinned(&l2, s) >= 1 << 28,
            "the default boot's pinned margin under 0.25 GiB: re-derive DEFAULT_FN_KV_MAX_CTX"
        );
        let c = fnx(&["--fn-serve", "8099", "--fn-kv-tier", "off"]).resolved();
        assert_eq!(c.max_ctx(), 8192);
        let bc = plan_boot(pack, &prebuilt, &[], &c).unwrap();
        assert_eq!(bc.host.kv_ram_per_rank, 0);
        // FN51: past the ceiling the law is admission by reservation; its control refuses. The
        // past-the-ceiling context is DERIVED from this boot's ceiling (the next power of two
        // above it, 65,536 at least), not pinned: the fp8 KV default (FN68b's flip, 2026-10-08)
        // halved the bytes a token, the host's ceiling doubled (34,816 at bf16 -> ~69k at fp8 on
        // this box), and the pinned 65,536 fit -- the test then refused nothing (main's one red
        // test after 7cb3709d; the elastic default was the first suspect, wrongly).
        let past = u32::try_from(ceil + 1)
            .expect("fn45b4: the KV ceiling fits a u32")
            .next_power_of_two()
            .max(65_536)
            .to_string();
        println!("fn45b4 past-the-ceiling context {past}");
        let big = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-max-ctx",
            past.as_str(),
            "--no-fn-kv-admit",
        ])
        .resolved();
        assert!(plan_boot(pack, &prebuilt, &[], &big).is_err());
        let over = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-max-ctx",
            past.as_str(),
            "--fn-kv-ram-overcommit",
        ])
        .resolved();
        plan_boot(pack, &prebuilt, &[], &over).unwrap();
    }

    /// FN51c: a harness sequence's admission covers the MTP walk's overshoot. A b8 group steps
    /// until its slowest sequence holds `--fn-decode` tokens, so a sequence that accepts every
    /// draft advances the full width each of those steps; FN51b's law (prompt + decode = 163 for
    /// a 35-token prompt at 128) refused it (`164 tokens past its admitted 163`). The fixed law
    /// admits the worst walk at the verify cap's width; SPEC 0 keeps prompt + decode (+ the
    /// step's open pages).
    #[test]
    fn fn51c_the_harness_admission_covers_the_mtp_walk() {
        use super::super::kvwire::{plan_step, ADMIT_EXTRA, BB};
        use saber_mem::kv_tier::{KvAddr, KvPageTable, KvTier};
        let prompts: Vec<Prompt> = (0..8)
            .map(|i| Prompt {
                id: format!("p{i}"),
                ids: vec![1; 35],
                ref_top: Vec::new(),
            })
            .collect();
        let spec0 = fnx(&["--fn-decode", "128", "--fn-pools-ctx", "262144"]).resolved();
        assert_eq!(harness_tokens(&spec0, &prompts), 35 + 128 + ADMIT_EXTRA);
        let auto = fnx(&[
            "--fn-decode",
            "128",
            "--fn-mtp",
            "auto",
            "--fn-pools-ctx",
            "262144",
        ])
        .resolved();
        let first = harness_tokens(&auto, &prompts);
        let w = super::super::exec::WMAX;
        assert_eq!(first, 35 + 128 * w + ADMIT_EXTRA);
        let ctx = pools_ctx(&auto, &prompts);
        assert_eq!(ctx, 262_144);
        assert!(auto.kv_admit(ctx));
        // the worst walk: slot 0 accepts the cap's width every step while slot 1 accepts none
        let walk = |first: u32| -> anyhow::Result<()> {
            let g = auto.kv_geom_for(ctx, first).expect("the tier");
            assert!(g.admit);
            let plan = g.plan()?;
            let mut tier = KvTier::new(plan.clone());
            let addr = KvAddr::new(
                BB,
                0x10_0000_0000,
                plan.hot_slots,
                0x20_0000_0000,
                plan.ram_slots,
                0x10_0000_0000 + u64::from(plan.hot_slots) * BB,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut table = KvPageTable::new(&plan, 9, addr).map_err(|e| anyhow::anyhow!("{e}"))?;
            plan_step(&mut tier, &mut table, &g, &[(0, 0, 35), (1, 0, 35)], true)?;
            let (mut a, mut b) = (35u32, 35u32);
            for _ in 0..128 {
                plan_step(
                    &mut tier,
                    &mut table,
                    &g,
                    &[(0, a, a + w), (1, b, b + w)],
                    false,
                )?;
                a += w;
                b += 1;
            }
            Ok(())
        };
        walk(first).unwrap();
        let old = walk(35 + 128).unwrap_err().to_string();
        assert!(old.contains("past its admitted"), "{old}");
    }

    /// LR-K: `--fn-kv-admit-slab-tokens` -- absent and at the default value the geometry is
    /// BYTE-IDENTICAL (the same `KvGeom`, pinned bytes, index pages, budget); a bigger slab
    /// grows the RAM slab, the scheduler's budget and the paged index by the same tokens and
    /// lazy growth still builds; under the default, without admission, or beside the byte form
    /// it is refused by name (NEGATIVE CONTROLS).
    #[test]
    fn lrk_the_admission_slab_arm_is_identical_at_default_and_grows_the_budget() {
        use super::super::kvwire::{admit_slab_tokens, ADMIT_EXTRA};
        let base = ["--fn-serve", "8099", "--fn-kv-lazy", "1024"];
        let with = |extra: &[&str]| {
            let mut v = base.to_vec();
            v.extend_from_slice(extra);
            fnx(&v).resolved()
        };
        let d = with(&[]);
        let ctx = pools_ctx(&d, &[]);
        assert!(d.kv_admit(ctx));
        let gd = d.kv_geom(ctx).unwrap();
        assert!(gd.admit && gd.lazy);
        assert_eq!(admit_slab_tokens(262_144), 273_104);
        assert_eq!(gd.active_tokens(), 272_528, "the live slab's budget a rank");
        // the default value spelled out == absent, byte for byte
        let dv = admit_slab_tokens(ctx).to_string();
        let ge = with(&["--fn-kv-admit-slab-tokens", dv.as_str()])
            .kv_geom(ctx)
            .unwrap();
        assert_eq!(ge, gd);
        assert_eq!(ge.ram_alloc_all(), gd.ram_alloc_all());
        assert_eq!(ge.ix_pages(), gd.ix_pages());
        assert_eq!(gd.admit_slab_tokens_in(gd.ram_alloc_all()), 273_104);
        // a bigger slab: the budget, the pinned bytes and the index grow by the same tokens
        for t in [400_000u64, 550_000, 640_000] {
            let ts = t.to_string();
            let f = with(&["--fn-kv-admit-slab-tokens", ts.as_str()]);
            f.kv_tier_check(2).unwrap();
            let g = f.kv_geom(ctx).unwrap();
            assert!(g.admit && g.lazy && g.mtp.is_some());
            assert_eq!(g.active_tokens(), t - 9 * u64::from(ADMIT_EXTRA));
            assert_eq!(
                g.ram_alloc_all() - gd.ram_alloc_all(),
                (t - 273_104) * g.token_bytes_all()
            );
            assert_eq!(u64::from(g.ix_pages()), t / 16 + 18);
            assert_eq!(g.admit_slab_tokens_in(g.ram_alloc_all()), t);
            assert_eq!(g.admit_slab_tokens_in(g.ram_alloc_all() - 1), t - 16);
            // the tier plans (the hot window / quota do not depend on the slab) and goes lazy
            let (pd, p) = (gd.plan().unwrap(), g.plan().unwrap());
            assert_eq!(
                (p.hot_slots, p.quota, p.window_tokens),
                (pd.hot_slots, pd.quota, pd.window_tokens)
            );
            assert!(p.ram_slots > pd.ram_slots);
            saber_mem::kv_tier::KvTier::new(p).lazy().unwrap();
            super::super::serve::scheduler_config(ctx, 5, false, 8192, 0, Some(g.active_tokens()))
                .unwrap();
            // and stays fp8-target / bf16-MTP: the twin holds the same blocks
            let tw = g.twin().unwrap();
            assert_eq!(
                tw.ram_bytes / tw.bb(),
                g.ram_bytes / g.bb() / u64::from(g.layers)
            );
        }
        // NEGATIVE CONTROLS: under the default, without admission, beside the byte form
        let under = (admit_slab_tokens(ctx) - 16).to_string();
        let e = with(&["--fn-kv-admit-slab-tokens", under.as_str()])
            .kv_tier_check(2)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("--fn-kv-admit-slab-tokens") && e.contains("under the default"),
            "{e}"
        );
        let e = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-max-ctx",
            "10240",
            "--fn-kv-admit-slab-tokens",
            "400000",
        ])
        .resolved()
        .kv_tier_check(2)
        .unwrap_err()
        .to_string();
        assert!(e.contains("admission by reservation only"), "{e}");
        #[derive(clap::Parser)]
        struct C {
            #[command(flatten)]
            f: FnFlags,
        }
        assert!(<C as clap::Parser>::try_parse_from([
            "saber",
            "--fn-serve",
            "8099",
            "--fn-kv-ram-gib",
            "3.2",
            "--fn-kv-admit-slab-tokens",
            "400000"
        ])
        .is_err());
    }

    /// LR-K: THE HOST LAW on the SERVED boot (tools/serve_flashnext.sh's flags + the hold's base
    /// arms): the default slab pins, the largest slab that pins at tree 4.5 / 2.5 GiB is found
    /// (printed: the hold script's arms), one past it and an absurd one are REFUSED BY NAME.
    #[test]
    fn lrk_the_host_law_bounds_the_admission_slab() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2-trunk8");
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        for tree in ["4.5", "2.5"] {
            let flags = |slab: Option<u64>| {
                let ts = slab.map(|t| t.to_string());
                let mut v = vec![
                    "--fn-serve",
                    "8099",
                    "--fn-trunk",
                    "w8-hc8-head",
                    "--fn-kv-dtype",
                    "fp8",
                    "--fn-vram-elastic-kv-max-gib",
                    "0.5",
                    "--fn-experts",
                    "tp2",
                    "--fn-thinking",
                    "on",
                    "--fn-vision",
                    "--fn-ring-idle",
                    "--fn-prefill-yield",
                    "1",
                    "--fn-kv-lazy",
                    "1024",
                    "--fn-prefix-tree-ram-gib",
                    tree,
                ];
                if let Some(t) = ts.as_deref() {
                    v.extend_from_slice(&["--fn-kv-admit-slab-tokens", t]);
                }
                fnx(&v).resolved()
            };
            let plan = |slab: Option<u64>| plan_boot(pack, &prebuilt, &[], &flags(slab));
            let bd = plan(None).unwrap();
            let l2: Vec<u64> = bd
                .laws
                .iter()
                .map(|l| u64::from(l.tier.ram_slots))
                .collect();
            let def = super::super::kvwire::admit_slab_tokens(262_144);
            let (mut lo, mut hi) = (def / 16, 4_000_000u64 / 16);
            let e = plan(Some(hi * 16))
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(
                e.contains("--fn-kv-admit-slab-tokens") && e.contains("the largest admission slab"),
                "an absurd slab must be refused by name: {e}"
            );
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if plan(Some(mid * 16)).is_ok() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let max = lo * 16;
            let bm = plan(Some(max)).unwrap();
            let l2m: Vec<u64> = bm
                .laws
                .iter()
                .map(|l| u64::from(l.tier.ram_slots))
                .collect();
            let e = plan(Some(max + 16))
                .err()
                .map(|e| e.to_string())
                .expect("one page past the max is refused");
            assert!(e.contains("--fn-kv-admit-slab-tokens"), "{e}");
            println!(
                "lrk tree {tree}: default {def} tokens pins {:.3} of {:.3} GiB (L2 {l2:?}, static {:?}); \
                 MAX slab {max} tokens a rank (+{:.1} %) pins {:.3} GiB (L2 {l2m:?}, static {:?}); \
                 one page past: {e}",
                bd.host.pinned(&l2, bd.budgets[0].stride) as f64 / f64::from(1u32 << 30),
                bd.host.pinnable() as f64 / f64::from(1u32 << 30),
                bd.budgets.iter().map(|x| x.slots).collect::<Vec<_>>(),
                100.0 * (max as f64 / def as f64 - 1.0),
                bm.host.pinned(&l2m, bm.budgets[0].stride) as f64 / f64::from(1u32 << 30),
                bm.budgets.iter().map(|x| x.slots).collect::<Vec<_>>(),
            );
            assert!(max >= def, "the default slab pins at tree {tree}");
        }
    }

    /// FN51: `--fn-max-ctx 262144` plans under ADMISSION BY RESERVATION inside this box: the
    /// expert RAM cache stays the COMPLETE tier (L2 == the complement), the 3.65 GiB slab + the
    /// paged index fit the pinnable bytes, the slab admits one 262,144-token request; the default
    /// 10,240 boot is untouched (same slots, same pools, no admission); `--no-fn-kv-admit` keeps
    /// FN45b4's refusal.
    #[test]
    fn fn51_a_262k_boot_plans_by_admission_and_the_default_boot_is_untouched() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2");
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let s = geometry::EXPERT_STRIDE;
        let d = fnx(&["--fn-serve", "8099", "--fn-max-ctx", "10240"]).resolved();
        assert!(!d.kv_admit(pools_ctx(&d, &[])));
        let bd = plan_boot(pack, &prebuilt, &[], &d).unwrap();
        let g = d.kv_geom(pools_ctx(&d, &[])).unwrap();
        assert_eq!(g.ix_pages(), 0, "the default boot keeps the identity index");
        // FN51f-b (2026-10-08, ELASTIC VRAM the default): the 262k plan is the SERVED one -- the
        // -trunk8 pack with `--fn-trunk w8` (tools/serve_flashnext.sh) -- because the elastic region
        // moves ~600 expert slots a rank from VRAM into the pinned L2, and with the ENGINE default
        // (the bf16 trunk resident, -511 slots) the host pins 37.46 GiB > the 36.40 pinnable: that
        // boot is REFUSED by the law (asserted below); the served w8 boot pins 35.06 and fits.
        let bf16_elastic = fnx(&["--fn-serve", "8099", "--fn-max-ctx", "262144"]).resolved();
        if bf16_elastic.vram_elastic()
            && bf16_elastic.kv_dtype() == saber_mem::kv_tier::KvDtype::Bf16
        {
            let e = plan_boot(pack, &prebuilt, &[], &bf16_elastic)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(
                e.contains("pins") && e.contains("past the"),
                "the bf16-trunk engine default at 262k with the elastic region must be refused by the pin law: {e}"
            );
        }
        let pack8 = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2-trunk8");
        let pack = if pack8.join("plan.json").exists() {
            pack8
        } else {
            pack
        };
        let f = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-max-ctx",
            "262144",
            "--fn-trunk",
            "w8",
        ])
        .resolved();
        let ctx = pools_ctx(&f, &[]);
        assert_eq!(ctx, 262_144);
        // FN51b: a PLAIN served boot runs DEFAULT_FN_SERVED_MAX_CTX (262,144 at FN51b; REVERTED to
        // 10,240 on 2026-10-02 after the fn_draft_wgather fault; 262,144 again at FN51g the same day
        // once every gate passed -- the test follows the constant); harness / scripted runs keep 10,240
        let plain = fnx(&["--fn-serve", "8099"]).resolved();
        assert_eq!(
            plain.max_ctx(),
            super::super::kvwire::DEFAULT_FN_SERVED_MAX_CTX
        );
        assert_eq!(
            pools_ctx(&plain, &[]),
            super::super::kvwire::DEFAULT_FN_SERVED_MAX_CTX
        );
        if super::super::kvwire::DEFAULT_FN_SERVED_MAX_CTX
            > super::super::kvwire::DEFAULT_FN_KV_MAX_CTX
        {
            assert!(plain.kv_admit(pools_ctx(&plain, &[])));
        }
        assert_eq!(
            fnx(&[]).resolved().max_ctx(),
            super::super::kvwire::DEFAULT_FN_KV_MAX_CTX
        );
        assert_eq!(
            fnx(&["--fn-serve", "8099", "--fn-kv-tier", "off"])
                .resolved()
                .max_ctx(),
            8192
        );
        assert!(f.kv_admit(ctx));
        let b = plan_boot(pack, &prebuilt, &[], &f).unwrap();
        let l2: Vec<u64> = b.laws.iter().map(|l| u64::from(l.tier.ram_slots)).collect();
        for (bb, l) in b.budgets.iter().zip(&l2) {
            assert_eq!(
                *l,
                bb.complement(),
                "the expert RAM cache is the complete tier"
            );
        }
        let gk = f.kv_geom(ctx).unwrap();
        gk.plan().unwrap();
        let tok = gk.active_tokens();
        println!(
            "fn51: slots {:?} (default {:?}), L2 {l2:?}, pinned {:.2} of {:.2} GiB, admits {tok} \
             tokens a rank, index pages {}",
            b.budgets.iter().map(|x| x.slots).collect::<Vec<_>>(),
            bd.budgets.iter().map(|x| x.slots).collect::<Vec<_>>(),
            b.host.pinned(&l2, s) as f64 / f64::from(1u32 << 30),
            b.host.pinnable() as f64 / f64::from(1u32 << 30),
            gk.ix_pages()
        );
        assert!(tok >= 262_144 + 8, "one 262k request fits the slab alone");
        assert!(
            b.host.pinnable() - b.host.pinned(&l2, s) >= 1 << 28,
            "a 0.25 GiB pinned margin"
        );
        super::super::serve::scheduler_config(262_144, 5, false, 8192, 0, Some(tok)).unwrap();
        let e = super::super::serve::scheduler_config(262_144, 5, false, 8192, 0, Some(200_000))
            .unwrap_err()
            .to_string();
        assert!(e.contains("the ceiling is --fn-max-ctx 199936"), "{e}");
        let c = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-max-ctx",
            "262144",
            "--no-fn-kv-admit",
        ])
        .resolved();
        assert!(plan_boot(pack, &prebuilt, &[], &c).is_err());
        assert_eq!(
            rope_bytes(rope_pos(ctx)),
            rope_table(rope_pos(ctx)).len() as u64
        );
        assert_eq!(
            rope_table(rope_pos(ctx))[..1 << 20],
            rope_table(ROPE_POS)[..1 << 20]
        );
    }

    /// FN38: the served boot's ledger at each widest chunk (the pack's `plan.json`; skipped where
    /// the pack is absent): the arena + the chunk buffers are paid in VRAM expert slots.
    #[test]
    fn fn38_the_wide_rung_is_paid_in_expert_slots() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2");
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let mut last = u64::MAX;
        for wide in ["2048", "4096", "8192"] {
            let f = fnx_ov(&[
                "--fn-serve",
                "8099",
                "--fn-max-chunk",
                wide,
                "--no-fn-arena-borrow",
            ])
            .resolved();
            let b = plan_boot(pack, &prebuilt, &[], &f).unwrap();
            let slots: Vec<u64> = b.budgets.iter().map(|b| b.slots).collect();
            println!(
                "fn38 wide {wide}: arena {} MiB, slots {slots:?}",
                b.dr.sizes.total() >> 20
            );
            assert!(slots[0] < last);
            last = slots[0];
        }
    }

    /// FN64d: a billed exec budget is paid OUT of the ledger's graph term: the plan budget grows by
    /// today's 704 unbilled execs (1,408 MiB), the class `fn.graph.exec` takes the budget, and the
    /// difference lands in VRAM expert slots (the pack's `plan.json`; skipped where absent);
    /// unbilled (no flag) = the same slots as before the flag existed.
    #[test]
    fn fn64d_a_billed_exec_budget_turns_the_unbilled_execs_into_slots() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2");
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let slots = |extra: &[&str]| -> Vec<u64> {
            let mut a = vec!["--fn-serve", "8099"];
            a.extend_from_slice(extra);
            let f = fnx_ov(&a).resolved();
            let b = plan_boot(pack, &prebuilt, &[], &f).unwrap();
            b.budgets.iter().map(|b| b.slots).collect()
        };
        let base = slots(&["--fn-graph-exec-budget-mib", "0"]);
        assert_eq!(
            slots(&["--fn-graph-form", "lazy"]),
            base,
            "unbilled is the lazy form's default"
        );
        // FN64e: the charge bills its bytes and grows the plan by the same: slots EQUAL
        assert_eq!(
            slots(&["--fn-graph-exec-charge"]),
            base,
            "the charge moved the slots"
        );
        // FN98: the one form's freed bytes (600 MiB) become static slots (the default)
        assert_eq!(slots(&[]), slots(&["--fn-graph-form", "one"]));
        let one = slots(&["--fn-graph-form", "one"]);
        let gain98 = (FN98_FREED_MIB << 20) / geometry::EXPERT_STRIDE;
        for r in 0..2 {
            assert!(
                one[r] + 1 >= base[r] + gain98 && one[r] <= base[r] + gain98 + 1,
                "{one:?} vs {base:?} (+{gain98})"
            );
        }
        let full = slots(&["--fn-graph-exec-budget-mib", "1408"]);
        let half = slots(&["--fn-graph-exec-budget-mib", "1016"]);
        let gain = (392u64 << 20) / geometry::EXPERT_STRIDE;
        for r in 0..2 {
            // 1,408 billed = today's bytes: the slots move by at most one slot's rounding
            assert!(full[r].abs_diff(base[r]) <= 1, "{full:?} vs {base:?}");
            assert!(
                half[r] + 1 >= base[r] + gain && half[r] <= base[r] + gain + 1,
                "{half:?} vs {base:?} (+{gain})"
            );
        }
    }

    /// FN42b: under `--fn-arena-borrow on` the wide arena's scratch past the narrow rung is the
    /// tier's lendable tail; the narrow arena fits every narrow pass and the static slots are the
    /// permanent cut's (the pack's `plan.json`; skipped where the pack is absent).
    #[test]
    fn fn42b_the_borrow_lends_the_wide_arenas_scratch_tail() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2");
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let off = fnx(&["--fn-serve", "8099", "--fn-arena-borrow", "off"]);
        assert!(!off.arena_borrow(), "the control");
        let narrow = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-max-chunk",
            "2048",
            "--fn-arena-borrow",
            "on",
        ]);
        assert!(
            !narrow.arena_borrow(),
            "the narrow rung has nothing to borrow"
        );
        if !pack.join("plan.json").exists() {
            return;
        }
        for wide in ["4096", "8192"] {
            let p = fnx_ov(&[
                "--fn-serve",
                "8099",
                "--fn-max-chunk",
                wide,
                "--no-fn-arena-borrow",
            ])
            .resolved();
            let f = fnx_ov(&[
                "--fn-serve",
                "8099",
                "--fn-max-chunk",
                wide,
                "--fn-arena-borrow",
                "on",
            ])
            .resolved();
            assert!(f.arena_borrow());
            let bp = plan_boot(pack, &prebuilt, &[], &p).unwrap();
            let b = plan_boot(pack, &prebuilt, &[], &f).unwrap();
            assert!(bp.narrow.is_none());
            let nw = b.narrow.unwrap();
            assert_eq!(b.dr.sizes, bp.dr.sizes, "the same wide arena");
            assert_eq!(
                b.budgets[0].slots, bp.budgets[0].slots,
                "static = the cut's"
            );
            let (at, bytes) = b.dr.sizes.borrow_tail(&nw, 1 << 16);
            let n = bytes / geometry::EXPERT_STRIDE;
            println!(
                "fn42b wide {wide}: arena {} MiB, narrow scratch {} of {} MiB, tail {} MiB at +{} = {n} \
                 lendable slots; static {} -> {} while lent",
                b.dr.sizes.total() >> 20,
                nw.scratch >> 20,
                b.dr.sizes.scratch >> 20,
                bytes >> 20,
                at >> 20,
                b.budgets[0].slots,
                b.budgets[0].slots + n
            );
            assert!(n > 0 && nw.scratch < b.dr.sizes.scratch);
        }
    }

    /// FN99: THE CENSUS of `--fn-embed-placement ram` on the served boot's line (the trunk8 pack,
    /// `--fn-trunk w8`, fp8 KV, elastic 0.5 GiB, 262,144): the embedding class leaves the VRAM
    /// ledger (its bytes + the old remainder become whole static slots), the RAM L2 shrinks by the
    /// same slots and stays the complement, and the host's pinned bytes move by under one stride a
    /// rank (PINNED-NEUTRAL). The negative control: an L2 explicitly below the complement boots at
    /// `vram` (NVMe demand reads) and is REFUSED at `ram`. Skipped where the pack is absent.
    #[test]
    fn fn99_embed_in_ram_is_pinned_neutral_and_refused_when_the_l2_goes_incomplete() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2-trunk8");
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let line = [
            "--fn-serve",
            "8099",
            "--fn-trunk",
            "w8",
            "--fn-kv-dtype",
            "fp8",
            "--fn-vram-elastic-kv-max-gib",
            "0.5",
        ];
        let with = |extra: &[&str]| {
            let mut v = line.to_vec();
            v.extend_from_slice(extra);
            fnx(&v).resolved()
        };
        // FN107: the default is `ram` (the plain boot plans exactly the explicit arm's ledger)
        assert_eq!(DEFAULT_FN_EMBED_PLACEMENT, "ram");
        let plain = with(&[]);
        assert!(plain.fn_embed_placement.is_none() && plain.embed_in_ram());
        assert!(!plain.embed_explicit());
        let d = with(&["--fn-embed-placement", "vram"]);
        assert!(!d.embed_in_ram());
        let r = with(&["--fn-embed-placement", "ram"]);
        assert!(r.embed_in_ram() && r.embed_explicit());
        let bp = plan_boot(pack, &prebuilt, &[], &plain).unwrap();
        let bd = plan_boot(pack, &prebuilt, &[], &d).unwrap();
        let br = plan_boot(pack, &prebuilt, &[], &r).unwrap();
        let s = geometry::EXPERT_STRIDE;
        let g = |x: u64| x as f64 / f64::from(1u32 << 30);
        let emb = br.host.embed_per_rank;
        assert_eq!(bd.host.embed_per_rank, 0);
        assert_eq!(bp.host.embed_per_rank, emb, "the plain boot runs ram");
        assert_eq!(
            bp.budgets.iter().map(|b| b.slots).collect::<Vec<_>>(),
            br.budgets.iter().map(|b| b.slots).collect::<Vec<_>>()
        );
        assert_eq!(
            emb,
            124_160 * H * 2,
            "the embedding half is 124,160 x 2,560 bf16"
        );
        let l2 =
            |b: &Boot| -> Vec<u64> { b.laws.iter().map(|l| u64::from(l.tier.ram_slots)).collect() };
        let (l2d, l2r) = (l2(&bd), l2(&br));
        for i in 0..2 {
            let (a, b) = (&bd.budgets[i], &br.budgets[i]);
            let left =
                a.budget - a.weight_bytes() - a.runtime_bytes() - a.slot_bytes() - a.stage_bytes();
            let gained = b.slots - a.slots;
            assert_eq!(
                gained,
                (left + emb) / s,
                "rank {i}: the freed bytes become whole slots"
            );
            assert_eq!(a.weight_bytes() - b.weight_bytes(), emb);
            assert_eq!(
                l2d[i] - l2r[i],
                gained,
                "rank {i}: the L2 sheds one record a slot gained"
            );
            assert!(l2r[i] >= b.complement(), "rank {i}: the L2 stays complete");
            assert!(l2d[i] >= a.complement());
            println!(
                "fn99 census rank{i}: embed {emb} B ({:.3} GiB) | static slots {} -> {} (+{gained}) | \
                 RAM L2 {} -> {} (complement {} -> {}) | unassigned {left} -> {} B",
                g(emb),
                a.slots,
                b.slots,
                l2d[i],
                l2r[i],
                a.complement(),
                b.complement(),
                b.budget - b.weight_bytes() - b.runtime_bytes() - b.slot_bytes() - b.stage_bytes()
            );
        }
        let (pd, pr) = (bd.host.pinned(&l2d, s), br.host.pinned(&l2r, s));
        let delta = i128::from(pr) - i128::from(pd);
        println!(
            "fn99 census host: pinned {:.3} -> {:.3} GiB of {:.3} ({delta:+} B: + 2 x {emb} embed - \
             2 x {} L2 records)",
            g(pd),
            g(pr),
            g(br.host.pinnable()),
            l2d[0] - l2r[0]
        );
        assert!(
            delta.unsigned_abs() < 2 * u128::from(s),
            "PINNED-NEUTRAL: under one stride a rank"
        );
        assert!(pr <= br.host.pinnable());
        // the negative control: an L2 one record short of the complement -- the PLAIN boot stands
        // down to vram (FN107), the explicit `ram` is refused
        let short = (br.budgets[0].complement() - 1).to_string();
        let sd = plan_boot(
            pack,
            &prebuilt,
            &[],
            &with(&["--xtier-ram-slots", short.as_str()]),
        )
        .unwrap();
        assert_eq!(
            sd.host.embed_per_rank, 0,
            "the plain boot stood down to vram"
        );
        assert_eq!(sd.budgets[0].slots, bd.budgets[0].slots);
        let e = plan_boot(
            pack,
            &prebuilt,
            &[],
            &with(&[
                "--fn-embed-placement",
                "ram",
                "--xtier-ram-slots",
                short.as_str(),
            ]),
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(
            e.contains("--fn-embed-placement ram") && e.contains("refused"),
            "an incomplete L2 under ram must be refused by name: {e:?}"
        );
    }

    /// FN76b: THE CENSUS of the narrow arena at decode -- the served boot's narrow arena (the
    /// trunk8 pack, `--fn-trunk w8`, fp8 KV, elastic 0.5 GiB, 262,144), the scratch every decode-
    /// form pass takes (host-emitted), a prefill chunk's at each rung, and the lendable top (the
    /// narrow scratch past the boundary, 2 MiB aligned) in expert slots at each keep-rows choice;
    /// the boundary always fits under the narrow scratch. Skipped where the pack is absent.
    #[test]
    fn fn76b_census_the_narrow_arenas_top_at_decode() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2-trunk8");
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let f = fnx_ov(&[
            "--fn-serve",
            "8099",
            "--fn-trunk",
            "w8",
            "--fn-kv-dtype",
            "fp8",
            "--fn-vram-elastic",
            "on",
            "--fn-vram-elastic-kv-max-gib",
            "0.5",
        ])
        .resolved();
        let b = plan_boot(pack, &prebuilt, &[], &f).unwrap();
        let nw = b.narrow.unwrap();
        let decls = Decls::load(&prebuilt).unwrap();
        let rules = FlashNextRules::prod().unwrap();
        let pctx = pools_ctx(&f, &[]);
        let mib = |x: u64| x as f64 / 1048576.0;
        println!(
            "fn76b census: pctx {pctx} | wide arena {:.1} MiB (scratch {:.1}) | narrow arena {:.1} MiB \
             = scratch {:.1} + zero {:.1} + ones {:.1} + carry 2 x {:.1}",
            mib(b.dr.sizes.total()),
            mib(b.dr.sizes.scratch),
            mib(nw.total()),
            mib(nw.scratch),
            mib(nw.zero),
            mib(nw.ones),
            mib(nw.carry)
        );
        let stride = geometry::EXPERT_STRIDE;
        let margin = 32u64 << 20;
        for rows in [64u32, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048] {
            let t = Instant::now();
            let (dec, pf, mp) = arena_top_need(&decls, &rules, &f, pctx, rows).unwrap();
            let need = dec.max(pf).max(mp);
            let at = (need + margin).div_ceil(1 << 21) * (1 << 21);
            let bytes = nw.scratch.saturating_sub(at);
            println!(
                "fn76b census rows {rows:>5}: decode forms {:.1} MiB | prefill chunk {:.1} | drafter \
                 prefill {:.1} | boundary (+32 MiB, 2 MiB) {:.1} | lendable {:.1} MiB = {} slots \
                 ({:.0} ms host)",
                mib(dec),
                mib(pf),
                mib(mp),
                mib(at),
                mib(bytes),
                bytes / stride,
                t.elapsed().as_secs_f64() * 1e3
            );
            assert!(dec <= nw.scratch && pf <= nw.scratch && mp <= nw.scratch);
        }
        // the row-scaled buffers outside the arena (prefill-only above the decode rows)
        for c in &b.budgets[0].runtime {
            println!("fn76b census class {} {:.1} MiB", c.name, mib(c.bytes));
        }
    }

    /// FN44b: the decode-time VRAM slots of the served default (static + the borrow's lent range +
    /// the ring's lent slots) under the TWO-LAYOUT borrow vs FN42b's cut layout vs the control
    /// (c2048, no borrow, no ring); the narrow arena is the control boot's arena byte for byte
    /// (every 2,048-row pass runs the c2048 boot's layout); the pack's `plan.json`, skipped where
    /// the pack is absent.
    #[test]
    fn fn44b_the_two_layout_borrow_lends_the_wide_regions_too() {
        let pack = Path::new("/mnt/nvme/saber-packs/qwen38-flashnext-quark-ep2");
        assert!(
            !fnx(&["--fn-serve", "8099"]).fn_borrow_cut,
            "two layouts = the default"
        );
        assert!(
            !fnx(&["--fn-serve", "8099"]).no_fn_lend_fill,
            "the fill = the default"
        );
        if !pack.join("plan.json").exists() {
            return;
        }
        let prebuilt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../prebuilt/gfx1201");
        let ctl = fnx_ov(&[
            "--fn-serve",
            "8099",
            "--fn-max-chunk",
            "2048",
            "--no-fn-arena-borrow",
            "--fn-wide-when",
            "idle",
            "--no-xtier-stage",
        ])
        .resolved();
        let c = plan_boot(pack, &prebuilt, &[], &ctl).unwrap();
        let decode = |args: &[&str]| {
            let f = fnx_ov(args).resolved();
            let b = plan_boot(pack, &prebuilt, &[], &f).unwrap();
            let nw = b.narrow.unwrap();
            let (_, bytes) = if f.fn_borrow_cut {
                b.dr.sizes.borrow_tail(&nw, 1 << 21)
            } else {
                b.dr.sizes.borrow_arena(&nw, 1 << 21)
            };
            let ring = u64::from(f.xtier.stage_ring_slots(f.ring_default_ok(), f.max_chunk()));
            // the runtime classes differ in the arena (lent) and the row-scaled buffers only
            let mut buffers = 0;
            for (x, y) in b.budgets[0].runtime.iter().zip(&c.budgets[0].runtime) {
                assert_eq!(x.name, y.name);
                if x.name == "fn.buffers" {
                    buffers = x.bytes - y.bytes;
                } else if x.name != "fn.arena.prefill" {
                    assert_eq!(x.bytes, y.bytes, "{}", x.name);
                }
            }
            (
                nw,
                b.budgets[0].slots,
                bytes / geometry::EXPERT_STRIDE,
                ring,
                buffers,
            )
        };
        let (nw, st, lent, ring, buffers) = decode(&["--fn-serve", "8099"]);
        let (_, cst, clent, cring, _) = decode(&["--fn-serve", "8099", "--fn-borrow-cut"]);
        println!(
            "fn44b decode slots: control {} | two layouts {st} + {lent} + ring {ring} = {} | cut \
             {cst} + {clent} + ring {cring} = {} | the row-scaled buffers +{} MiB",
            c.budgets[0].slots,
            st + lent + ring,
            cst + clent + cring,
            buffers >> 20
        );
        assert_eq!(
            nw, c.dr.sizes,
            "the narrow arena IS the control boot's arena"
        );
        assert_eq!(st, cst, "the static cut is the same");
        assert!(
            lent > clent,
            "zero / ones / carry at the wide rung are lent too"
        );
        // the gap left = the row-scaled buffers outside the arena (emb / rows / route log / the
        // drafter's hidden rows / the reduce mailbox at 8,192 rows: persistent, not lendable)
        assert!(
            (st + lent + ring + buffers / geometry::EXPERT_STRIDE + 2) >= c.budgets[0].slots,
            "the decode slots short of the control's by more than the wide buffers"
        );
    }

    /// FN44 (the user's word, 2026-09-30 ~20:30 CDT): the served default is c8192 + the arena
    /// borrow + `always` + the 128-slot ring above 2,048 rows; the control is the pre-FN44
    /// default by explicit flags; a 27B boot refuses every one of them, the control included.
    #[test]
    fn fn44_the_wide_chunk_is_the_default_and_the_control_names_the_old_boot() {
        assert_eq!(
            (
                DEFAULT_FN_MAX_CHUNK,
                DEFAULT_FN_WIDE_WHEN,
                DEFAULT_FN_ARENA_BORROW
            ),
            (8192, WideWhen::Always, OnOff::On)
        );
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!(
            (
                d.max_chunk(),
                d.wide_when(),
                d.busy_chunk(),
                d.arena_borrow()
            ),
            (8192, WideWhen::Always, 0, true)
        );
        let law = d
            .xtier
            .resolve(7_553, 4_520, d.ring_default_ok(), d.max_chunk())
            .unwrap();
        let s = law.stage.expect("the ring is on");
        assert_eq!((s.slots, s.buffers, s.above_rows), (128, 2, 2_049));
        // the control = the pre-FN44 boot, by name
        let c = fnx(&[
            "--fn-serve",
            "8099",
            "--fn-max-chunk",
            "2048",
            "--no-fn-arena-borrow",
            "--fn-wide-when",
            "idle",
            "--no-xtier-stage",
        ]);
        assert_eq!(
            (c.max_chunk(), c.busy_chunk(), c.arena_borrow()),
            (2048, 0, false)
        );
        assert_eq!(
            c.xtier
                .resolve(7_768, 4_520, c.ring_default_ok(), c.max_chunk())
                .unwrap()
                .stage,
            None
        );
        // c2048 alone keeps FN43c's 48-slot ring at every 2,048-row chunk
        let n = fnx(&["--fn-serve", "8099", "--fn-max-chunk", "2048"]);
        let ns = n
            .xtier
            .resolve(7_768, 4_520, n.ring_default_ok(), n.max_chunk())
            .unwrap()
            .stage
            .unwrap();
        assert_eq!((ns.slots, ns.above_rows), (48, 2_048));
        // the borrow's default stands down where it cannot run; explicit `on` stays on (the boot
        // refuses it there by name); `--no-fn-arena-borrow` = `--fn-arena-borrow off`
        assert!(!fnx(&["--fn-one-process"]).arena_borrow());
        assert!(!fnx(&["--fn-poison", "255"]).arena_borrow());
        assert!(fnx(&["--fn-one-process", "--fn-arena-borrow", "on"]).arena_borrow());
        assert!(!fnx(&["--no-fn-arena-borrow"]).arena_borrow());
        // a 27B boot: no flag passes; every explicit wide flag is refused, default value or not
        assert!(fnx(&[]).refuse_on_27b().is_ok());
        for a in [
            &["--fn-max-chunk", "2048"][..],
            &["--fn-max-chunk", "8192"],
            &["--fn-wide-when", "idle"],
            &["--fn-wide-when", "always"],
            &["--fn-arena-borrow", "on"],
            &["--fn-arena-borrow", "off"],
            &["--no-fn-arena-borrow"],
        ] {
            assert!(fnx(a).refuse_on_27b().is_err(), "{a:?} on a 27B pack");
        }
    }

    #[test]
    fn the_async_tier_plan_is_the_default_and_sync_is_the_control() {
        // FN59b: the cheap victim selection + the submitted copy batches are the default ...
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!(d.xtier.xtier_plan, "async");
        assert!(d.xtier.plan_fast() && d.xtier.plan_submit());
        // ... `--xtier-plan sync` is the pre-FN59b law, by name; `fast` = the selection alone
        let c = fnx(&["--fn-serve", "8099", "--xtier-plan", "sync"]);
        assert!(!c.xtier.plan_fast() && !c.xtier.plan_submit());
        let f = fnx(&["--fn-serve", "8099", "--xtier-plan", "fast"]);
        assert!(f.xtier.plan_fast() && !f.xtier.plan_submit());
    }

    #[test]
    fn the_exact_ledger_lrfu_and_the_pack_profile_are_the_default_and_the_control_names_fn7s() {
        // FN18c (hold 2's paired win): the exact runtime ledger, LRFU 0.98 and the PACK's
        // real-traffic profile are the served placement ...
        let pack = Path::new("/packs/fn");
        let d = fnx(&["--fn-serve", "8099"]);
        assert!(DEFAULT_FN_LEDGER_EXACT && d.ledger_exact());
        assert_eq!(
            d.profile_path(pack),
            pack.join("profile/route_profile.json")
        );
        let law = d
            .xtier
            .resolve(7_802, 4_486, d.ring_default_ok(), d.max_chunk())
            .unwrap();
        assert_eq!(law.tier.lrfu_lambda, 0.98);
        assert!(law.profile);
        // ... the control is FN7's planned ledger + FN8's LRU + FN7's FN3-prompt profile, by name
        let fn7 = "~/saber-cache/ws-fn7/profile/route_profile.json";
        let c = fnx(&[
            "--fn-serve",
            "8099",
            "--no-fn-ledger-exact",
            "--xtier-policy-lru",
            "--xtier-profile",
            fn7,
        ]);
        assert!(!c.ledger_exact());
        assert_eq!(c.profile_path(pack), Path::new(fn7));
        assert_eq!(
            c.xtier
                .resolve(7_404, 4_884, c.ring_default_ok(), c.max_chunk())
                .unwrap()
                .tier
                .lrfu_lambda,
            0.0
        );
        // `--fn-profile` still names a file; `--xtier-profile` wins over it
        let fp = fnx(&["--fn-profile", "/p/a.json"]);
        assert_eq!(fp.profile_path(pack), Path::new("/p/a.json"));
        let both = fnx(&["--fn-profile", "/p/a.json", "--xtier-profile", "/p/b.json"]);
        assert_eq!(both.profile_path(pack), Path::new("/p/b.json"));
    }

    #[test]
    fn fn_profile_user_names_the_added_member_pack_stays_the_default_and_the_recorder_is_off() {
        // FN65: the default is the pack's own profile; `pack` names it; `user` = the ADDED member
        let pack = Path::new("/packs/fn");
        assert_eq!(
            fnx(&["--fn-serve", "8099"]).profile_path(pack),
            pack.join(DEFAULT_PROFILE_IN_PACK)
        );
        assert_eq!(
            fnx(&["--fn-profile", "pack"]).profile_path(pack),
            pack.join(DEFAULT_PROFILE_IN_PACK)
        );
        assert_eq!(
            fnx(&["--fn-profile", "user"]).profile_path(pack),
            pack.join("profile/route_profile.user.json")
        );
        assert_ne!(USER_PROFILE_IN_PACK, DEFAULT_PROFILE_IN_PACK);
        // `--xtier-profile` still wins over a keyword
        assert_eq!(
            fnx(&["--fn-profile", "user", "--xtier-profile", "/p/b.json"]).profile_path(pack),
            Path::new("/p/b.json")
        );
        // the recorder: off by default, refused beside FN18's trace and on a 27B pack
        let d = fnx(&["--fn-serve", "8099"]);
        assert!(d.fn_route_record.is_none() && d.fn_route_trace.is_none());
        let r = fnx(&["--fn-serve", "8099", "--fn-route-record", "/r"]);
        let c = r.record_cfg(Path::new("/r"), 48, 10, 512);
        assert_eq!(c.rotate_steps, DEFAULT_FN_ROUTE_RECORD_STEPS);
        assert_eq!(c.max_bytes, 16 << 30);
        assert!(fnx(&["--fn-route-record", "/r"]).refuse_on_27b().is_err());
        assert!(std::panic::catch_unwind(|| fnx(&[
            "--fn-route-record",
            "/r",
            "--fn-route-trace",
            "/t"
        ]))
        .is_err());
    }

    /// FN106d: `--fn-pinned-pages` absent = `huge` (the default since FN106d's hold), `shmem` =
    /// today's `hipHostMalloc` (the control), any other name refused by name; a 27B boot refuses
    /// the flag at all.
    #[test]
    fn the_pinned_pages_default_is_huge_and_shmem_is_the_control() {
        assert_eq!(DEFAULT_FN_PINNED_PAGES, saber_hip::PinnedPages::Huge);
        let plain = fnx(&["--fn-serve", "8099"]).resolved();
        assert_eq!(plain.pinned_pages().unwrap(), saber_hip::PinnedPages::Huge);
        assert!(fnx(&[]).refuse_on_27b().is_ok());
        let huge = fnx(&["--fn-serve", "8099", "--fn-pinned-pages", "huge"]).resolved();
        assert_eq!(huge.pinned_pages().unwrap(), saber_hip::PinnedPages::Huge);
        let h27 = fnx(&["--fn-pinned-pages", "huge"]);
        assert!(
            h27.refuse_on_27b().is_err(),
            "a 27B boot refuses --fn-pinned-pages"
        );
        let ctl = fnx(&["--fn-pinned-pages", "shmem"]);
        assert_eq!(ctl.pinned_pages().unwrap(), saber_hip::PinnedPages::Shmem);
        let bad = fnx(&["--fn-pinned-pages", "thp"]);
        assert!(format!("{}", bad.pinned_pages().unwrap_err()).contains("shmem | huge"));
    }

    /// FN112: `--fn-overlap` absent = [`DEFAULT_FN_OVERLAP`] (`on`, the default), `off` = the
    /// control (the reads after the wait), any other value refused by name; a 27B boot refuses it.
    #[test]
    fn the_overlap_scheduler_is_the_default_and_off_is_the_control() {
        const { assert!(DEFAULT_FN_OVERLAP) };
        assert!(fnx(&["--fn-serve", "8099"]).resolved().overlap().unwrap());
        assert!(fnx(&["--fn-overlap", "on"]).overlap().unwrap());
        assert!(!fnx(&["--fn-overlap", "off"]).overlap().unwrap());
        let bad = fnx(&["--fn-overlap", "yes"]);
        assert!(format!("{}", bad.overlap().unwrap_err()).contains("off | on"));
        assert!(fnx(&["--fn-overlap", "on"]).refuse_on_27b().is_err());
    }

    /// FN117b: `--fn-hand` absent = `all` (S1 + S2 + S6, THE DEFAULT since the FN117b flip);
    /// `none` = the control (the pre-FN117b host hand); an unknown item is refused by name; a 27B
    /// boot refuses the flag and the skew diagnostic.
    #[test]
    fn the_host_hand_slices_are_the_default_and_none_is_the_control() {
        assert_eq!(super::super::hand::DEFAULT_FN_HAND, "all");
        let d = fnx(&["--fn-serve", "8099"]).resolved().hand().unwrap();
        assert!(d.s1 && d.s2 && d.s6 && !d.check);
        assert!(!fnx(&["--fn-hand", "none"]).hand().unwrap().any());
        let a = fnx(&["--fn-hand", "all"]).hand().unwrap();
        assert!(a.s1 && a.s2 && a.s6 && !a.check);
        assert!(fnx(&["--fn-hand", "s9"]).hand().is_err());
        assert!(fnx(&["--fn-hand", "s1"]).refuse_on_27b().is_err());
        assert!(fnx(&["--fn-hand-skew-us", "50"]).refuse_on_27b().is_err());
    }

    #[test]
    fn the_mtp_auto_law_is_the_served_default_and_fn_mtp_0_is_the_control() {
        // FN16f (the user's word, 2026-09-29): the SERVED default is MTP auto (FN16's table) ...
        assert_eq!(DEFAULT_SERVE_MTP, FnMtp::Auto);
        let serve = fnx(&["--fn-serve", "8099"]).resolved();
        assert_eq!(serve.fn_mtp, FnMtp::Auto);
        let law = serve.mtp_law().unwrap().expect("auto has a law");
        assert_eq!(
            law.kmax(),
            3,
            "AUTO_TABLE's widest width is 4 rows (k = 3 at b1)"
        );
        assert_eq!(
            serve.fn_mtp_table, None,
            "the default runs verify::AUTO_TABLE unchanged"
        );
        // ... `--fn-mtp 0` is the SPEC-0 control arm ...
        let ctl = fnx(&["--fn-serve", "8099", "--fn-mtp", "0"]).resolved();
        assert_eq!(ctl.fn_mtp, FnMtp::Off);
        assert!(ctl.mtp_law().unwrap().is_none());
        // ... fixed k and the table override are unchanged ...
        let k2 = fnx(&["--fn-serve", "8099", "--fn-mtp", "2"]).resolved();
        assert_eq!(k2.fn_mtp, FnMtp::K(2));
        let t = fnx(&["--fn-serve", "8099", "--fn-mtp-table", "1:4,2:4,4:3,8:1"]).resolved();
        assert_eq!(t.fn_mtp, FnMtp::Auto);
        assert!(t.mtp_law().unwrap().is_some());
        // ... the prompt-set harness stays SPEC 0 (its byte gates are SPEC-0 gates) ...
        assert_eq!(DEFAULT_HARNESS_MTP, FnMtp::Off);
        let h = fnx(&["--fn-out", "/tmp/x"]).resolved();
        assert_eq!(h.fn_mtp, FnMtp::Off);
        assert_eq!(fnx(&["--fn-mtp", "auto"]).resolved().fn_mtp, FnMtp::Auto);
        // ... and a 27B boot is unchanged: no `--fn-*` flag (or the old default `0`) passes, any
        // MTP arm is refused
        assert!(fnx(&[]).refuse_on_27b().is_ok());
        assert!(fnx(&["--fn-mtp", "0"]).refuse_on_27b().is_ok());
        assert!(fnx(&["--fn-mtp", "auto"]).refuse_on_27b().is_err());
        assert!(fnx(&["--fn-mtp", "3"]).refuse_on_27b().is_err());
    }

    #[test]
    fn fn92_the_table_depth_is_the_default_and_adaptive_is_an_arm_at_b1_only() {
        // FN92: absent = the table's b1 cell (AUTO_TABLE unchanged, no extra pre-captured form)
        assert_eq!(DEFAULT_FN_MTP_DEPTH, "table");
        let d = fnx(&["--fn-serve", "8099"]).resolved();
        let law = d.mtp_law().unwrap().unwrap();
        assert!(!law.adaptive && law.adaptive_ceil().is_none());
        assert_eq!(law.widths_at(1), vec![4]);
        assert_eq!(law.kmax(), 3);
        // the arm: b1 runs every width 2..=5 (ceiling = the verify cap), b >= 2 the table's
        let a = fnx(&["--fn-serve", "8099", "--fn-mtp-depth", "adaptive"]).resolved();
        let la = a.mtp_law().unwrap().unwrap();
        assert!(la.adaptive);
        assert_eq!(la.widths_at(1), vec![2, 3, 4, 5]);
        assert_eq!(
            [2, 4, 8].map(|b| la.widths_at(b)),
            [vec![3], vec![3], vec![3]]
        );
        assert_eq!(la.kmax(), 4, "the pools hold the ceiling's width");
        assert!(la.to_string().contains("adaptive depth at b1"));
        // refused: beside the EMA cap, on a fixed k / SPEC 0, an unknown law, on a 27B pack
        assert!(fnx(&[
            "--fn-mtp-depth",
            "adaptive",
            "--fn-mtp-dynamic",
            "--fn-serve",
            "1"
        ])
        .resolved()
        .mtp_law()
        .is_err());
        assert!(fnx(&["--fn-mtp", "3", "--fn-mtp-depth", "adaptive"])
            .resolved()
            .mtp_law()
            .is_err());
        assert!(fnx(&["--fn-serve", "1", "--fn-mtp-depth", "deep"])
            .resolved()
            .mtp_law()
            .is_err());
        assert!(fnx(&["--fn-mtp-depth", "adaptive"])
            .refuse_on_27b()
            .is_err());
    }

    #[test]
    fn the_device_chain_is_the_default_and_the_host_chain_is_the_control() {
        // FN22: no flag = the device-side chain; `--no-fn-mtp-device-chain` = FN16's host chain;
        // the check / selftest / chain-graph arms are off by default; a 27B boot refuses them all
        let d = fnx(&["--fn-serve", "8099"]).resolved();
        assert!(!d.no_fn_mtp_device_chain);
        assert!(!d.fn_mtp_chain_check && !d.fn_mtp_chain_selftest && !d.fn_mtp_chain_graph);
        assert!(
            fnx(&["--fn-serve", "8099", "--no-fn-mtp-device-chain"])
                .resolved()
                .no_fn_mtp_device_chain
        );
        for a in [
            "--no-fn-mtp-device-chain",
            "--fn-mtp-chain-check",
            "--fn-mtp-chain-selftest",
            "--fn-mtp-chain-graph",
        ] {
            assert!(fnx(&[a]).refuse_on_27b().is_err(), "{a} on a 27B pack");
        }
    }

    #[test]
    fn the_producer_rr_arm_is_off_by_default_and_a_27b_boot_refuses_it() {
        // FN37: no flag = the shipped reduce + the down's own norm (DEFAULT_FN_HC_RR_PRODUCER);
        // `--fn-hc-rr-producer` = the arm, `--no-fn-hc-rr-producer` = the explicit control; both
        // are Flash-Next flags away from their default, so a 27B pack refuses each (as it does
        // FN30c's mover-hold pair)
        let d = fnx(&["--fn-serve", "8099"]).resolved();
        assert_eq!(d.hc_rr_producer(), DEFAULT_FN_HC_RR_PRODUCER);
        assert!(fnx(&["--fn-serve", "8099", "--fn-hc-rr-producer"])
            .resolved()
            .hc_rr_producer());
        assert!(!fnx(&["--fn-serve", "8099", "--no-fn-hc-rr-producer"])
            .resolved()
            .hc_rr_producer());
        for a in [
            "--fn-hc-rr-producer",
            "--no-fn-hc-rr-producer",
            "--fn-mover-hold",
            "--no-fn-mover-hold",
        ] {
            assert!(fnx(&[a]).refuse_on_27b().is_err(), "{a} on a 27B pack");
        }
    }

    #[test]
    fn the_boot_fill_route_and_the_pool_bounce_arm_parse_by_name() {
        // FN71c: `--xtier-boot-fill` (auto | pool | host; the default is the constant) and the
        // pool-bounce ARM `--xtier-nvme-p2p` (off by default); the pool's allocate dir default
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!(
            d.xtier.xtier_boot_fill,
            crate::xtier_flags::DEFAULT_XTIER_BOOT_FILL
        );
        assert!(!d.xtier.nvme_p2p());
        assert_eq!(
            d.xtier.xtier_p2p_dir,
            std::path::PathBuf::from(crate::xtier_flags::DEFAULT_XTIER_P2P_DIR)
        );
        assert_eq!(
            fnx(&["--xtier-boot-fill", "host"]).xtier.boot_fill_pool(),
            (false, false)
        );
        assert_eq!(
            fnx(&["--xtier-boot-fill", "auto"]).xtier.boot_fill_pool(),
            (true, false)
        );
        assert_eq!(
            fnx(&["--xtier-boot-fill", "pool"]).xtier.boot_fill_pool(),
            (true, true)
        );
        assert!(fnx(&["--xtier-nvme-p2p", "on"]).xtier.nvme_p2p());
    }

    #[test]
    fn the_mover_marks_default_is_word_and_layer_is_the_control() {
        // FN72b: no flag = `word` (the default since FN72b); `layer` (a HIP event per layer
        // batch) is the control; `word` / `step` are refused by name with two-queue / hsa-lane,
        // where the default falls back to the mover's own marks
        use saber_mem::device_mover::MarkLaw;
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!(
            d.xtier.xtier_mover_marks,
            crate::xtier_flags::DEFAULT_XTIER_MOVER_MARKS
        );
        assert_eq!(d.xtier.mover_marks().unwrap(), MarkLaw::Word);
        assert_eq!(
            fnx(&["--xtier-mover-marks", "layer"])
                .xtier
                .mover_marks()
                .unwrap(),
            MarkLaw::Layer
        );
        assert_eq!(
            fnx(&["--xtier-mover-marks", "word"])
                .xtier
                .mover_marks()
                .unwrap(),
            MarkLaw::Word
        );
        assert_eq!(
            fnx(&["--xtier-mover-marks", "step"])
                .xtier
                .mover_marks()
                .unwrap(),
            MarkLaw::Step
        );
        for m in ["two-queue", "hsa-lane"] {
            let e = fnx(&["--xtier-mover-marks", "word", "--xtier-mover", m])
                .xtier
                .mover_marks()
                .unwrap_err()
                .to_string();
            assert!(e.contains("one-stream mover only"), "{e}");
            assert_eq!(
                fnx(&["--xtier-mover", m]).xtier.mover_marks().unwrap(),
                MarkLaw::Layer
            );
        }
    }

    #[test]
    fn the_sdma_lanes_default_is_shared_and_split_is_an_arm() {
        // FN72: no flag = ROCclr's pooled queues (the control); `split` / `split-both` give the
        // mover's copy stream (and the step stream) a hardware queue of their own; a 27B pack
        // accepts the default and refuses the arms; an unknown form is refused at parse
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!(d.fn_sdma_lanes, DEFAULT_FN_SDMA_LANES);
        assert_eq!(DEFAULT_FN_SDMA_LANES, "shared");
        for a in ["split", "split-both"] {
            assert_eq!(fnx(&["--fn-sdma-lanes", a]).fn_sdma_lanes, a);
            assert!(
                fnx(&["--fn-sdma-lanes", a]).refuse_on_27b().is_err(),
                "{a} on a 27B pack"
            );
        }
        assert!(fnx(&["--fn-sdma-lanes", "shared"]).refuse_on_27b().is_ok());
        #[derive(clap::Parser)]
        struct C {
            #[command(flatten)]
            f: FnFlags,
        }
        assert!(
            <C as clap::Parser>::try_parse_from(["saber", "--fn-sdma-lanes", "engine1"]).is_err()
        );
    }

    #[test]
    fn the_fused_moe_train_is_the_default_and_legacy_is_the_control() {
        // FN30d: no flag = the fused train (its 240 KiB sync class declared); `--fn-moe-train
        // legacy` = the control (a 256 B stub); a 27B pack accepts the default and refuses legacy
        use super::super::exec::MoeTrain;
        let d = fnx(&["--fn-serve", "8099"]);
        assert_eq!(d.fn_moe_train, MoeTrain::Fused);
        assert_eq!(d.fn_moe_train.to_string(), DEFAULT_FN_MOE_TRAIN);
        assert!(moe_sync_class(&d, 0) > 256);
        let l = fnx(&["--fn-serve", "8099", "--fn-moe-train", "legacy"]);
        assert_eq!(l.fn_moe_train, MoeTrain::Legacy);
        assert_eq!(moe_sync_class(&l, 0), 256);
        assert!(fnx(&[]).refuse_on_27b().is_ok());
        assert!(fnx(&["--fn-moe-train", "fused"]).refuse_on_27b().is_ok());
        assert!(fnx(&["--fn-moe-train", "legacy"]).refuse_on_27b().is_err());
    }

    #[test]
    fn the_int2_draft_head_is_the_default_and_bf16_is_the_control() {
        // FN33: no flag = the int2 head at K 80 on the device chain (its ledger class declared);
        // `--fn-draft-head bf16` = FN22's pass (a 256 B stub class); the host chain never builds it
        use saber_kernels::fn33::{pack_bytes, DraftHead, DEFAULT_K};
        let d = fnx(&["--fn-serve", "8099"]).resolved();
        assert_eq!(
            (d.fn_draft_head, d.fn_draft_head_k),
            (DraftHead::Int2, DEFAULT_K)
        );
        assert!(draft_int2_on(&d));
        assert_eq!(
            draft_int2_class(&d),
            pack_bytes(super::super::exec::VOCAB / 2)
        );
        let c = fnx(&["--fn-serve", "8099", "--fn-draft-head", "bf16"]).resolved();
        assert!(!draft_int2_on(&c));
        assert_eq!(draft_int2_class(&c), 256);
        let h = fnx(&["--fn-serve", "8099", "--no-fn-mtp-device-chain"]).resolved();
        assert!(!draft_int2_on(&h));
        for a in [["--fn-draft-head", "bf16"], ["--fn-draft-head-k", "1"]] {
            assert!(fnx(&a).refuse_on_27b().is_err(), "{a:?} on a 27B pack");
        }
    }

    #[test]
    fn inv_freq_is_the_formula_within_one_ulp() {
        for (i, &b) in INV_FREQ_1E7_ROT64.iter().enumerate() {
            let f = 1.0 / 1.0e7f64.powf((2 * i) as f64 / 64.0);
            let d = (i64::from(b) - i64::from((f as f32).to_bits())).abs();
            assert!(d <= 1, "inv_freq[{i}] {d} ulp from the formula");
        }
    }
}
