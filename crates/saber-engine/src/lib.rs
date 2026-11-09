//! `saber-engine` — the P3 seam: a real [`StepExecutor`](saber_sched::StepExecutor) over the
//! whole-model forward.
//!
//! `saber-sched` decides *what* a step is (which bucket, which rows, which chunk of which prompt)
//! and `saber-exec` knows *how* to launch one. Until now nothing implemented the trait between
//! them: every gate above the scheduler ran on a mock executor, and every gate below it ran on a
//! one-shot `saber forward`. This crate is the join.
//!
//! # What is real here, and what is not
//!
//! Real: the plan, the weights (every one bound out of the device-final pack by name), the arena
//! and its slot reuse across 64 layers, the per-sequence KV and GDN state buffers, the launch count
//! per step against the plan's **declared** budget, and the wall clock.
//!
//! Not yet: two ops have no code object on this branch (`qk_norm_rope` and `attn_gate_quant`, 16 of
//! each per body plan — one per attention layer), and the K/V append is state that no op writes at
//! all, so the attention layers do not compute their value and **the tokens are not the model's
//! tokens**. That is a statement about the
//! kernel set, not about the engine — `saber forward` says the same thing with the same banner. The
//! gate is therefore *structural and relative*: the same prompt through `saber forward` and through
//! the serve path must produce the **same** ids, both partial.
//!
//! # Why there is a thread in here
//!
//! Every HIP handle is `!Send` by construction (`PhantomData<*mut ()>` in `saber-hip`), and
//! `StepExecutor` is `Send` because the scheduler's step loop owns it on a plain `std::thread`.
//! Those two cannot both be satisfied by a struct that holds a `Stream`. So the device side lives
//! on **one thread of its own** — the architecture's own rule, "confine every HIP call to one OS
//! thread, structurally" — and [`HipForward`] is a channel handle to it. That also gives the host
//! a seam it can test through: [`ForwardRunner`] is the trait, [`NullForward`] is the stub, and
//! nothing above them knows which one is underneath. The device module carries no feature gate: it
//! compiles on a host with no ROCm (`saber-hip` loads `libamdhip64` lazily and answers
//! `NotAvailable`), so a `cargo test` on this box builds exactly the code the card runs.
//!
//! # The API this crate wants and does not have
//!
//! `saber forward` in `saber-cli` is a *command*, not a library: the pass, the index tables and the
//! head's three extra launches are private to it. This crate therefore re-derives them against the
//! same `saber-exec` planner APIs rather than calling them. Every piece of that duplication is
//! listed in the branch report; the honest fix is for the pass to move into `saber-exec` (H owns
//! it) and for this crate to drive it.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod aux_tap;
pub mod capacity;
pub mod census;
pub mod device;
pub mod device_step;
pub mod draft;
pub mod draft_capture;
pub mod draft_inputs;
pub mod draft_plan;
pub mod draft_runs;
pub mod draft_tier;
pub mod executor;
pub mod forward;
pub mod fuse27;
pub mod gdn_cuf;
pub mod gdn_replay;
pub mod head_share;
pub mod host_levers;
pub mod host_trace;
pub mod index;
pub mod lookup;
pub mod mixed_class;
pub mod null;
pub mod park;
pub mod race;
pub mod selector_rank;
pub mod small_chunk;
pub mod spec_tree;
pub mod spec_tree_step;
pub mod startup;
pub mod tp2;
pub mod vision;

pub use aux_tap::{AuxTapPlan, TapAt, TapVerdict};
pub use capacity::{render_waste, Capacity, Granted, KvGeometry, Waste};
pub use census::{engine_defs, shutdown_census, ENGINE_CLEAN};
pub use device::{
    run_worker, spawn, DraftCounters, DraftCounts, DraftReplayOut, EngineConfig, HipForward,
    ParkImageLaw,
};
pub use draft::{
    AcceptError, Acceptance, Drafter, EngineDrafter, FixedDrafter, Greedy, NoDrafter,
    RecordedDrafter, Verdict,
};
pub use draft_inputs::{DraftGeometry, DraftSeqStep, DraftStepIndex};
pub use draft_plan::{DraftArena, DraftPhase, DraftPlan, DraftStep};
pub use executor::{EngineCounters, RealExecutor, SlotDraws};
pub use forward::{
    drawn_rows, AuxTapFacts, Decider, DecodeSlot, DraftResidency, ForwardError, ForwardFacts,
    ForwardRunner, MixedChunk, ParkDiscard, ProbeDepth, Proposal, ProposeAt, RowDecider, RowProbe,
    RowRequest, SeqImages, SlotCommit, StepOutcome, StepPath, StepRows, TokenFeed,
};
pub use index::{index_table, IndexError, SeqFacts, StepIndex};
pub use lookup::{LookupConfig, LookupCounts, LookupDrafter, SeqLookup};
pub use null::NullForward;
pub use park::{ImageCounts, ImageError, ParkImage};
pub use startup::{Phase, PhaseRecord, Phases};
pub use tp2::{Command, PeerConfig, PeerRole, Pick, RankLink};
