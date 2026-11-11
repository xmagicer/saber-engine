//! What a session park costs on the card, next to what a step costs.
//!
//! Ignored by default: it wants a GPU, the TP=1 device-final pack and the committed plugins, and it
//! is a measurement rather than a gate. Run it inside the dev image, under the shared lock:
//!
//! ```text
//! flock ~/saber/.cache/gpu.lock.1 tools/dev.sh \
//!   cargo test --release -p saber-engine --test park_bench -- --ignored --nocapture
//! ```
//!
//! The question it answers is the one the parking policy needs: a park evicts a session so that
//! another can be admitted, and the eviction is paid on the **step thread**. So the number that
//! matters is not GB/s, it is the park and the restore next to the step they displace.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use saber_engine::{spawn, DecodeSlot, EngineConfig, ForwardRunner, RowRequest};
use saber_mem::ledger::{Ledger, Pool};
use saber_sched::SeqId;
use saber_serve::engine::TokenId;

/// The TP=1 device-final tier. `saber_mem::pack::PACK_ROOT` is where every tool looks.
fn pack() -> PathBuf {
    PathBuf::from(saber_mem::pack::PACK_ROOT).join("qwen38-27b-nvfp4-devfinal-tp1")
}

/// The HIP runtime to dlopen. Inside the dev image ROCm is not on the loader path, so the full
/// path is the one that works there; on a host with ROCm installed the soname is. A filesystem
/// probe rather than an environment variable — the engine reads none, and a measurement that is
/// not reproducible from its command line is not one.
fn hip_lib() -> String {
    const FULL: &str = "/opt/rocm/lib/libamdhip64.so.7";
    if std::path::Path::new(FULL).exists() {
        FULL.to_owned()
    } else {
        "libamdhip64.so.7".to_owned()
    }
}

fn prebuilt() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../prebuilt/gfx1201"
    ))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn p90(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[(v.len() as f64 * 0.9) as usize % v.len()]
}

#[test]
#[ignore = "measurement: needs a GPU, the TP=1 device-final pack and the prebuilt plugins"]
fn park_and_restore_next_to_a_step() {
    const CTX: u32 = 8192;
    const ITERS: usize = 20;

    // One image at 8k context: 512 pages of KV plus one GDN state slot -- 403 MiB on this
    // checkpoint. 1 GiB of pinned staging is more than that, and the engine refuses to START if it
    // is not, which is the point of stating the number here rather than deriving it silently.
    let staging_bytes: u64 = 1 << 30;

    // One view of one card, exactly as `saber engine` builds it. `Pool::HostPinned` is NOT declared
    // here: the engine declares it at startup, wide enough for the loader's staging ring plus this
    // buffer, and reserves `engine:park-staging` against it before the allocation.
    let mut ledger = Ledger::new();
    ledger
        .declare(Pool::Vram(0), 32 << 30)
        .expect("declare vram");
    ledger
        .declare(Pool::HostPageable, 48 << 30)
        .expect("declare pageable");
    ledger.declare(Pool::Nvme, 64 << 30).expect("declare nvme");

    let cfg = EngineConfig {
        pack: pack(),
        // SPEC=0: this bench measures what a park displaces from the shipped step.
        decode_widths: vec![1],
        decode_bands: Vec::new(),
        // One card: parking is a TP=1 route, and the engine refuses it above that by name (half a
        // session's KV lives on the other rank's card).
        world: 1,
        peer: None,
        rank: 0,
        prebuilt: prebuilt(),
        hip_lib: hip_lib(),
        rccl_lib: None,
        host_ar: false,
        dspine_socket: None,
        draft_capture_dir: None,
        draft_capture_layers: Vec::new(),
        device: 0,
        budget_mib: 26_000,
        max_seqs: 3,
        max_ctx: CTX,
        prefill_rows: vec![2048],
        // Batch 1, captured, ids fed on the device: the step a park displaces is the shipped one,
        // not the eager A side.
        decode_batches: vec![1],
        capture: true,
        device_feed: true,
        live_row_attn: true,
        draft_block: 0,
        draft_pack: None,
        ar_exact: false,
        host_trace: false,
        whole_step_frame: false,
        frame_digest: false,
        race_check: false,
        race_shadow: false,
        poison_boot: None,
        constant_seed_control: false,
        gather_per_row: false,
        propose_at: saber_engine::ProposeAt::AfterCommit,
        fixed_step_wait: false,
        fail_after_steps: 0,
        host_wait: saber_exec::HostWait::Aimed,
        gdn_state_f16: false,
        draft_store_narrow: false,
        draft_store_natural_off: false,
        draft_store_check: false,
        draft_store_deferred: false,
        draft_kv_paged: false,
        draft_embed_share: false,
        selector_rank: saber_engine::selector_rank::SelectorRank::Walk,
        spec_tree: saber_engine::spec_tree::SpecTreeForm::Off,
        spec_tree_sampled: false,
        spec_tree_sampled_control: false,
        gdn_replay: false,
        host_levers: saber_engine::host_levers::HostLevers::default(),
        device_step: saber_engine::device_step::DeviceStep::default(),
        prefill_ar_overlap: 0,
        prefill_ar_push_blocks: 0,
        small_chunk_gemm: 0,
        small_chunk_band: false,
        shared_prefill_arena: false,
        shared_head_arena: false,
        mid_chunk_gemm: false,
        mid_chunk_atiled: false,
        skinny_gate: false,
        decode_pf: false,
        head_m64: false,
        drafter_gateup_m64: 0,
        ar_wpar: false,
        ar_fuse: false,
        b1_arms: Default::default(),
        c4_arms: Default::default(),
        step_arms: Default::default(),
        mixed_class_proxy: false,
        decode_fusion: saber_engine::device::DecodeFusion::Off,
        vision_max_patches: 0,
        vision_lazy: false,
        drafter_rope_index: false,
        head: saber_engine::head_share::TargetHead::Mxfp4,
        kv_pool_blocks: (CTX.div_ceil(16)) * 2 + 2,
        step_deadline: Duration::from_secs(120),
        park_staging_bytes: staging_bytes,
        // One slot: the shipped default. The two-slot overlap is a separate measurement.
        park_staging_slots: 1,
        park_tier_bytes: 0,
        park_image_law: saber_engine::ParkImageLaw::Live,
        park_probe: false,
        kv_headroom_mib: 1024,
    };
    let (mut engine, ledger) = match spawn(cfg, ledger) {
        Ok(pair) => pair,
        Err(e) => {
            println!("SKIP: the engine would not start: {e}");
            return;
        }
    };
    let facts = engine.facts().clone();
    println!("model     {}", facts.model);
    if let Some(banner) = facts.partial_banner() {
        println!("{banner}");
    }
    // `kv_block_bytes` is the PER-LAYER figure the plan sums; the arena strides by all layers at
    // once, so the page the image actually copies is `layers/att-layers x` bigger. The K8 ledger
    // said "32 KiB pages" from this field and the pages were 512 KiB; printed from the image below.
    println!(
        "session   max_ctx {}, {} block(s) per sequence, GDN slot {} MiB, KV pool {} blocks",
        facts.max_ctx,
        facts.max_blocks,
        facts.gdn_bytes_per_slot >> 20,
        facts.kv_pool_blocks
    );
    println!(
        "park      staging {} MiB pinned, reserved in the ledger before it was allocated",
        facts.park_staging_bytes >> 20
    );
    assert!(engine.can_park(), "this build should be able to park");

    // A sequence holding the whole 8k context: all 512 pages, so the image is the worst case the
    // staging buffer was sized for. The prefill fills all but the last chunk of them with real
    // tokens -- real bytes rather than zeroes a copy engine might shortcut -- and the tail is the
    // headroom the decode steps below write into. A decode AT `max_ctx` has nowhere to put its
    // token: the block table is `max_ctx / block_size` wide, so position `max_ctx` is off the end
    // of it, and the attention kernel faults on a page that is not mapped.
    let blocks: Vec<u32> = (0..facts.max_blocks).collect();
    let seq = SeqId(1);
    engine
        .admit(seq, CTX, &blocks)
        .expect("admit a full-context sequence");
    let prompt: Vec<TokenId> = (0..2048u32).map(|i| 1000 + (i % 30000)).collect();
    let filled = CTX - 2048;
    let mut done = 0u32;
    while done < filled {
        let last = done + 2048 >= filled;
        engine
            .prefill(seq, &prompt, done, last, &RowRequest::greedy())
            .expect("prefill chunk");
        done += 2048;
    }

    // The step this displaces, measured the same way. `context_len` is what the engine already has
    // resident, which after the prefill is exactly the prompt and then one more per decode — a
    // batch that says anything else is refused by name, which is how this loop's first cut was
    // caught reusing a stale one.
    let mut step_ms = Vec::with_capacity(ITERS);
    let last = *prompt.last().expect("non-empty");
    let mut ctx = filled;
    let step = |engine: &mut saber_engine::HipForward, ctx: u32| {
        engine
            .decode(&[DecodeSlot::single(seq, last, ctx)], 1, 1)
            .expect("decode");
    };
    for _ in 0..ITERS {
        let t = Instant::now();
        step(&mut engine, ctx);
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        ctx += 1;
    }

    // The ENCODED seam, which is the one `RealExecutor` uses: the blob is built straight out of the
    // staging buffer and published straight back out of it, so the numbers are the serving path's
    // and not a convenience wrapper's.
    let mut park_ms = Vec::with_capacity(ITERS);
    let mut restore_ms = Vec::with_capacity(ITERS);
    let mut image_bytes = 0u64;
    for _ in 0..ITERS {
        let t = Instant::now();
        let blob = engine.park_encoded(seq).expect("park");
        park_ms.push(t.elapsed().as_secs_f64() * 1e3);
        image_bytes = blob.len() as u64;

        let t = Instant::now();
        engine.restore_encoded(seq, blob).expect("restore");
        restore_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let mib = image_bytes as f64 / (1 << 20) as f64;
    let kv_bytes = image_bytes - facts.gdn_bytes_per_slot;
    println!();
    println!(
        "image     {mib:.1} MiB ({image_bytes} B), {} regions: {} KV page(s) of {} KiB + a {} MiB \
         slot",
        facts.max_blocks + 1,
        facts.max_blocks,
        (kv_bytes / u64::from(facts.max_blocks)) >> 10,
        facts.gdn_bytes_per_slot >> 20
    );
    println!("| what | median | p90 | GB/s |");
    println!("|---|---:|---:|---:|");
    for (what, v) in [
        ("park", &mut park_ms),
        ("restore", &mut restore_ms),
        ("decode step", &mut step_ms),
    ] {
        let med = median(v);
        let hi = p90(v);
        let gbs = if what == "decode step" {
            f64::NAN
        } else {
            image_bytes as f64 / (med / 1e3) / 1e9
        };
        if gbs.is_nan() {
            println!("| {what} | {med:.2} ms | {hi:.2} ms | — |");
        } else {
            println!("| {what} | {med:.2} ms | {hi:.2} ms | {gbs:.2} |");
        }
    }
    let med_step = median(&mut step_ms);
    println!();
    println!(
        "park+restore = {:.2} ms = {:.1} steps of {:.2} ms",
        median(&mut park_ms) + median(&mut restore_ms),
        (median(&mut park_ms) + median(&mut restore_ms)) / med_step,
        med_step
    );

    // ---- off the step loop ---------------------------------------------------------------------
    // The question the asynchronous path exists to answer: what does a park cost the sequences that
    // are NOT parking? So the park is driven the way the scheduler drives it -- one poll per step --
    // with a second sequence decoding on every one of those steps, and what is reported is that
    // sequence's ms/step across the park against its ms/step with no park in flight.
    let neighbour = SeqId(2);
    engine
        .admit(neighbour, 16, &[facts.max_blocks])
        .expect("admit a neighbour");
    let nb_prompt: Vec<TokenId> = (0..16u32).map(|i| 2000 + i).collect();
    engine
        .prefill(neighbour, &nb_prompt, 0, true, &RowRequest::greedy())
        .expect("prefill the neighbour");
    let nb_step = |engine: &mut saber_engine::HipForward, ctx: u32| -> f64 {
        let t = Instant::now();
        engine
            .decode(&[DecodeSlot::single(neighbour, nb_prompt[0], ctx)], 1, 1)
            .expect("neighbour decode");
        t.elapsed().as_secs_f64() * 1e3
    };
    // Baseline: the neighbour alone, nothing in flight.
    let mut nb_ctx = nb_prompt.len() as u32;
    let mut base_ms = Vec::with_capacity(60);
    for _ in 0..60 {
        base_ms.push(nb_step(&mut engine, nb_ctx));
        nb_ctx += 1;
    }

    // Now with a park in flight, polled once per step, exactly as `poll_transfers` does.
    assert!(
        engine.parks_async(),
        "a build with a staging buffer parks asynchronously"
    );
    let t = Instant::now();
    engine.park_begin(seq).expect("issue the park");
    let issue_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut during_ms = Vec::new();
    let mut poll_ms = Vec::new();
    let mut blob = None;
    let t0 = Instant::now();
    while blob.is_none() {
        let t = Instant::now();
        blob = engine.park_poll(seq).expect("poll the park");
        poll_ms.push(t.elapsed().as_secs_f64() * 1e3);
        during_ms.push(nb_step(&mut engine, nb_ctx));
        nb_ctx += 1;
    }
    let park_wall = t0.elapsed().as_secs_f64() * 1e3;
    let steps_of_park = during_ms.len();
    let blob = blob.expect("the park finished");

    // ...and the restore, the same way.
    let t = Instant::now();
    engine.restore_begin(seq, blob).expect("issue the restore");
    let restore_issue_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut during_restore = Vec::new();
    let t0 = Instant::now();
    loop {
        let done = engine.restore_poll(seq).expect("poll the restore");
        during_restore.push(nb_step(&mut engine, nb_ctx));
        nb_ctx += 1;
        if done {
            break;
        }
    }
    let restore_wall = t0.elapsed().as_secs_f64() * 1e3;

    let base = median(&mut base_ms.clone());
    let during = median(&mut during_ms.clone());
    let during_r = median(&mut during_restore.clone());
    println!();
    println!("**Off the step loop**, polled once per step with a neighbour decoding on every one:");
    println!();
    println!("| what | value |");
    println!("|---|---:|");
    println!("| park issue (returns immediately) | {issue_ms:.3} ms |");
    println!("| restore issue (verifies, then returns) | {restore_issue_ms:.2} ms |");
    println!(
        "| one poll, median / p90 / max | {:.3} / {:.3} / {:.2} ms |",
        median(&mut poll_ms.clone()),
        p90(&mut poll_ms.clone()),
        poll_ms.iter().copied().fold(0.0, f64::max)
    );
    println!("| steps the park took | {steps_of_park} |");
    println!("| park wall | {park_wall:.2} ms |");
    println!("| restore wall | {restore_wall:.2} ms |");
    println!(
        "| **neighbour ms/step, no park** | **{base:.2} ms** (p90 {:.2}) |",
        p90(&mut base_ms)
    );
    println!(
        "| **neighbour ms/step, during the park** | **{during:.2} ms** ({:+.2} ms, {:+.1}%) |",
        during - base,
        100.0 * (during - base) / base
    );
    println!(
        "| **neighbour ms/step, during the restore** | **{during_r:.2} ms** ({:+.2} ms, {:+.1}%) |",
        during_r - base,
        100.0 * (during_r - base) / base
    );
    engine.release(neighbour);

    // ---- where the time goes -------------------------------------------------------------------
    // The image is 513 regions and 512 of them are 32 KiB KV pages; the GDN slot is the one big
    // one. A second session admitted with ONE page has the same slot and 1/512 of the pages, so the
    // difference between the two parks is what the 511 extra pages cost, and the remainder is the
    // fixed cost of the slot. (Its pages are not prefilled -- this measures the copy, not the
    // model, and a `hipMemcpy` does not care what the bytes are.)
    let tiny = SeqId(3);
    engine
        .admit(tiny, 16, &[facts.max_blocks])
        .expect("admit a one-page sequence");
    let mut tiny_park = Vec::with_capacity(ITERS);
    let mut tiny_restore = Vec::with_capacity(ITERS);
    let mut tiny_bytes = 0u64;
    for _ in 0..ITERS {
        let t = Instant::now();
        let blob = engine
            .park_encoded(tiny)
            .expect("park the one-page session");
        tiny_park.push(t.elapsed().as_secs_f64() * 1e3);
        tiny_bytes = blob.len() as u64;
        let t = Instant::now();
        engine.restore_encoded(tiny, blob).expect("restore it");
        tiny_restore.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let tp = median(&mut tiny_park);
    let tr = median(&mut tiny_restore);
    let pages = f64::from(facts.max_blocks - 1);
    println!();
    println!("| image | regions | bytes | park | restore |",);
    println!("|---|---:|---:|---:|---:|");
    println!(
        "| {} pages + slot | {} | {:.1} MiB | {:.2} ms | {:.2} ms |",
        facts.max_blocks,
        facts.max_blocks + 1,
        image_bytes as f64 / (1 << 20) as f64,
        median(&mut park_ms),
        median(&mut restore_ms)
    );
    println!(
        "| 1 page + slot | 2 | {:.1} MiB | {tp:.2} ms | {tr:.2} ms |",
        tiny_bytes as f64 / (1 << 20) as f64
    );
    println!(
        "| per extra page | | | {:.3} ms | {:.3} ms |",
        (median(&mut park_ms) - tp) / pages,
        (median(&mut restore_ms) - tr) / pages
    );
    engine.release(tiny);

    // The bytes came back: one more decode after a restore must still run, at the context the
    // restored state itself carries.
    step(&mut engine, ctx);
    engine.release(seq);
    // The pinned staging is still on the engine's books here, which is the point of billing it.
    let _ = &ledger;
}

/// Does a second staging slot pay for itself? 403 MiB of pinned memory is the price.
///
/// With one slot a park and a restore take turns: the second is told `Busy` and retries next step,
/// so a park followed by a restore is `park wall + restore wall`. With two they have a slot each and
/// overlap. What has to be measured is both halves of that trade — the wall it saves, and what the
/// extra traffic costs the sequence that is doing neither.
#[test]
#[ignore = "measurement: needs a GPU, the TP=1 device-final pack and the prebuilt plugins"]
fn two_staging_slots_against_one() {
    const CTX: u32 = 8192;
    for slots in [1u32, 2] {
        // One image at 8k is 403 MiB, so two slots is 806 MiB of page-locked host memory. Stated,
        // because it is the whole cost of the option.
        let staging_bytes: u64 = u64::from(slots) * (1 << 30);
        let mut ledger = Ledger::new();
        ledger
            .declare(Pool::Vram(0), 32 << 30)
            .expect("declare vram");
        ledger
            .declare(Pool::HostPageable, 48 << 30)
            .expect("declare pageable");
        ledger.declare(Pool::Nvme, 64 << 30).expect("declare nvme");
        let cfg = EngineConfig {
            pack: pack(),
            world: 1,
            rank: 0,
            peer: None,
            prebuilt: prebuilt(),
            hip_lib: hip_lib(),
            rccl_lib: None,
            host_ar: false,
            dspine_socket: None,
            draft_capture_dir: None,
            draft_capture_layers: Vec::new(),
            device: 0,
            budget_mib: 26_000,
            max_seqs: 4,
            max_ctx: CTX,
            prefill_rows: vec![2048],
            decode_batches: vec![1],
            decode_widths: vec![1],
            decode_bands: Vec::new(),
            capture: true,
            device_feed: true,
            live_row_attn: true,
            draft_block: 0,
            draft_pack: None,
            ar_exact: false,
            host_trace: false,
            whole_step_frame: false,
            frame_digest: false,
            race_check: false,
            race_shadow: false,
            poison_boot: None,
            constant_seed_control: false,
            gather_per_row: false,
            propose_at: saber_engine::ProposeAt::AfterCommit,
            fixed_step_wait: false,
            fail_after_steps: 0,
            host_wait: saber_exec::HostWait::Aimed,
            gdn_state_f16: false,
            draft_store_narrow: false,
            draft_store_natural_off: false,
            draft_store_check: false,
            draft_store_deferred: false,
            draft_kv_paged: false,
            draft_embed_share: false,
            selector_rank: saber_engine::selector_rank::SelectorRank::Walk,
            spec_tree: saber_engine::spec_tree::SpecTreeForm::Off,
            spec_tree_sampled: false,
            spec_tree_sampled_control: false,
            gdn_replay: false,
            host_levers: saber_engine::host_levers::HostLevers::default(),
            device_step: saber_engine::device_step::DeviceStep::default(),
            prefill_ar_overlap: 0,
            prefill_ar_push_blocks: 0,
            small_chunk_gemm: 0,
            small_chunk_band: false,
            shared_prefill_arena: false,
            shared_head_arena: false,
            mid_chunk_gemm: false,
            mid_chunk_atiled: false,
            skinny_gate: false,
            decode_pf: false,
            head_m64: false,
            drafter_gateup_m64: 0,
            ar_wpar: false,
            ar_fuse: false,
            b1_arms: Default::default(),
            c4_arms: Default::default(),
            step_arms: Default::default(),
            mixed_class_proxy: false,
            decode_fusion: saber_engine::device::DecodeFusion::Off,
            vision_max_patches: 0,
            vision_lazy: false,
            drafter_rope_index: false,
            head: saber_engine::head_share::TargetHead::Mxfp4,
            kv_pool_blocks: (CTX.div_ceil(16)) * 3 + 2,
            step_deadline: Duration::from_secs(120),
            park_staging_bytes: staging_bytes,
            park_staging_slots: slots,
            park_tier_bytes: 0,
            park_image_law: saber_engine::ParkImageLaw::Live,
            park_probe: false,
            kv_headroom_mib: 1024,
        };
        let (mut engine, _ledger) = match spawn(cfg, ledger) {
            Ok(pair) => pair,
            Err(e) => {
                println!("SKIP slots={slots}: the engine would not start: {e}");
                return;
            }
        };
        let facts = engine.facts().clone();
        assert_eq!(facts.park_staging_slots, slots);

        // Two full-context sessions and a neighbour that decodes throughout.
        let per = facts.max_blocks;
        let a = SeqId(1);
        let bb = SeqId(2);
        let nb = SeqId(3);
        let prompt: Vec<TokenId> = (0..2048u32).map(|i| 1000 + (i % 30000)).collect();
        let filled = CTX - 2048;
        for (seq, lo) in [(a, 0), (bb, per)] {
            let blocks: Vec<u32> = (lo..lo + per).collect();
            engine.admit(seq, CTX, &blocks).expect("admit");
            let mut done = 0u32;
            while done < filled {
                engine
                    .prefill(
                        seq,
                        &prompt,
                        done,
                        done + 2048 >= filled,
                        &RowRequest::greedy(),
                    )
                    .expect("prefill");
                done += 2048;
            }
        }
        engine
            .admit(nb, 16, &[2 * per])
            .expect("admit the neighbour");
        let nb_prompt: Vec<TokenId> = (0..16u32).map(|i| 2000 + i).collect();
        engine
            .prefill(nb, &nb_prompt, 0, true, &RowRequest::greedy())
            .expect("prefill the neighbour");
        let mut nb_ctx = nb_prompt.len() as u32;
        let step = |engine: &mut saber_engine::HipForward, ctx: u32| -> f64 {
            let t = Instant::now();
            engine
                .decode(
                    &[DecodeSlot {
                        seq: nb,
                        input: nb_prompt[0],
                        context_len: ctx,
                        // SPEC=0: no draft, and the previous step committed the one token it ran.
                        draft: Vec::new(),
                        committed_before: 1,
                        sampling: RowRequest::greedy(),
                        decider: None,
                    }],
                    1,
                    1,
                )
                .expect("neighbour decode");
            t.elapsed().as_secs_f64() * 1e3
        };
        let mut base_ms = Vec::with_capacity(40);
        for _ in 0..40 {
            base_ms.push(step(&mut engine, nb_ctx));
            nb_ctx += 1;
        }

        // Park B once so there is something to restore.
        let blob_b = {
            engine.park_begin(bb).expect("issue B's park");
            loop {
                if let Some(b) = engine.park_poll(bb).expect("poll B") {
                    break b;
                }
                nb_ctx += 1;
                step(&mut engine, nb_ctx - 1);
            }
        };

        // Now ask for A's park and B's restore at the same time, and poll both every step.
        let t0 = Instant::now();
        engine.park_begin(a).expect("issue A's park");
        let restore_started = engine.restore_begin(bb, blob_b).is_ok();
        let mut during = Vec::new();
        let mut park_done = false;
        let mut restore_done = !restore_started;
        while !park_done || !restore_done {
            if !park_done && engine.park_poll(a).expect("poll A").is_some() {
                park_done = true;
            }
            if !restore_done && engine.restore_poll(bb).expect("poll B") {
                restore_done = true;
            }
            during.push(step(&mut engine, nb_ctx));
            nb_ctx += 1;
        }
        let both_wall = t0.elapsed().as_secs_f64() * 1e3;

        let base = median(&mut base_ms.clone());
        let d = median(&mut during.clone());
        println!();
        println!(
            "slots={slots} ({} MiB pinned): a park and a restore asked for together",
            staging_bytes >> 20
        );
        println!("  overlapped at once      {restore_started}");
        println!("  steps to finish both    {}", during.len());
        println!("  wall for both           {both_wall:.2} ms");
        println!(
            "  neighbour ms/step        {base:.2} -> {d:.2} ({:+.2} ms, {:+.1}%)",
            d - base,
            100.0 * (d - base) / base
        );
        for seq in [a, bb, nb] {
            engine.release(seq);
        }
    }
}
