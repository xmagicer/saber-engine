//! Boot-time memory gauges for the native target and resident drafter.
use std::fmt::Write;

use super::DraftBuf;
use saber_model::dflash_graph::DraftRope;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
pub(super) struct RankMemory {
    pub rank: u32,
    pub pci_bdf: Option<String>,
    pub resident_weight_bytes: u64,
    pub kv_pool_allocated_bytes: u64,
}

/// The upload callback reports tensor payload bytes, not transfer padding or arena size.
/// Count aliased embeddings/heads at their sole upload; an individually uploaded copy
/// (when sharing is disabled) consumes memory on the card and is counted independently.
pub(super) fn weight_payload(name: &str, bytes: u64) -> u64 {
    if name == DraftRope::RECORD {
        0
    } else {
        bytes
    }
}

/// A native drafter owns one contiguous allocation for all K layers and one for all V layers.
/// If these names no longer describe two whole allocations, leave the measurement unavailable.
pub(super) fn draft_kv_bytes(
    named: &BTreeMap<String, DraftBuf>,
    allocation_len: impl Fn(usize) -> Option<u64>,
) -> Option<u64> {
    let k = named.get("kv.k")?;
    let v = named.get("kv.v")?;
    if k.idx == v.idx || k.off != 0 || v.off != 0 {
        return None;
    }
    let (k_bytes, v_bytes) = (allocation_len(k.idx)?, allocation_len(v.idx)?);
    if (k_bytes, v_bytes) != (k.bytes, v.bytes) {
        return None;
    }
    k_bytes.checked_add(v_bytes)
}

pub(super) fn render(ranks: &mut [RankMemory]) -> String {
    if ranks.is_empty() {
        return String::new();
    }
    ranks.sort_unstable_by_key(|r| r.rank);
    let mut out = String::new();
    for (metric, help, get) in [
        ("saber:resident_weight_bytes",
         "Resident uploaded target and drafter weight tensor payload bytes on this rank, excluding RoPE and padding.",
         (|r: &RankMemory| r.resident_weight_bytes) as fn(&RankMemory) -> u64),
        ("saber:kv_pool_allocated_bytes",
         "Allocated GPU bytes in the target and native drafter K/V pools on this rank, including free pages.",
         |r: &RankMemory| r.kv_pool_allocated_bytes),
    ] {
        writeln!(out, "# HELP {metric} {help}\n# TYPE {metric} gauge").unwrap();
        for row in ranks.iter() {
            write!(out, "{metric}{{rank=\"{}\"", row.rank).unwrap();
            if let Some(bdf) = &row.pci_bdf {
                let label = bdf.to_ascii_lowercase().replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
                write!(out, ",pci_bdf=\"{label}\"").unwrap();
            }
            writeln!(out, "}} {}", get(row)).unwrap();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_kv_counts_whole_roots_once_and_omits_unknown_layouts() {
        let sizes = [32, 48];
        let mut named = BTreeMap::from([
            (
                "kv.k".into(),
                DraftBuf {
                    idx: 0,
                    off: 0,
                    bytes: 32,
                },
            ),
            (
                "kv.v".into(),
                DraftBuf {
                    idx: 1,
                    off: 0,
                    bytes: 48,
                },
            ),
            (
                "kv.k.l1".into(),
                DraftBuf {
                    idx: 0,
                    off: 16,
                    bytes: 16,
                },
            ),
        ]);
        let bytes = |i: usize| sizes.get(i).copied();
        assert_eq!(draft_kv_bytes(&named, bytes), Some(80));
        named.get_mut("kv.v").unwrap().idx = 0;
        assert_eq!(draft_kv_bytes(&named, bytes), None);
        named.get_mut("kv.v").unwrap().idx = 2;
        assert_eq!(draft_kv_bytes(&named, bytes), None);
        named.remove("kv.v");
        assert_eq!(draft_kv_bytes(&named, bytes), None);
    }

    #[test]
    fn uploaded_payloads_omit_rope_and_count_shared_records_at_their_upload() {
        let target = [("embed_tokens.weight", 200), ("layers.0.weight", 300)];
        let draft = [
            (DraftRope::RECORD, 80),
            ("head.rerank.w8", 100),
            ("head.rerank.ws", 4),
        ];
        assert_eq!(
            target
                .into_iter()
                .chain(draft)
                .map(|(n, b)| weight_payload(n, b))
                .sum::<u64>(),
            604
        );
    }

    #[test]
    fn both_gauges_keep_rank_device_identity_and_their_own_sizes() {
        assert!(render(&mut []).is_empty());
        let mut rows = [
            RankMemory {
                rank: 1,
                pci_bdf: None,
                resident_weight_bytes: 21,
                kv_pool_allocated_bytes: 23,
            },
            RankMemory {
                rank: 0,
                pci_bdf: Some("0000:AB:00.0".into()),
                resident_weight_bytes: 11,
                kv_pool_allocated_bytes: 13,
            },
        ];
        let text = render(&mut rows);
        for (metric, a, b) in [
            ("saber:resident_weight_bytes", 11, 21),
            ("saber:kv_pool_allocated_bytes", 13, 23),
        ] {
            assert_eq!(text.matches(&format!("# TYPE {metric} gauge")).count(), 1);
            assert!(text
                .lines()
                .any(|s| s == format!("{metric}{{rank=\"0\",pci_bdf=\"0000:ab:00.0\"}} {a}")));
            assert!(text
                .lines()
                .any(|s| s == format!("{metric}{{rank=\"1\"}} {b}")));
        }
    }
}
