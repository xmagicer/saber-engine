//! Offline pack verification using the runtime reader, without a GPU or model startup.
use std::path::PathBuf;

use saber_mem::dio::OpenMode;
use saber_mem::pack::PackReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        return Err("usage: saber-pack-verify rank0.pack [rank1.pack ...]".into());
    }
    let mut identity = None;
    let mut ranks = std::collections::BTreeSet::new();
    for path in paths {
        let pack = PackReader::open_with(&path, OpenMode::Buffered)?;
        let h = pack.header();
        let key = (h.tp_size, h.plan_hash, h.tier_id);
        if identity.as_ref().is_some_and(|old| *old != key) || !ranks.insert(h.rank) {
            return Err("input packs disagree about tier/plan/TP or repeat a rank".into());
        }
        identity = Some(key);
        pack.verify_all()?;
        println!(
            "{}",
            serde_json::json!({
                "passed": true, "path": path, "rank": h.rank, "tp_size": h.tp_size,
                "records": h.record_count, "bytes": h.file_bytes, "tier_id": hex::encode(h.tier_id)
            })
        );
    }
    Ok(())
}
