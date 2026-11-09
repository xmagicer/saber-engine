//! Opt-in target auxiliary-state capture for offline drafter validation.
//!
//! Input tokens and BF16 rows are committed together in an events journal. The directory is
//! private and must be new; discontinuous positions are refused instead of exporting partial
//! prefix-cache state. This diagnostic is never enabled by the serving defaults.
use saber_sched::SeqId;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

struct SequenceFiles {
    next: u32,
    aux: File,
    tokens: File,
    events: File,
}

/// A private, append-only export of each sequence's actual committed target inputs.
pub struct TargetAuxCapture {
    directory: PathBuf,
    row_bytes: usize,
    sequences: BTreeMap<SeqId, SequenceFiles>,
}

fn private_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

impl TargetAuxCapture {
    /// Create a new export directory. An existing directory is deliberately never reused.
    pub fn new(directory: &Path, hidden: u32, layers: &[u32]) -> io::Result<Self> {
        if hidden == 0 || layers.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "empty target feature geometry",
            ));
        }
        if let Some(parent) = directory.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(directory)?;
        let row_bytes = hidden as usize * layers.len() * 2;
        let metadata = serde_json::json!({
            "schema": "saber-target-aux/1", "dtype": "BF16", "byte_order": "little",
            "hidden_size": hidden, "target_layer_ids": layers,
            "row_bytes": row_bytes, "position_semantics": "input token state after target layer id-1",
            "diagnostic_only": true, "requires_prefix_cache_off": true
        });
        serde_json::to_writer_pretty(private_file(&directory.join("metadata.json"))?, &metadata)?;
        Ok(Self {
            directory: directory.to_owned(),
            row_bytes,
            sequences: BTreeMap::new(),
        })
    }

    /// Append contiguous input-token features. Rejected draft rows and the un-forwarded bonus
    /// token must not be included. Journal publication happens after both payload writes.
    pub fn record(
        &mut self,
        seq: SeqId,
        position: u32,
        tokens: &[u32],
        aux: &[u8],
        kind: &str,
    ) -> io::Result<()> {
        if tokens.is_empty() || aux.len() != tokens.len() * self.row_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "capture token/auxiliary row length mismatch",
            ));
        }
        if !self.sequences.contains_key(&seq) {
            if position != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "capture starts after position zero; disable prefix caching",
                ));
            }
            let stem = format!("seq-{}", seq.0);
            self.sequences.insert(
                seq,
                SequenceFiles {
                    next: 0,
                    aux: private_file(&self.directory.join(format!("{stem}.aux.bf16")))?,
                    tokens: private_file(&self.directory.join(format!("{stem}.tokens.u32le")))?,
                    events: private_file(&self.directory.join(format!("{stem}.events.jsonl")))?,
                },
            );
        }
        let files = self.sequences.get_mut(&seq).expect("created above");
        if files.next != position {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "capture gap or duplicate for {seq}: expected position {}, got {position}",
                    files.next
                ),
            ));
        }
        let next =
            position
                .checked_add(u32::try_from(tokens.len()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "capture row overflow")
                })?)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "capture position overflow")
                })?;
        files.aux.write_all(aux)?;
        for token in tokens {
            files.tokens.write_all(&token.to_le_bytes())?;
        }
        serde_json::to_writer(
            &mut files.events,
            &serde_json::json!({"seq":seq.0,"position":position,"rows":tokens.len(),"kind":kind}),
        )?;
        files.events.write_all(b"\n")?;
        files.next = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "saber-aux-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
    fn cleanup(path: &Path) {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            std::fs::remove_file(entry.path()).unwrap();
        }
        std::fs::remove_dir(path).unwrap();
    }
    #[test]
    fn committed_rows_and_tokens_append_in_position_order() {
        let path = path();
        {
            let mut sink = TargetAuxCapture::new(&path, 2, &[1, 16]).unwrap();
            sink.record(SeqId(7), 0, &[13, 17], &[1; 16], "prefill")
                .unwrap();
            sink.record(SeqId(7), 2, &[19], &[2; 8], "decode").unwrap();
        }
        assert_eq!(
            std::fs::read(path.join("seq-7.aux.bf16")).unwrap(),
            [vec![1; 16], vec![2; 8]].concat()
        );
        assert_eq!(
            std::fs::read(path.join("seq-7.tokens.u32le")).unwrap(),
            [13u32, 17, 19]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        );
        let text = std::fs::read_to_string(path.join("seq-7.events.jsonl")).unwrap();
        let events: Vec<serde_json::Value> = text
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        assert_eq!(events[1]["position"], 2);
        assert_eq!(events[1]["rows"], 1);
        cleanup(&path);
    }
    #[test]
    fn gaps_duplicates_and_existing_output_are_refused() {
        let path = path();
        {
            let mut sink = TargetAuxCapture::new(&path, 1, &[1]).unwrap();
            assert!(sink.record(SeqId(1), 4, &[1], &[0; 2], "decode").is_err());
            sink.record(SeqId(1), 0, &[1], &[0; 2], "prefill").unwrap();
            assert!(sink.record(SeqId(1), 0, &[1], &[0; 2], "decode").is_err());
            assert!(sink.record(SeqId(1), 2, &[1], &[0; 2], "decode").is_err());
            assert!(sink.record(SeqId(1), 1, &[1], &[0; 1], "decode").is_err());
            assert!(TargetAuxCapture::new(&path, 1, &[1]).is_err());
        }
        cleanup(&path);
    }
    #[test]
    fn separate_sequences_cannot_share_payload_files() {
        let path = path();
        {
            let mut sink = TargetAuxCapture::new(&path, 1, &[1]).unwrap();
            sink.record(SeqId(1), 0, &[11], &[1, 2], "prefill").unwrap();
            sink.record(SeqId(2), 0, &[22], &[3, 4], "prefill").unwrap();
        }
        assert_eq!(std::fs::read(path.join("seq-1.aux.bf16")).unwrap(), [1, 2]);
        assert_eq!(std::fs::read(path.join("seq-2.aux.bf16")).unwrap(), [3, 4]);
        cleanup(&path);
    }
}
