//! Host transport for the experimental DSpine proposal backend.
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use saber_sched::SeqId;
use serde::Deserialize;
use serde_json::{json, Value};

pub(crate) const LAYERS: [u32; 5] = [2, 17, 32, 47, 62];
const BYTES_PER_ROW: usize = 5120 * LAYERS.len() * 2;
const MAX_FRAME_ROWS: usize = 2048;
const MAX_REPLY: u64 = 4096;

#[derive(Deserialize)]
struct WireNode {
    token: u32,
    #[serde(deserialize_with = "Option::deserialize")]
    parent: Option<usize>,
}

#[derive(Deserialize)]
struct Reply {
    ok: bool,
    error: Option<String>,
    tokens: Option<Vec<u32>>,
    tree: Option<Vec<WireNode>>,
}

struct Offer {
    seq: SeqId,
    position: u32,
    anchor: u32,
    chain: Vec<u32>,
    tree: crate::spec_tree::TreePlan,
}

pub(crate) struct DspineSocket {
    offer: Option<Offer>,
    connection: BufReader<UnixStream>,
    sequence: Option<SeqId>,
    next: u32,
    failed: bool,
}

impl DspineSocket {
    pub fn connect(path: &Path, timeout: Duration) -> io::Result<Self> {
        let connection = UnixStream::connect(path)?;
        connection.set_read_timeout(Some(timeout))?;
        connection.set_write_timeout(Some(timeout))?;
        Ok(Self {
            connection: BufReader::new(connection),
            sequence: None,
            next: 0,
            failed: false,
            offer: None,
        })
    }

    fn exchange(&mut self, header: Value, payload: &[u8]) -> io::Result<Reply> {
        if self.failed {
            return Err(io::Error::other(
                "DSpine connection failed an earlier exchange",
            ));
        }
        // After a partial write, timeout, or malformed ACK the stream cannot be reused.
        self.failed = true;
        let socket = self.connection.get_mut();
        let mut frame = serde_json::to_vec(&header)?;
        frame.push(b'\n');
        socket.write_all(&frame)?;
        socket.write_all(payload)?;
        let mut response = Vec::new();
        (&mut self.connection)
            .take(MAX_REPLY)
            .read_until(b'\n', &mut response)?;
        if response.last() != Some(&b'\n') {
            return Err(io::Error::other("truncated or oversized DSpine reply"));
        }
        let reply: Reply = serde_json::from_slice(&response)?;
        if !reply.ok {
            return Err(io::Error::other(
                reply
                    .error
                    .unwrap_or_else(|| "DSpine refused request".into()),
            ));
        }
        self.failed = false;
        Ok(reply)
    }

    pub fn append(&mut self, seq: SeqId, position: u32, bytes: &[u8]) -> io::Result<()> {
        if self.sequence.is_some_and(|current| current != seq)
            || position != self.next
            || bytes.is_empty()
            || bytes.len() % BYTES_PER_ROW != 0
        {
            return Err(io::Error::other(
                "invalid or noncontiguous DSpine target features",
            ));
        }
        for frame in bytes.chunks(MAX_FRAME_ROWS * BYTES_PER_ROW) {
            let rows = (frame.len() / BYTES_PER_ROW) as u32;
            let next = self
                .next
                .checked_add(rows)
                .ok_or_else(|| io::Error::other("DSpine position overflow"))?;
            self.exchange(
                json!({"op":"append", "seq":seq.0, "position":self.next, "rows":rows}),
                frame,
            )?;
            self.next = next;
            self.sequence = Some(seq);
        }
        Ok(())
    }

    pub fn propose(
        &mut self,
        seq: SeqId,
        position: u32,
        anchor: u32,
        width: u32,
        vocab: u32,
        tree: bool,
    ) -> io::Result<Vec<u32>> {
        self.offer = None;
        if self.sequence != Some(seq) || self.next != position || anchor >= vocab {
            return Err(io::Error::other(
                "DSpine proposal has an invalid anchor or position",
            ));
        }
        if width == 0 {
            return Ok(Vec::new());
        }
        let mut request =
            json!({"op":"propose","seq":seq.0,"position":position,"anchor":anchor,"width":width});
        if tree {
            if width != 15 {
                return Err(io::Error::other("DSpine tree requires all 16 target rows"));
            }
            request["tree_nodes"] = json!(15);
        }
        let reply = self.exchange(request, &[])?;
        let tokens = reply
            .tokens
            .ok_or_else(|| io::Error::other("DSpine reply omitted tokens"))?;
        if tokens.len() != width as usize || tokens.iter().any(|t| *t >= vocab) {
            return Err(io::Error::other(
                "DSpine reply has the wrong verify width or contains an out-of-vocabulary token",
            ));
        }
        if tree {
            let entries = reply
                .tree
                .ok_or_else(|| io::Error::other("DSpine reply omitted the requested tree"))?;
            let nodes: Vec<_> = entries
                .iter()
                .map(|entry| (entry.parent, entry.token))
                .collect();
            let tree = crate::spec_tree::TreePlan::from_external(anchor, &nodes, vocab)
                .map_err(io::Error::other)?;
            self.offer = Some(Offer {
                seq,
                position,
                anchor,
                chain: tokens.clone(),
                tree,
            });
        } else if reply.tree.is_some() {
            return Err(io::Error::other("unexpected tree on a chain request"));
        }
        Ok(tokens)
    }

    pub fn take_tree(
        &mut self,
        seq: SeqId,
        position: u32,
        anchor: u32,
        chain: &[u32],
    ) -> Option<crate::spec_tree::TreePlan> {
        self.offer
            .take()
            .filter(|offer| {
                offer.seq == seq
                    && offer.position == position
                    && offer.anchor == anchor
                    && offer.chain == chain
            })
            .map(|offer| offer.tree)
    }

    pub fn clear(&mut self, seq: SeqId) -> io::Result<()> {
        if self.sequence == Some(seq) {
            self.exchange(json!({"op":"clear","seq":seq.0}), &[])?;
            self.sequence = None;
            self.next = 0;
            self.offer = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> (DspineSocket, BufReader<UnixStream>) {
        let (client, server) = UnixStream::pair().unwrap();
        let timeout = Some(Duration::from_secs(5));
        for connection in [&client, &server] {
            connection.set_read_timeout(timeout).unwrap();
            connection.set_write_timeout(timeout).unwrap();
        }
        (
            DspineSocket {
                connection: BufReader::new(client),
                sequence: None,
                next: 0,
                failed: false,
                offer: None,
            },
            BufReader::new(server),
        )
    }

    fn header(peer: &mut BufReader<UnixStream>) -> Value {
        let mut line = String::new();
        peer.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    }

    #[test]
    fn only_contiguous_committed_rows_and_the_host_anchor_reach_the_proposer() {
        let (mut client, mut peer) = pair();
        let seq = SeqId(42);
        let worker = std::thread::spawn(move || {
            assert_eq!(
                header(&mut peer),
                json!({"op":"append","seq":42,"position":0,"rows":2})
            );
            let mut payload = vec![0; 2 * BYTES_PER_ROW];
            peer.read_exact(&mut payload).unwrap();
            assert!(payload[..BYTES_PER_ROW].iter().all(|&v| v == 12));
            assert!(payload[BYTES_PER_ROW..].iter().all(|&v| v == 34));
            peer.get_mut().write_all(b"{\"ok\":true}\n").unwrap();
            assert_eq!(
                header(&mut peer),
                json!({"op":"propose","seq":42,"position":2,"anchor":15,"width":3})
            );
            peer.get_mut()
                .write_all(b"{\"ok\":true,\"tokens\":[1,2,3]}\n")
                .unwrap();
            assert_eq!(header(&mut peer), json!({"op":"clear","seq":42}));
            peer.get_mut().write_all(b"{\"ok\":true}\n").unwrap();
        });
        let mut payload = vec![12; 2 * BYTES_PER_ROW];
        payload[BYTES_PER_ROW..].fill(34);
        assert!(client.append(seq, 1, &payload).is_err());
        client.append(seq, 0, &payload).unwrap();
        assert!(client.append(seq, 1, &payload).is_err());
        assert!(client.propose(seq, 3, 15, 7, 100, false).is_err());
        assert_eq!(
            client.propose(seq, 2, 15, 3, 100, false).unwrap(),
            [1, 2, 3]
        );
        client.clear(seq).unwrap();
        assert!(client.sequence.is_none());
        worker.join().unwrap();
    }

    #[test]
    fn refuses_out_of_vocabulary_tokens_short_proposals_and_truncated_frames() {
        let (mut client, mut peer) = pair();
        let seq = SeqId(7);
        client.sequence = Some(seq);
        client.next = 8;
        let worker = std::thread::spawn(move || {
            for reply in [
                "{\"ok\":true,\"tokens\":[100]}\n",
                "{\"ok\":true,\"tokens\":[1]}\n",
                "{\"ok\":true",
            ] {
                header(&mut peer);
                peer.get_mut().write_all(reply.as_bytes()).unwrap();
            }
        });
        assert!(client.propose(seq, 8, 4, 1, 100, false).is_err());
        assert!(client.propose(seq, 8, 4, 15, 100, false).is_err());
        assert!(client.propose(seq, 8, 4, 15, 100, false).is_err());
        assert!(client.failed);
        assert!(client.propose(seq, 8, 4, 15, 100, false).is_err());
        worker.join().unwrap();
    }
    #[test]
    fn conditional_tree_matches_only_its_owner_and_accepts_a_non_chain_branch() {
        let (mut client, mut peer) = pair();
        let seq = SeqId(3);
        client.sequence = Some(seq);
        client.next = 91;
        let worker = std::thread::spawn(move || {
            for _ in 0..2 {
                assert_eq!(
                    header(&mut peer),
                    json!({"op":"propose","seq":3,"position":91,"anchor":76,"width":15,"tree_nodes":15})
                );
                let mut nodes = vec![
                    json!({"token":10,"parent":null}),
                    json!({"token":11,"parent":null}),
                ];
                // 11 -> 12 -> 13 ... is the second root child and deliberately off the mailbox chain.
                for i in 2..15 {
                    nodes.push(json!({"token":10+i, "parent":i-1}));
                }
                let mut frame =
                    serde_json::to_vec(&json!({"ok":true,"tokens":vec![9;15],"tree":nodes}))
                        .unwrap();
                frame.push(b'\n');
                peer.get_mut().write_all(&frame).unwrap();
            }
        });
        let chain = client.propose(seq, 91, 76, 15, 100, true).unwrap();
        assert!(client.take_tree(seq, 92, 76, &chain).is_none());
        let chain = client.propose(seq, 91, 76, 15, 100, true).unwrap();
        let plan = client.take_tree(seq, 91, 76, &chain).unwrap();
        let verdict = plan.verdict(|row| match row {
            0 => 11,
            2 => 12,
            _ => 99,
        });
        assert_eq!(verdict.path_rows, [0, 2, 3]);
        assert_eq!(verdict.committed, [11, 12, 99]);
        assert!(client.take_tree(seq, 91, 76, &chain).is_none());
        worker.join().unwrap();
    }

    #[test]
    fn validates_external_depth_15_masks_topology_and_vocabulary() {
        use crate::spec_tree::TreePlan;
        let mut entries: Vec<_> = (0usize..15)
            .map(|i| (i.checked_sub(1), 8 + i as u32))
            .collect();
        let plan = TreePlan::from_external(2, &entries, 100).unwrap();
        assert_eq!(plan.words()[15], u16::MAX);
        assert_eq!(plan.depth(15), 15);
        assert!(TreePlan::from_external(2, &entries[..14], 100).is_err());
        entries[14].0 = Some(14);
        assert!(TreePlan::from_external(2, &entries, 100).is_err());
        entries[14] = entries[13];
        assert!(TreePlan::from_external(2, &entries, 100).is_err());
        entries[14] = (None, 100);
        assert!(TreePlan::from_external(2, &entries, 100).is_err());
        assert!(serde_json::from_value::<WireNode>(json!({"token":4})).is_err());
    }
}
