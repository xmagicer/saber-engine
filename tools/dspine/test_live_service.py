import contextlib
import json
import math
import socket
import threading
import unittest

import live_service


class FixtureDrafter:
    vocab = 256

    def __init__(self):
        self.histories = {}

    def append(self, seq, position, rows, payload):
        history = self.histories.setdefault(seq, [])
        assert len(history) == position
        history.extend(payload[i * live_service.ROW_BYTES] for i in range(rows))

    def propose(self, seq, position, anchor, width):
        history = self.histories[seq]
        assert position == len(history)
        return [(anchor + token) % self.vocab for token in history[-1:] * width]

    def clear(self, seq):
        self.histories.pop(seq, None)


@contextlib.contextmanager
def session(backend, max_seqs=2):
    server, client = socket.socketpair()
    client.settimeout(5)
    def run():
        with server:
            live_service.serve_connection(server, backend, max_seqs)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        with client, client.makefile("rb") as reader:
            yield client, lambda: json.loads(reader.readline())
    finally:
        worker.join(timeout=5)
        if worker.is_alive():
            raise RuntimeError("Session did not release after disconnect")


def header(connection, op, seq, **params):
    line = json.dumps({"op": op, "seq": seq, **params}).encode()
    connection.sendall(line + b"\n")


def append(connection, receive, seq, position, markers):
    header(connection, "append", seq, position=position, rows=len(markers))
    for marker in markers:
        row = bytes([marker]) * live_service.ROW_BYTES
        connection.sendall(row[:21])
        connection.sendall(row[21:])
    return receive()


class LiveProtocolTests(unittest.TestCase):
    def test_interleaved_sequences_fragmented_frames_and_id_reuse(self):
        backend = FixtureDrafter()
        with session(backend) as (client, receive):
            self.assertTrue(append(client, receive, 1, 0, [4, 7])["ok"])
            self.assertTrue(append(client, receive, 2, 0, [20])["ok"])
            for seq, pos, expected in [(1, 2, 10), (2, 1, 23), (1, 2, 10)]:
                header(client, "propose", seq, position=pos, anchor=3, width=2)
                self.assertEqual(receive(), {"ok": True, "tokens": [expected, expected]})
            header(client, "clear", 1)
            self.assertTrue(receive()["ok"])
            self.assertTrue(append(client, receive, 1, 0, [81])["ok"])
            header(client, "propose", 1, position=1, anchor=2, width=1)
            self.assertEqual(receive()["tokens"], [83])
        self.assertEqual(backend.histories, {})

    def test_tree_request_returns_chain_and_zero_based_branching_parents(self):
        candidates = [[2*depth, 2*depth+1] for depth in range(15)]
        transition = [math.log(.51), math.log(.49)]
        logp = [[transition, transition] for _ in candidates]

        class TreeDrafter(FixtureDrafter):
            tree_calls = 0

            def propose_tree(self, seq, position, anchor, width):
                self.tree_calls += 1
                return [row[0] for row in candidates], live_service.wire_tree(candidates, logp)

        backend = TreeDrafter()
        with session(backend) as (client, receive):
            self.assertTrue(append(client, receive, 1, 0, [4])["ok"])
            header(client, "propose", 1, position=1, anchor=2, width=15, tree_nodes=15)
            reply = receive()
            self.assertTrue(reply["ok"])
            self.assertEqual(reply["tokens"], [row[0] for row in candidates])
            tree = reply["tree"]
            self.assertEqual(len(tree), 15)
            self.assertEqual(tree[:3], [{"token": 0, "parent": None},
                                         {"token": 1, "parent": None},
                                         {"token": 2, "parent": 0}])
            self.assertTrue(any(node["parent"] == 1 for node in tree))
            for index,node in enumerate(tree):
                self.assertTrue(node["parent"] is None or 0 <= node["parent"] < index)
            header(client, "propose", 1, position=1, anchor=3, width=3)
            self.assertEqual(receive(), {"ok": True, "tokens": [7]*3})
        self.assertEqual(backend.tree_calls, 1)

    def test_unsupported_or_malformed_tree_request_fails_before_proposal(self):
        for width, nodes in [(15, True), (14, 15), (15, 9), (15, None), (15, 15)]:
            backend = FixtureDrafter()
            with self.subTest(width=width,nodes=nodes), session(backend) as (client,receive):
                self.assertTrue(append(client,receive,1,0,[7])["ok"])
                header(client,"propose",1,position=1,anchor=2,width=width,tree_nodes=nodes)
                self.assertFalse(receive()["ok"])
            self.assertEqual(backend.histories, {})

    def test_tree_rejects_ambiguous_siblings_or_forward_parent(self):
        base = [{"token": i,"parent": None} for i in range(15)]
        for replacement in [{"token":0,"parent":None}, {"token":7,"parent":14},
                            {"token":7,"parent":True}, {"token":999,"parent":0}]:
            tree = base[:-1] + [replacement]
            with self.subTest(node=replacement), self.assertRaises(ValueError):
                live_service.validate_tree(tree,256)

    def test_gap_is_rejected_and_all_session_caches_are_cleared(self):
        backend = FixtureDrafter()
        with session(backend) as (client, receive):
            self.assertTrue(append(client, receive, 1, 0, [8])["ok"])
            header(client, "append", 1, position=3, rows=1)
            self.assertFalse(receive()["ok"])
        self.assertEqual(backend.histories, {})

    def test_truncated_append_does_not_reach_backend(self):
        backend = FixtureDrafter()
        with session(backend) as (client, receive):
            header(client, "append", 9, position=0, rows=2)
            client.sendall(bytes(live_service.ROW_BYTES + 3))
            client.shutdown(socket.SHUT_WR)
            self.assertFalse(receive()["ok"])
        self.assertEqual(backend.histories, {})

    def test_partial_backend_failure_frees_new_sequence(self):
        class FailingDrafter(FixtureDrafter):
            def append(self, seq, position, rows, payload):
                self.histories[seq] = [17]
                raise RuntimeError("Failed to finish context")
        backend = FailingDrafter()
        with session(backend) as (client, receive):
            self.assertFalse(append(client, receive, 5, 0, [17])["ok"])
        self.assertEqual(backend.histories, {})

    def test_large_frames_and_excess_sequences_are_rejected_before_payload(self):
        backend = FixtureDrafter()
        with session(backend) as (client, receive):
            header(client, "append", 1, position=0, rows=live_service.MAX_FRAME_ROWS + 1)
            self.assertFalse(receive()["ok"])
        with session(backend, 1) as (client, receive):
            self.assertTrue(append(client, receive, 1, 0, [8])["ok"])
            header(client, "append", 2, position=0, rows=1)
            self.assertFalse(receive()["ok"])
        self.assertEqual(backend.histories, {})


if __name__ == "__main__":
    unittest.main()
