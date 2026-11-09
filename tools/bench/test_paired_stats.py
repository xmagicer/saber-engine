"""Local end-to-end and window-boundary checks for paired_http, without a GPU."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import paired_http as bench


class Handler(BaseHTTPRequestHandler):
    reject_ids=False
    slow=False
    received=[]
    def do_POST(self):
        payload=self.rfile.read(int(self.headers["Content-Length"]))
        body=json.loads(payload);self.received.append((body,hashlib.sha256(payload).hexdigest()))
        self.send_response(200)
        if not body.get("stream"):
            self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(b'{"choices":[]}');return
        self.send_header("Content-Type","text/event-stream");self.end_headers()
        def frame(o):
            self.wfile.write(b"data: "+json.dumps(o).encode()+b"\n\n");self.wfile.flush()
        frame({"choices":[{"index":0,"delta":{"role":"assistant"}}]})
        for i,(word,ids) in enumerate((("abcd",[11,12,13,14]),("ef",[15,16]),("gh",[17,18]))):
            if i:time.sleep(.57 if self.slow else .025)
            ch={"index":0,"delta":{"content":word},"finish_reason":"length" if i==2 else None}
            if body.get("return_token_ids") and not self.reject_ids:ch["token_ids"]=ids
            frame({"choices":[ch]})
        frame({"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":8}})
        self.wfile.write(b"data: [DONE]\n\n");self.wfile.flush()
    def log_message(self,*args):pass


def due(t,n):
    return [{"delta":{"content":"x"},"token_ids":list(range(n))}]


def windows():
    t=bench.TokenArrivals()
    for i,(at,n) in enumerate([(100,3),(600_000_000,2),(1_000_000_100,5)]):t.record(i,at,due(at,n))
    r=t.report({"completion_tokens":10},True)
    assert r["peak_1s_tok_s"]==7 and r["peak_window_ns"]==[100,1_000_000_100],r
    assert not t.report({"completion_tokens":9},True)["available"]
    t=bench.TokenArrivals()
    t.record(0,100,due(100,7));t.record(1,200,due(200,2))
    assert not t.report({"completion_tokens":9},True)["available"]
    t=bench.TokenArrivals()
    t.record(0,10,[{"delta":{"content":"ab"}}])
    t.record(1,500_000_010,due(0,2));t.record(2,1_600_000_010,due(0,1))
    assert not t.report({"completion_tokens":3},True)["available"]


def main():
    windows()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);files=root/"payload";files.mkdir()
        manifest=[]
        for i in range(3):
            name="short-code-c1-r%d-i0"%i;f=files/(name+".json")
            bench.write_json(f,bench.body_for(name,"fixture",8))
            manifest.append({"name":name,"file":f.name,"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"suite":"short","kind":"code","concurrency":1,"repeat":i})
        bench.write_json(files/"manifest.json",manifest)
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            args=argparse.Namespace(base_url=f"http://127.0.0.1:{server.server_port}",key=None,timeout=5,stream_timing=False,
                                    payload_dir=files,output_dir=root/"plain",label="fixture",suite="short")
            assert bench.run(args)==0
            summary=json.loads((args.output_dir/"summary.json").read_text())
            rates=[json.loads((args.output_dir/(e["name"]+".result.json")).read_text())["decode_tps_approx"] for e in manifest]
            group=summary["request_rate_groups"][0]
            assert group["mean_tok_s"]==statistics.mean(rates) and group["median_tok_s"]==statistics.median(rates) and group["max_request_tok_s"]==max(rates)
            assert all("return_token_ids" not in body for body,_ in Handler.received)
            res=subprocess.run([sys.executable,str(Path(bench.__file__)),"--payload-dir",str(files),"--summarize-existing",str(args.output_dir)],capture_output=True,text=True,check=True)
            assert json.loads(res.stdout)["request_rate_groups"]==summary["request_rate_groups"]
            args.output_dir=root/"timed";args.output_dir.mkdir();args.stream_timing=True;Handler.slow=True
            d=bench.stream_one(args,manifest[0],threading.Barrier(1));body,wire=Handler.received[-1]
            trace=d["token_arrivals"]
            assert d["passed"] and trace["available"] and trace["peak_1s_tok_s"]==6 and trace["observed_tokens"]==8
            assert [ev["tokens"] for ev in trace["events"]]==[4,2,2]
            assert d["text_chunks"]==3 and d["wire_sha256"]==wire and d["payload_sha256"]==manifest[0]["sha256"]
            original=json.loads((files/manifest[0]["file"]).read_text());assert body=={**original,"return_token_ids":True}
            low,high=trace["peak_window_ns"];assert high-low==10**9
            assert sum(ev["tokens"] for ev in trace["events"] if low<ev["elapsed_ns"]<=high)==6
            Handler.slow=False;Handler.reject_ids=True
            e=bench.stream_one(args,manifest[1],threading.Barrier(1));assert e["passed"] and not e["token_arrivals"]["available"] and e["token_arrivals"]["peak_1s_tok_s"] is None
            print("PASS: unchanged requests, original/on-wire hashes, grouping and historical replay; genuine 8 IDs in 3 SSE events, 1.14s stream and peak 6; unsupported/short/mismatched ID streams have no peak.")
        finally:server.shutdown();server.server_close();thread.join()


if __name__=="__main__":main()
