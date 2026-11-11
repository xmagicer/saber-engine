#!/usr/bin/env python3
"""Paired, fixed-output real HTTP benchmark; SSE chunks are not tokens."""
import argparse
import bisect
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import threading
import time
import urllib.error
import urllib.request

MODEL = 'qwen3.8-27b-mxfp4'
PROMPTS = {
    'code': 'Write a complete Python implementation of an asyncio job scheduler with bounded '
            'concurrency, cancellation, retries with exponential backoff, typed result objects, '
            'and example usage. Include substantial code and explain its design.',
    'prose': 'Write a detailed technical guide on diagnosing intermittent latency in a distributed '
             'inference service. Discuss queues, scheduling, prefill, decode, network transport, '
             'GPU memory, tracing, and controlled experiments. Use diverse concrete examples.'}


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def read_key(path):
    if not path:
        return None
    for line in Path(path).read_text().splitlines():
        parts = shlex.split(line, comments=True)
        for part in parts:
            if part.startswith('VLLM_API_KEY='):
                return part.split('=', 1)[1]
    raise RuntimeError('VLLM_API_KEY missing from configured key file')


def wire_bytes(body):
    return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()


def request(args, path, body):
    headers = {'Content-Type': 'application/json'}
    if args.key:
        headers['Authorization'] = 'Bearer '+args.key
    req = urllib.request.Request(args.base_url.rstrip('/')+path,
        data=wire_bytes(body), headers=headers)
    return urllib.request.urlopen(req, timeout=args.timeout)


def body_for(name, text, output):
    nonce = hashlib.sha256(('saber-comparison-20261108/'+name).encode()).hexdigest()[:24]
    return {'model': MODEL, 'messages': [
        {'role': 'system', 'content': f'Independent benchmark nonce {nonce}. Follow the user request.'},
        {'role': 'user', 'content': text}], 'temperature': 0, 'max_tokens': output,
        'ignore_eos': True, 'chat_template_kwargs': {'enable_thinking': False},
        'stream': True, 'stream_options': {'include_usage': True}}


def corpus():
    pieces = []
    for i in range(6000):
        digest = hashlib.sha256(str(i).encode()).hexdigest()[:12]
        pieces.append(f'''\nTechnical observation {i}, sample identifier {digest}.
An inference worker processes queued requests in bounded batches. Example batch {i % 17 + 1}
uses {i % 4096 + 512} rows, while a tracing span records admission, prefill and token generation.
Memory pressure depends on sequence lengths, shared prefix reuse and the capacity of attention
state. A controlled experiment must hold the input, output budget, hardware power cap and
sampling parameters constant. Report individual request latency separately from aggregate
throughput. This observation differs from record {max(0,i-1)} by its queue and memory sample.
def sample_{i}(queue_length, allocated_pages):
    reserve = {i % 113 + 16}
    return max(0, queue_length - {i % 7}) + allocated_pages + reserve
''')
    return ''.join(pieces)


def prepare(args):
    args.payload_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    workloads = [('short', 'code', 1), ('short', 'prose', 1)]
    workloads += [('short', 'mixed', c) for c in args.concurrency if c > 1]
    for suite, kind, concurrency in workloads:
        for repeat in range(args.repeats):
            for index in range(concurrency):
                name = f'{suite}-{kind}-c{concurrency}-r{repeat}-i{index}'
                selected = kind if kind != 'mixed' else ('code' if index % 2 == 0 else 'prose')
                payload = body_for(name, PROMPTS[selected], args.max_tokens)
                p = args.payload_dir/(name+'.json')
                write_json(p, payload)
                manifest.append({'name': name, 'suite': suite, 'kind': kind, 'concurrency': concurrency,
                    'repeat': repeat, 'index': index, 'file': p.name,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    document = corpus()
    for target in args.long_targets:
        for repeat in range(args.repeats):
            name = f'long-document-c1-r{repeat}-t{target}'
            chars = target*4
            count = None
            for attempt in range(8):
                text = ('Review these independent technical observations as reference data.\n'+
                    document[:chars]+'\nEnd of reference. '+PROMPTS['prose'])
                payload = body_for(name, text, args.long_max_tokens)
                token_body = {k: payload[k] for k in ['model', 'messages', 'chat_template_kwargs']}
                token_body['add_generation_prompt'] = True
                with request(args, '/tokenize', token_body) as response:
                    count = json.load(response)['count']
                if abs(count-target) <= 32:
                    break
                chars = max(1, int(chars*(target-80)/max(1,count-80)))
            if count is None or abs(count-target) > 128:
                raise RuntimeError(f'Could not calibrate long prompt {target}: {count}')
            p = args.payload_dir/(name+'.json')
            write_json(p, payload)
            manifest.append({'name': name, 'suite': 'long', 'kind': 'document', 'concurrency': 1,
                'repeat': repeat, 'index': 0, 'target_prompt_tokens': target,
                'original_tokenizer_prompt_tokens': count, 'file': p.name,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    write_json(args.payload_dir/'manifest.json', manifest)
    print('PREPARED', len(manifest), 'paired payloads', flush=True)


WINDOW_NS = 1_000_000_000


class TokenArrivals:
    """Client delivery times of generated IDs; retain frame numbers for auditing the raw SSE."""
    def __init__(self):
        self.events = []
        self.problem = None

    def record(self, event_index, elapsed_ns, choices):
        count = 0
        for choice in choices:
            ids = choice.get('token_ids')
            if ids is None:
                # A role envelope and the usage-only event carry no generated IDs.
                delta = choice.get('delta') or {}
                if any(value for key, value in delta.items() if key != 'role'):
                    self.problem = 'A generated delta did not provide per-step token_ids'
                continue
            if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                self.problem = 'Invalid per-step token_ids'
                continue
            count += len(ids)
        if count:
            self.events.append({'event_index':event_index, 'elapsed_ns':elapsed_ns, 'tokens':count})

    def report(self, usage, completed):
        times = [e['elapsed_ns'] for e in self.events]
        totals = [0]
        for event in self.events:
            totals.append(totals[-1]+event['tokens'])
        result = {'window_ns':WINDOW_NS, 'events':self.events, 'observed_tokens':totals[-1],
                  'peak_1s_tok_s':None, 'peak_window_ns':None, 'available':False}
        problem = self.problem
        if not completed:
            problem = 'Stream or fixed-length validation did not complete'
        elif type(usage.get('completion_tokens')) is not int or not times or totals[-1] != usage['completion_tokens']:
            problem = 'Per-step token_ids absent or their sum differs from completion_tokens'
        elif times[-1]-times[0] < WINDOW_NS:
            problem = 'First-to-last generated ID delivery spans less than one second'
        if problem:
            return {**result,'reason':problem}
        # Full windows (end-1s, end] since request start. The first possible full
        # window, and then generated-ID arrivals, are the only times the maximum can increase.
        best = -1
        for end in [WINDOW_NS, *(t for t in times if t >= WINDOW_NS)]:
            left = bisect.bisect_right(times,end-WINDOW_NS)
            right = bisect.bisect_right(times,end)
            count = totals[right]-totals[left]
            if count > best:
                best = count
                result['peak_window_ns'] = [end-WINDOW_NS,end]
        result.update(available=True, peak_1s_tok_s=best)
        return result


def grouped_request_rates(manifest, results):
    groups = {}
    for entry in manifest:
        key = (entry['suite'],entry['kind'],entry['concurrency'],entry.get('target_prompt_tokens'))
        groups.setdefault(key,[]).append(entry)
    report = []
    for (suite,kind,concurrency,target), entries in groups.items():
        rates = {}
        for entry in entries:
            row = results.get(entry['name'])
            if not row:
                continue
            if row.get('payload_sha256') != entry['sha256']:
                raise ValueError('Result payload fingerprint differs for '+entry['name'])
            rate = row.get('decode_tps_approx')
            if (row.get('passed') and row.get('fixed_length_valid') and
                    type(rate) in (int,float) and math.isfinite(rate) and rate>0):
                rates[entry['name']] = rate
        complete = len(rates) == len(entries)
        values = list(rates.values())
        peak = max(values) if complete else None
        report.append({'suite':suite,'kind':kind,'concurrency':concurrency,'target_prompt_tokens':target,
                       'expected_requests':len(entries),'valid_requests':len(values),'complete':complete,
                       'mean_tok_s':statistics.mean(values) if complete else None,
                       'median_tok_s':statistics.median(values) if complete else None,
                       'max_request_tok_s':peak,'max_requests':[name for name,v in rates.items() if v==peak],
                       'requests':rates})
    return report


def stream_one(args, entry, barrier):
    payload_path = args.payload_dir/entry['file']
    payload = json.loads(payload_path.read_text())
    if hashlib.sha256(payload_path.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError('Paired payload changed after preparation')
    timing = getattr(args,'stream_timing',False)
    if timing:
        payload['return_token_ids'] = True
    wire_sha = hashlib.sha256(wire_bytes(payload)).hexdigest()
    arrivals = TokenArrivals() if timing else None
    barrier.wait()
    start_ns = time.perf_counter_ns() if timing else 0
    start = time.monotonic()
    first = last = None
    usage, finish, chunks, done = {}, None, 0, False
    content, reasoning = [], []
    result = {'name': entry['name'], 'payload_sha256': entry['sha256'], 'wire_sha256':wire_sha, 'stream_timing':timing, 'start': start}
    raw_path = args.output_dir/(entry['name']+'.sse')
    try:
        with request(args, '/v1/chat/completions', payload) as response, raw_path.open('wb') as raw:
            event_lines = []
            event_index = 0
            for line in response:
                # Stamp completion of each data frame before JSON parsing or writing its delimiter.
                if timing and line in (b'\n',b'\r\n'):
                    frame_ns = time.perf_counter_ns()-start_ns
                raw.write(line)
                stripped = line.rstrip(b'\r\n')
                if stripped:
                    if stripped.startswith(b'data:'):
                        event_lines.append(stripped[5:].lstrip())
                    continue
                if not event_lines:
                    continue
                data = b'\n'.join(event_lines)
                event_lines = []
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if arrivals is not None:
                    arrivals.record(event_index,frame_ns,event.get('choices',[]))
                    event_index += 1
                if event.get('error'):
                    raise RuntimeError(json.dumps(event['error'], ensure_ascii=False))
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    text = delta.get('content') or ''
                    thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                    if text or thought:
                        last = time.monotonic()
                        if first is None:
                            first = last
                        chunks += 1
                        content.append(text)
                        reasoning.append(thought)
                    finish = choice.get('finish_reason') or finish
        n = usage.get('completion_tokens')
        result['passed'] = bool(done and first is not None and n == payload['max_tokens'])
    except Exception as exc:
        result['passed'] = False
        if isinstance(exc, urllib.error.HTTPError):
            result['error'] = f'HTTP {exc.code}: '+exc.read(2000).decode(errors='replace')
        else:
            result['error'] = f'{type(exc).__name__}: {exc}'[:2000]
        if args.key:
            result['error'] = result['error'].replace(args.key, '[REDACTED]')
    end = time.monotonic()
    result.update(end=end, elapsed_s=end-start, ttft_s=first-start if first else None,
        usage=usage, finish_reason=finish, done=done, text_chunks=chunks,
        content=''.join(content), reasoning=''.join(reasoning),
        fixed_length_valid=usage.get('completion_tokens') == payload['max_tokens'],
        decode_tps_approx=(usage.get('completion_tokens',0)/(last-first)
            if first is not None and last is not None and last > first else None),
        prompt_tokens_per_ttft_s=(usage.get('prompt_tokens',0)/(first-start) if first is not None and first>start else None))
    if arrivals is not None:
        result['token_arrivals'] = arrivals.report(usage,result['passed'])
    write_json(args.output_dir/(entry['name']+'.result.json'), result)
    return result


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.payload_dir/'manifest.json').read_text())
    selected = [x for x in manifest if args.suite == 'all' or x['suite'] == args.suite]
    warm = body_for('warmup', PROMPTS['prose'], 128)
    warm['stream'] = False
    warm.pop('stream_options')
    with request(args, '/v1/chat/completions', warm) as response:
        write_json(args.output_dir/'warmup.json', json.load(response))
    groups = {}
    for entry in selected:
        key = (entry['suite'],entry['kind'],entry['concurrency'],entry.get('target_prompt_tokens'),entry['repeat'])
        groups.setdefault(key, []).append(entry)
    summary = {'label': args.label, 'base_url': args.base_url, 'batches': [],
        'metric_notes': {'decode_tps_approx': 'completion_tokens/(last text chunk-first text chunk); '
            'first chunk tokens are not subtracted; chunks are not counted as tokens',
            'prompt_tokens_per_ttft_s': 'actual prompt tokens/TTFT, includes first decode and scheduling'},
        'started_utc': time.strftime('%FT%TZ',time.gmtime())}
    summary['stream_timing_requested'] = getattr(args,'stream_timing',False)
    summary['metric_notes']['max_request_tok_s'] = 'Highest whole-request decode_tps_approx in a complete workload group'
    summary['metric_notes']['peak_1s_tok_s'] = 'Client-arrival generated IDs in a full rolling second (start,end]; requires per-step token_ids and matching final usage'
    per_request = {}
    for key, entries in groups.items():
        barrier = threading.Barrier(len(entries))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(entries)) as pool:
            results = list(pool.map(lambda entry: stream_one(args, entry, barrier), entries))
        per_request.update((r['name'],r) for r in results)
        span = max(r['end'] for r in results)-min(r['start'] for r in results)
        ttfts = [r['ttft_s'] for r in results if r['ttft_s'] is not None]
        rates = [r['decode_tps_approx'] for r in results if r['decode_tps_approx'] is not None]
        batch = {'suite': key[0], 'kind': key[1], 'concurrency': key[2], 'target_prompt_tokens': key[3],
            'repeat': key[4], 'passed': all(r['passed'] for r in results), 'elapsed_s': span,
            'aggregate_e2e_tps': sum(r['usage'].get('completion_tokens',0) for r in results)/span,
            'mean_decode_tps_approx': statistics.mean(rates) if rates else None,
            'ttft_p50_s': percentile(ttfts,.5), 'ttft_p95_s': percentile(ttfts,.95),
            'actual_prompt_tokens': [r['usage'].get('prompt_tokens') for r in results],
            'actual_completion_tokens': [r['usage'].get('completion_tokens') for r in results],
            'requests': [r['name'] for r in results],
            'errors': [r.get('error','incomplete response') for r in results if not r['passed']]}
        summary['batches'].append(batch)
        write_json(args.output_dir/'summary.json', summary)
        print(json.dumps(batch,ensure_ascii=False),flush=True)
    summary['passed'] = bool(summary['batches']) and all(x['passed'] for x in summary['batches'])
    summary['request_rate_groups'] = grouped_request_rates(selected,per_request)
    if summary['stream_timing_requested']:
        summary['request_arrival_peaks_1s'] = {
            name:{k:v for k,v in r['token_arrivals'].items() if k!='events'} for name,r in per_request.items()
        }
    summary['completed_utc'] = time.strftime('%FT%TZ',time.gmtime())
    write_json(args.output_dir/'summary.json', summary)
    return 0 if summary['passed'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url',default='http://127.0.0.1:18080')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--payload-dir',type=Path,required=True)
    parser.add_argument('--label',default='engine')
    parser.add_argument('--api-key-file')
    parser.add_argument('--suite',choices=['short','long','all'],default='all')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--concurrency',default='1,2,4,8')
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--long-max-tokens',type=int,default=512)
    parser.add_argument('--long-targets',default='16384,65536,131072')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--stream-timing',action='store_true',
                        help='Ask for per-step token_ids in benchmark requests and timestamp their SSE delivery; save original payload and effective wire hashes')
    parser.add_argument('--summarize-existing',type=Path,metavar='RESULT_DIR',
                        help='Print request-level mean, median and maximum from existing validated result files without HTTP calls')
    args = parser.parse_args()
    if args.summarize_existing:
        if args.prepare or args.stream_timing:
            parser.error('--summarize-existing reads historical results only')
        manifest = json.loads((args.payload_dir/'manifest.json').read_text())
        entries = [e for e in manifest if args.suite=='all' or e['suite']==args.suite]
        results = {}
        for e in entries:
            if hashlib.sha256((args.payload_dir/e['file']).read_bytes()).hexdigest()!=e['sha256']:
                raise ValueError('Paired payload changed: '+e['name'])
            path = args.summarize_existing/(e['name']+'.result.json')
            if path.exists():
                results[e['name']] = json.loads(path.read_text())
        report = grouped_request_rates(entries,results)
        print(json.dumps({'request_rate_groups':report},ensure_ascii=False,indent=2))
        return 0 if report and all(g['complete'] for g in report) else 1
    args.key = read_key(args.api_key_file)
    args.concurrency = [int(x) for x in args.concurrency.split(',')]
    args.long_targets = [int(x) for x in args.long_targets.split(',')]
    if args.prepare:
        prepare(args)
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required for benchmarking')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
