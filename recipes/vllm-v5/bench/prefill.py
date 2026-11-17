#!/usr/bin/env python3
"""Prefill benchmark: time to first token for exact-length unique prompts (max_tokens=1, streaming).

Prompts are random word sequences tokenized and truncated to exact lengths, sent as token IDs with a
unique cache salt, and verified to have zero cached prompt tokens. Prefill tok/s = prompt tokens / TTFT.
Also reports concurrent prefill (C parallel requests) as aggregate prompt tok/s.
"""
import argparse, concurrent.futures as cf, json, pathlib, random, re, statistics, time, urllib.request

MODEL = 'glm-5.3-flash'


def post(base, path, body, stream=False):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1800) if stream else json.loads(urllib.request.urlopen(req, timeout=600).read())


def requests_counter(base):
    text = urllib.request.urlopen(base + '/metrics', timeout=30).read().decode()
    return sum(float(m) for m in re.findall(r'^vllm:request_success_total\{[^}]*\} ([0-9.e+]+)$', text, re.M))


def make_ids(base, n, seed):
    rnd = random.Random(seed)
    words = ['alpha', 'river', 'compile', 'matrix', 'garden', 'socket', 'violet', 'kernel', 'harbor', 'lemon',
             'vector', 'thunder', 'pixel', 'canyon', 'orbit', 'token', 'marble', 'signal', 'forest', 'cursor']
    text = ' '.join(rnd.choice(words) + str(rnd.randrange(1000)) for _ in range(n))
    ids = post(base, '/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': False})['tokens']
    assert len(ids) >= n, (len(ids), n)
    return ids[:n]


def one(base, ids, salt, t0=None):
    """TTFT in seconds, measured from t0 (a common batch start) when given, else from this request's start."""
    t0 = time.perf_counter() if t0 is None else t0; ttft = None; usage = None
    body = {'model': MODEL, 'prompt': ids, 'max_tokens': 1, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}, 'cache_salt': salt}
    with post(base, '/v1/completions', body, stream=True) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            d = json.loads(line[6:])
            if ttft is None and d.get('choices'):
                ttft = time.perf_counter() - t0
            usage = d.get('usage') or usage
    cached = ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    assert usage['prompt_tokens'] == len(ids) and cached == 0, usage
    return ttft


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--lengths', type=int, nargs='+', default=[1024, 4096, 16384, 65536, 131072])
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--concurrent', type=int, nargs='*', default=[8], help='C parallel prompts of --concurrent-length')
    p.add_argument('--concurrent-length', type=int, default=8192)
    p.add_argument('--label', required=True)
    p.add_argument('--output', default='prefill-results')
    a = p.parse_args()
    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.label}-{time.time_ns()}'
    one(a.base_url, make_ids(a.base_url, 2048, 'warm'), tag + '-warm')
    before = requests_counter(a.base_url); sent = 0
    rows = []
    for n in a.lengths:
        ids = [make_ids(a.base_url, n, f'{n}-{i}') for i in range(a.reps)]
        ttfts = [one(a.base_url, x, f'{tag}-{n}-{i}') for i, x in enumerate(ids)]
        sent += len(ids)
        row = {'mode': 'single', 'prompt_tokens': n, 'ttft_s': ttfts, 'median_ttft_s': statistics.median(ttfts),
               'prefill_tps': n / statistics.median(ttfts)}
        rows.append(row); print(json.dumps(row), flush=True)
    for c in a.concurrent:
        n = a.concurrent_length; walls = []
        for rep in range(a.reps):
            ids = [make_ids(a.base_url, n, f'c{c}-{rep}-{i}') for i in range(c)]
            with cf.ThreadPoolExecutor(c) as ex:
                # All TTFTs are measured from one batch start, so thread start/scheduling delays count
                # toward the batch wall time (last first-token arrival - batch start).
                t0 = time.perf_counter()
                ttfts = list(ex.map(lambda x: one(a.base_url, x[1], f'{tag}-c{c}-{rep}-{x[0]}', t0), enumerate(ids)))
            walls.append(max(ttfts)); sent += c
        row = {'mode': f'C{c}', 'prompt_tokens': n, 'last_ttft_s': walls, 'aggregate_prefill_tps': c * n / statistics.median(walls)}
        rows.append(row); print(json.dumps(row), flush=True)
    delta = requests_counter(a.base_url) - before
    result = {'label': a.label, 'base_url': a.base_url, 'rows': rows, 'requests_sent': sent, 'server_request_delta': delta,
              'competing_requests': delta - sent}
    (out / f'{a.label}.json').write_text(json.dumps(result, indent=1))
    print('SUMMARY', json.dumps({'label': a.label, 'competing_requests': delta - sent,
                                 'prefill_tps': {r['mode'] + ':' + str(r['prompt_tokens']): round(r.get('prefill_tps') or r['aggregate_prefill_tps']) for r in rows}}))


if __name__ == '__main__':
    main()
