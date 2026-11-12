#!/usr/bin/env python3
"""Client for the local open teacher (Qwen3-30B-A3B-Instruct-2507, Apache-2.0) served by llama-server
on 127.0.0.1:8795. Parallel requests, on-disk cache keyed by the request, so re-runs are free."""
import concurrent.futures as cf, hashlib, json, os, time, urllib.request

URL = "http://127.0.0.1:8795/v1/chat/completions"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data2", "teacher_cache.jsonl")
_cache = None


def _load():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE):
            for l in open(CACHE):
                try:
                    o = json.loads(l); _cache[o["k"]] = o["v"]
                except Exception:
                    pass
    return _cache


def chat(messages, temperature=0.7, max_tokens=700, top_p=0.9, seed=0, retries=3):
    body = {"messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed}
    k = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
    c = _load()
    if k in c:
        return c[k]
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                o = json.load(r)
            ch = o["choices"][0]
            v = {"text": ch["message"].get("content") or "", "finish": ch.get("finish_reason")}
            c[k] = v
            with open(CACHE, "a") as f:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
            return v
        except Exception as e:
            time.sleep(3 * (i + 1))
    return {"text": "", "finish": "error"}


def map_chat(jobs, workers=8, log_every=100):
    """jobs: list of dicts with chat() kwargs. Returns results in order."""
    out = [None] * len(jobs)
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(chat, **j): i for i, j in enumerate(jobs)}
        for n, f in enumerate(cf.as_completed(futs), 1):
            out[futs[f]] = f.result()
            if n % log_every == 0:
                print(f"  teacher {n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    t = time.time()
    rs = map_chat([{"messages": [{"role": "user", "content": f"In 3 sentences, explain funding rates on perps. Variant {i}."}], "max_tokens": 200}
                   for i in range(16)])
    dt = time.time() - t
    print(rs[0]["text"])
    print(f"{dt:.1f}s for 16; approx tokens {sum(len(r['text']) for r in rs) / 4:.0f}")
