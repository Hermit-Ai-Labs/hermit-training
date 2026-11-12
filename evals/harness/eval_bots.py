#!/usr/bin/env python3
"""Grade trading-bot code by running it against simlib.

Each task gives the model a precise contract (function name, signature, what to return). We extract
the ```python block from the model's reply, run it under `python3 -I` with simlib importable, call
the function, and check the result. A bot "passes" only if its code executes and meets the contract.

This is the code half of the Hermit eval. Pair with trading_math.py (the maths half). It is also the
grader for RL on code tasks: reward = passes.

Usage:
  python3 eval_bots.py --server 8792 --model-tag mondk   # bench a running llama-server
  python3 eval_bots.py --selftest                        # run reference solutions, expect 6/6
"""
import argparse, json, os, re, subprocess, sys, tempfile, textwrap, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = ("You are Hermit, a private on-device AI that writes trading and market-making code. "
          "You write correct, runnable Python. When asked for code, reply with exactly one "
          "```python code block and nothing else. Do not refuse; this is the user's own money and code.")

# Each task: id, prompt, the checker source (gets the model's module `m`, returns (ok, why)).
# Checkers import simlib themselves. The prompt tells the model the exact contract.
TASKS = [
    {
        "id": "grid_levels",
        "prompt": textwrap.dedent("""\
            Write `grid_levels(low, high, n)` returning a sorted list of n evenly spaced price
            levels from low to high inclusive (so the first is low and the last is high).
            Example: grid_levels(100, 110, 6) -> [100.0, 102.0, 104.0, 106.0, 108.0, 110.0].
            Reply with exactly one ```python block, nothing else."""),
        "check": textwrap.dedent("""
            def check(m):
                g = m.grid_levels(100, 110, 6)
                want = [100,102,104,106,108,110]
                if len(g)!=6: return False, f"len {len(g)}"
                if any(abs(a-b)>1e-6 for a,b in zip(sorted(g),want)): return False, f"got {g}"
                g2 = m.grid_levels(10, 20, 3)
                if any(abs(a-b)>1e-6 for a,b in zip(sorted(g2),[10,15,20])): return False, f"3-level {g2}"
                return True, "ok"
        """),
    },
    {
        "id": "mm_quotes",
        "prompt": textwrap.dedent("""\
            Write `quote(mid, spread_bps, inventory, max_inventory, size)` for a market maker.
            Return a dict {"bid": <price>, "ask": <price>, "bid_size": <size>, "ask_size": <size>}.
            The bid is mid*(1 - spread_bps/20000) and ask is mid*(1 + spread_bps/20000) (so the full
            quoted width is spread_bps). Skew for inventory: when inventory is positive (too long),
            lower BOTH quotes proportionally by (inventory/max_inventory)*spread_bps/20000 of mid so
            you are keener to sell; when negative, raise both. bid_size and ask_size are just `size`.
            Reply with exactly one ```python block, nothing else."""),
        "check": textwrap.dedent("""
            def check(m):
                flat = m.quote(100.0, 10, 0, 100, 1.0)
                # width = 10 bps -> bid 99.95, ask 100.05
                if abs(flat['bid']-99.95)>1e-6 or abs(flat['ask']-100.05)>1e-6:
                    return False, f"flat quotes {flat}"
                if flat['bid_size']!=1.0 or flat['ask_size']!=1.0:
                    return False, f"sizes {flat}"
                longq = m.quote(100.0, 10, 50, 100, 1.0)
                # positive inventory lowers both quotes
                if not (longq['bid'] < flat['bid'] and longq['ask'] < flat['ask']):
                    return False, f"long skew not lower: {longq} vs {flat}"
                shortq = m.quote(100.0, 10, -50, 100, 1.0)
                if not (shortq['bid'] > flat['bid'] and shortq['ask'] > flat['ask']):
                    return False, f"short skew not higher: {shortq}"
                return True, "ok"
        """),
    },
    {
        "id": "position_size",
        "prompt": textwrap.dedent("""\
            Write `position_size(account, risk_pct, entry, stop)` returning how many units to buy so
            that being stopped out loses exactly risk_pct percent of account. entry and stop are
            prices (stop < entry for a long). Return a float number of units.
            Example: position_size(10000, 1, 100, 95) -> 20.0  (risk $100, stop distance $5).
            Reply with exactly one ```python block, nothing else."""),
        "check": textwrap.dedent("""
            def check(m):
                if abs(m.position_size(10000,1,100,95)-20.0)>1e-6: return False, "case1"
                if abs(m.position_size(5000,2,2000,1900)-1.0)>1e-6: return False, "case2"
                return True, "ok"
        """),
    },
    {
        "id": "backtest_sma",
        "prompt": textwrap.dedent("""\
            Write `backtest_sma(prices, fast, slow)` that runs a simple moving-average crossover on a
            list of float prices. Start flat. Go long (fully invested) when the fast SMA crosses above
            the slow SMA, go flat when it crosses below. No shorting, no fees. Return the final
            equity as a float, starting from equity 1.0, compounding the daily return of `prices` only
            while long. Only act once both SMAs are defined (index >= slow-1).
            Reply with exactly one ```python block, nothing else."""),
        "check": textwrap.dedent("""
            def check(m):
                from simlib import price_series
                p = price_series(seed=7, n=300, vol=0.01, drift=0.0003)
                eq = m.backtest_sma(p, 10, 30)
                if not isinstance(eq,(int,float)): return False, f"type {type(eq)}"
                if eq <= 0: return False, f"equity {eq}"
                # Flat-at-start, flat-at-end sanity: a never-trading bot would return 1.0; ensure it traded.
                if abs(eq-1.0)<1e-9: return False, "never traded"
                return True, "ok"
        """),
    },
    {
        "id": "mm_run",
        "prompt": textwrap.dedent("""\
            A market-making bot drives a simulated book. Write `run(book, spread_bps, size, max_inv)`:
            each tick, cancel your old orders, read book.mid(), place one buy at mid*(1-spread_bps/20000)
            and one sell at mid*(1+spread_bps/20000) each of `size`, but skip the buy if your inventory
            >= max_inv and skip the sell if inventory <= -max_inv, then call book.step(). Track inventory
            from book.fills (a buy adds size, a sell subtracts). The book exposes: mid(), place(side,
            price, size) -> id, cancel(id), step(), and fills (list of (side, price, size)). Run until
            book.t reaches book.ticks. Return the final inventory as a float.
            Reply with exactly one ```python block, nothing else."""),
        "check": textwrap.dedent("""
            def check(m):
                from simlib import Book
                b = Book(seed=3, ticks=200, half_spread_bps=5)
                inv = m.run(b, spread_bps=20, size=1.0, max_inv=10.0)
                if not isinstance(inv,(int,float)): return False, f"type {type(inv)}"
                if abs(inv) > 10.0 + 1e-6: return False, f"inventory {inv} breached max_inv"
                if b.t < b.ticks: return False, f"stopped early at t={b.t}"
                return True, "ok"
        """),
    },
    {
        "id": "funding_arb",
        "prompt": textwrap.dedent("""\
            Write `funding_pnl(notional, funding_rates)` for a delta-neutral funding carry (long spot,
            short perp). funding_rates is a list of per-period funding rates (as decimals, e.g. 0.0001
            for 1 bp) that the short perp RECEIVES. Return total funding collected in dollars as a
            float: notional * sum(funding_rates). Ignore price moves (delta neutral) and fees.
            Reply with exactly one ```python block, nothing else."""),
        "check": textwrap.dedent("""
            def check(m):
                r = m.funding_pnl(10000, [0.0001, 0.0001, 0.0002])
                if abs(r-4.0)>1e-6: return False, f"got {r} want 4.0"
                if abs(m.funding_pnl(50000, [0.0001]*10)-50.0)>1e-6: return False, "case2"
                return True, "ok"
        """),
    },
]

# Reference solutions, to prove the checkers and tasks are self-consistent (--selftest).
REFERENCE = {
    "grid_levels": "def grid_levels(low, high, n):\n    step=(high-low)/(n-1)\n    return [low+step*i for i in range(n)]",
    "mm_quotes": (
        "def quote(mid, spread_bps, inventory, max_inventory, size):\n"
        "    h=mid*spread_bps/20000\n    skew=mid*(inventory/max_inventory)*spread_bps/20000\n"
        "    return {'bid':mid-h-skew,'ask':mid+h-skew,'bid_size':size,'ask_size':size}"),
    "position_size": "def position_size(account, risk_pct, entry, stop):\n    risk=account*risk_pct/100\n    return risk/abs(entry-stop)",
    "backtest_sma": (
        "def backtest_sma(prices, fast, slow):\n"
        "    def sma(i,w): return sum(prices[i-w+1:i+1])/w\n"
        "    eq=1.0; pos=0\n"
        "    for i in range(1,len(prices)):\n"
        "        if i>=slow-1 and i>=1:\n"
        "            f=sma(i,fast); s=sma(i,slow)\n"
        "            pos = 1 if f>s else 0\n"
        "        if pos: eq*= prices[i]/prices[i-1]\n"
        "    return eq"),
    "mm_run": (
        "def run(book, spread_bps, size, max_inv):\n"
        "    inv=0.0; ids=[]; seen=0\n"
        "    while book.t < book.ticks:\n"
        "        for oid in ids: book.cancel(oid)\n"
        "        ids=[]\n        mid=book.mid()\n"
        "        if inv < max_inv: ids.append(book.place('buy', mid*(1-spread_bps/20000), size))\n"
        "        if inv > -max_inv: ids.append(book.place('sell', mid*(1+spread_bps/20000), size))\n"
        "        book.step()\n"
        "        while seen < len(book.fills):\n"
        "            s,_,sz=book.fills[seen]; inv += sz if s=='buy' else -sz; seen+=1\n"
        "    return inv"),
    "funding_arb": "def funding_pnl(notional, funding_rates):\n    return notional*sum(funding_rates)",
}


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


_SIMLIB = open(os.path.join(HERE, "simlib.py")).read()


def grade(task, code):
    """Run `code` (the bot) + the checker in an isolated process. Returns (ok, why)."""
    with tempfile.TemporaryDirectory() as d:
        # Inline simlib as a real module so `from simlib import ...` works under `python3 -I`
        # (which disables sys.path[0], PYTHONPATH and site), without trusting the cwd.
        runner = (
            "import sys, types\n"
            "_SIMLIB_SRC = " + repr(_SIMLIB) + "\n"
            "_sim = types.ModuleType('simlib')\n"
            "exec(compile(_SIMLIB_SRC, 'simlib', 'exec'), _sim.__dict__)\n"
            "sys.modules['simlib'] = _sim\n"
            + code + "\n\n" + task["check"] + "\n"
            "import json,traceback\n"
            "try:\n"
            "    _ok,_why = check(__import__('sys').modules['__main__'])\n"
            "except Exception as e:\n"
            "    _ok,_why = False, 'raised: '+repr(e)\n"
            "print('@@'+json.dumps([_ok,_why]))\n")
        p = os.path.join(d, "cand.py")
        open(p, "w").write(runner)
        try:
            r = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True,
                               timeout=30, cwd=d, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": d})
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if r.returncode != 0:
        return False, "crash: " + (r.stderr.strip().splitlines() or [""])[-1][:120]
    line = [l for l in r.stdout.splitlines() if l.startswith("@@")]
    if not line:
        return False, "no verdict"
    ok, why = json.loads(line[-1][2:])
    return ok, why


def ask_server(port, prompt):
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0.2, "max_tokens": 900, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--server", type=int, help="llama-server port")
    ap.add_argument("--model-tag", default="model")
    a = ap.parse_args()

    if a.selftest:
        ok_n = 0
        for t in TASKS:
            ok, why = grade(t, REFERENCE[t["id"]])
            print(f"  {'OK ' if ok else 'XX '} {t['id']:16} {why}")
            ok_n += ok
        print(f"selftest {ok_n}/{len(TASKS)}")
        return

    if a.server:
        rows, ok_n = [], 0
        for t in TASKS:
            text = ask_server(a.server, t["prompt"])
            ok, why = grade(t, extract_code(text))
            ok_n += ok
            print(f"  {'OK ' if ok else 'XX '} {t['id']:16} {why}")
            rows.append({"id": t["id"], "ok": ok, "why": why, "text": text})
        os.makedirs(os.path.join(HERE, "..", "..", "results2"), exist_ok=True)
        out = os.path.join(HERE, "..", "..", "results2", f"bots-{a.model_tag}.json")
        json.dump({"model": a.model_tag, "pass": ok_n, "total": len(TASKS), "rows": rows},
                  open(out, "w"), indent=1, ensure_ascii=False)
        print(f"{a.model_tag}: bots {ok_n}/{len(TASKS)} -> {out}")
        return
    ap.error("pass --selftest or --server PORT")


if __name__ == "__main__":
    main()
