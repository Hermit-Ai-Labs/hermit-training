#!/usr/bin/env python3
"""Tool simulators for every eval5 suite (Hermit 1.3: tools v3, packs). Used by eval/eval_hermit4.py and gen4/eval_suites4.py.

    run = make_runner(item, family)       # item = one eval5 row (loaded with eval_suites4.load_suite, which sets item["_suite5"])
    text = run("prepare_send", {...})     # JSON string, exactly what the app would put in the tool message
    run.begin_turn(k, user_text)          # before script turn k (1-based): typed text, events (followups), stage (multistep)

Result order for a call (design of the v3 harness):
  1. outside the item's pack (item.pack for 4B, item.pack_lite for Lite) -> {"error": "unknown tool <name>"}, flagged in run.log
  2. RECORDED: item.reference is split into user-turn segments; each recorded (call -> tool message) is queued per segment
     under the canonical key name + json.dumps(args, sort_keys=True). A call equal to a recorded one gets that result,
     byte for byte (so the reference path reproduces the gold transcript exactly). The call also runs on the simulator so
     its state advances.
  3. SCRIPTED (recovery_errors only): item.script = [{name, arguments, results[]}]: results served in order per call, the
     last one repeated once the list is used up.
  4. SIM: the generator's own simulator, rebuilt from item.world (with the context's typed text and, where the world is the
     state at the start of the chat, the context's tool calls replayed).
  5. otherwise {"error": "This tool is not available right now."}
"Face ID" in a simulator result is replaced by the item's device word, like the app does.
gen2/, gen3/ and the gen4 generators are imported, never edited.
"""
import collections, copy, importlib, json, os, random, re, sys
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(LAB, "gen2"), os.path.join(LAB, "gen3"), HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

EVAL4_SIMS = ("spot_solana", "orders_protect", "contacts_pay", "auto_pages", "portfolio_caps", "multistep", "notes_name",
              "recovery_errors", "followups_stale", "rounding_brief", "ask_wording", "lite_cloud", "drift_fix", "safety_mixed",
              "cloud_examples_v3")
EVAL3_SIMS = ("multisend_typed", "multisend_file", "aave", "send_nochain", "route_rounding", "honesty", "trade_view",
              "stocks_price", "file_qa", "knowledge_fix", "buy_jupiter", "close_position", "activity_tool", "open_app_tool")
NO_TOOL_ERROR = json.dumps({"error": "This tool is not available right now."})

_MODS = {}


def mod(name):
    if name not in _MODS:
        _MODS[name] = importlib.import_module(name)
    return _MODS[name]


def ckey(name, args):
    try:
        return str(name) + " " + json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(name) + " " + repr(args)


def suite5(item):
    s = item.get("_suite5")
    if s:
        return s
    iid = str(item.get("id", ""))
    for pre, s in (("chain_multi-", "multistep"), ("followups-", "followups_stale"), ("tool_errors_v3-", "recovery_errors"),
                   ("ask_wording-", "ask_wording"), ("lite_cloud-", "lite_cloud"), ("cloud_examples_v3-", "cloud_examples_v3")):
        if iid.startswith(pre):
            return s
    return item.get("suite")


def bio_of(item):
    if item.get("bio"):
        return item["bio"]
    ctx = item.get("context") or []
    s = ctx[0]["content"] if ctx and ctx[0]["role"] == "system" else ""
    return "Touch ID" if ("Touch ID" in s and "Face ID" not in s) else "Face ID"


def pack_of(item, family):
    p = item.get("pack_lite") if family == "lite" else item.get("pack")
    if p is None and family == "lite":
        p = item.get("pack")
    return list(p or [])


def split_reference(reference):
    """-> (segments, users): segments[k] = [(call, result_text)] of user turn k (0 = the turn the context ends on)."""
    segs, users = [[]], []
    ref = reference or []
    i = 0
    while i < len(ref):
        m = ref[i]
        if m["role"] == "user":
            segs.append([])
            users.append(m["content"])
        elif m["role"] == "assistant" and m.get("tool_calls"):
            k = len(m["tool_calls"])
            outs = ref[i + 1:i + 1 + k]
            for tc, tm in zip(m["tool_calls"], outs):
                segs[-1].append(({"name": tc["name"], "arguments": tc["arguments"]}, tm.get("content") if tm.get("role") == "tool" else None))
            i += k
        i += 1
    return segs, users


def script_users(item):
    """Later user turns the harness feeds: the item.script strings that the reference answers (the scored ones).
    A script turn with no gold in the reference (4 orders_protect cancel_noKind items) is not fed: it would cost GPU time and
    could not be scored. recovery_errors' script holds scripted tool results instead of user turns."""
    sc = [s for s in (item.get("script") or []) if isinstance(s, str)]
    n_ref = sum(1 for m in item.get("reference") or [] if m["role"] == "user")
    return sc[:n_ref]


def ctx_users(item):
    return [m["content"] for m in item.get("context") or [] if m["role"] == "user"]


def ctx_calls(item):
    return [tc for m in item.get("context") or [] for tc in (m.get("tool_calls") or [])]


# ============================================================================================ simulator adapters
class Sim:
    """Base adapter: call(name, args) -> JSON text; user(text) when a new user turn starts; begin_turn(k, text)."""
    exact = True

    def __init__(self, item):
        self.item = item
        self.bio = bio_of(item)
        self.typed = list(ctx_users(item))

    def call(self, name, args):
        raise NotImplementedError

    def user(self, text):
        self.typed.append(text)

    def begin_turn(self, k, text):
        self.user(text)


class FnSim(Sim):
    """fn(name, args, sim) -> result object; dump(obj) -> text."""

    def __init__(self, item, fn, dump, replay_context=False):
        super().__init__(item)
        self.fn, self.dump = fn, dump
        if replay_context:
            for tc in ctx_calls(item):
                try:
                    self.fn(tc["name"], tc["arguments"], self)
                except Exception:
                    pass

    def call(self, name, args):
        return self.dump(self.fn(name, args, self)).replace("Face ID", self.bio)


def _jd(o):
    return json.dumps(o, ensure_ascii=False)


def _jds(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True)


def sim_spot_solana(item):
    m = mod("spot_solana")
    run = m.make_runner(item)

    class S(Sim):
        def call(self, name, args):
            return run(name, args).replace("Face ID", self.bio)

        def user(self, text):
            self.typed.append(text)
            run.add_user(text)
    return S(item)


def sim_orders_protect(item):
    m = mod("orders_protect")
    w = m.OPWorld.from_state(item["world"], 0)
    return FnSim(item, lambda n, a, s: w.call(n, a), _jd, replay_context=True)


def sim_contacts_pay(item):
    m = mod("contacts_pay")
    S3 = mod("send_nochain")
    w = m.World4.from_state(item["world"], 0) if "seed" in m.World4.from_state.__code__.co_varnames else m.World4.from_state(item["world"])

    def fn(n, a, s):
        typed = "\n".join(S3.without_files(u) for u in s.typed)
        return m.run_tool4(w, n, a, typed)
    return FnSim(item, fn, _jds)


def sim_auto_pages(item):
    m = mod("auto_pages")
    w = m.World4.from_state(item["world"], 0)
    us = ctx_users(item)
    w.last_user = us[-1] if us else ""

    class S(FnSim):
        def user(self, text):
            self.typed.append(text)
            w.last_user = text
    return S(item, lambda n, a, s: w.call(n, a), _jd)


def sim_portfolio_caps(item):
    m = mod("portfolio_caps")
    w = copy.deepcopy(item["world"])
    return FnSim(item, lambda n, a, s: m.run_tool(w, n, a), lambda o: m.dumps(o), replay_context=True)


def sim_multistep(item):
    m = mod("multistep")
    wd = item["world"]
    stages = wd["stages"]
    first = wd.get("first_turn", 0)

    class S(Sim):
        def __init__(self, it):
            super().__init__(it)
            self.resolved = set()
            self.stage = first
            self._mk()

        def _mk(self):
            st = stages[min(self.stage, len(stages) - 1)]
            self.w = m.World.from_state(st, typed=list(self.typed), resolved=set(self.resolved))

        def call(self, name, args):
            r = self.w.call(name, args)
            self.resolved |= set(getattr(self.w, "resolved", set()))
            return m.dumps_result(r, self.bio)

        def begin_turn(self, k, text):
            self.typed.append(text)
            self.stage = first + k
            self._mk()
    return S(item)


def sim_notes_name(item):
    m = mod("notes_name")
    w = m.NWorld.from_state(item["world"])

    class S(Sim):
        def __init__(self, it):
            super().__init__(it)
            self.known = set()
            for mm in it.get("context") or []:
                if mm["role"] == "tool":
                    try:
                        self.known |= m.contact_addrs(json.loads(mm["content"]))
                    except Exception:
                        pass

        def call(self, name, args):
            typed = "\n".join(self.typed)
            r = w.call_g(name, args, typed, self.known)
            try:
                self.known |= m.contact_addrs(r)
            except Exception:
                pass
            return _jd(r).replace("Face ID", self.bio)
    return S(item)


def sim_followups_stale(item):
    m = mod("followups_stale")
    rn = m.make_runner(item)

    class S(Sim):
        def call(self, name, args):
            return rn.run(name, args)

        def begin_turn(self, k, text):
            self.typed.append(text)
            rn.begin_turn(k)
    return S(item)


def sim_rounding_brief(item):
    m = mod("rounding_brief")
    w = m.RBWorld.from_state(item["world"])
    return FnSim(item, lambda n, a, s: w.call(n, a), m.dump)


def sim_ask_wording(item):
    m = mod("ask_wording")
    w = m.AskWorld.from_state(item["world"], 0)

    def fn(n, a, s):
        return m.run_tool(w, n, a, "\n".join(s.typed))
    return FnSim(item, fn, m.dumps_result, replay_context=True)


def sim_lite_cloud(item):
    m = mod("lite_cloud")
    w = m.W.from_state(item["world"])
    return FnSim(item, lambda n, a, s: w.call(n, copy.deepcopy(a)), m.jd)


def sim_drift_fix(item):
    m = mod("drift_fix")
    w = m.DriftWorld.from_state(item["world"])
    return FnSim(item, lambda n, a, s: w.call(n, a), _jd, replay_context=True)


def sim_eval3(item):
    s3 = mod("sims3")
    go = s3.make_runner(item)

    class S(Sim):
        def call(self, name, args):
            return go(name, args)
    return S(item)


SIM_FACTORY = {
    "spot_solana": sim_spot_solana, "orders_protect": sim_orders_protect, "contacts_pay": sim_contacts_pay,
    "auto_pages": sim_auto_pages, "portfolio_caps": sim_portfolio_caps, "multistep": sim_multistep,
    "notes_name": sim_notes_name, "followups_stale": sim_followups_stale, "rounding_brief": sim_rounding_brief,
    "ask_wording": sim_ask_wording, "lite_cloud": sim_lite_cloud, "drift_fix": sim_drift_fix,
    "cloud_examples_v3": sim_lite_cloud,
}


def make_sim(item):
    s = suite5(item)
    f = SIM_FACTORY.get(s)
    if f is None and (s in EVAL3_SIMS or s == "tools_fixed" or str(s).startswith("legacy_")):
        f = sim_eval3                               # gen2 sim.World (calculator, prices, ...) for the old suites and agent maths
    if f is None or (item.get("world") is None and f is not sim_eval3):
        return None
    try:
        return f(item)
    except Exception as e:                         # a broken world snapshot must not stop the eval: recorded results still work
        sys.stderr.write(f"[sims4] no simulator for {item.get('id')} ({s}): {type(e).__name__}: {e}\n")
        return None


# ============================================================================================ the runner
class Runner:
    def __init__(self, item, family="4b", use_sim=True):
        self.item, self.family = item, family
        self.bio = bio_of(item)
        self.suite = suite5(item)
        self.pack = set(pack_of(item, family))
        self.segs, self.ref_users = split_reference(item.get("reference"))
        self.queues = []
        for seg in self.segs:
            q = collections.defaultdict(collections.deque)
            for call, res in seg:
                if res is not None:
                    q[ckey(call["name"], call["arguments"])].append(res)
            self.queues.append(q)
        self.scripted = collections.defaultdict(list)
        self.scripted_i = collections.Counter()
        if self.suite == "recovery_errors":
            for s in item.get("script") or []:
                if isinstance(s, dict) and s.get("name"):
                    self.scripted[ckey(s["name"], s.get("arguments") or {})] += [_jd(r).replace("Face ID", self.bio) for r in s.get("results") or []]
        self.sim = make_sim(item) if use_sim else None
        self.seg = 0
        self.log = []                     # (segment, name, source, ok_in_pack)

    def begin_turn(self, k, text):
        self.seg = k
        if self.sim is not None:
            try:
                self.sim.begin_turn(k, text)
            except Exception as e:
                sys.stderr.write(f"[sims4] begin_turn failed for {self.item.get('id')}: {e}\n")

    def _sim_call(self, name, args):
        if self.sim is None:
            return None
        try:
            return self.sim.call(name, copy.deepcopy(args))
        except Exception as e:
            return json.dumps({"error": f"bad arguments: {type(e).__name__}"})

    def __call__(self, name, args):
        if not isinstance(args, dict):
            self.log.append((self.seg, name, "bad_args"))
            return json.dumps({"error": "bad arguments"})
        if self.pack and name not in self.pack:
            self.log.append((self.seg, name, "outside_pack"))
            return json.dumps({"error": f"unknown tool {name}"})
        key = ckey(name, args)
        q = self.queues[self.seg].get(key) if self.seg < len(self.queues) else None
        if q:
            res = q.popleft()
            self._sim_call(name, args)                  # keep the simulator's state in step with the recorded path
            self.log.append((self.seg, name, "recorded"))
            return res
        if key in self.scripted:
            lst = self.scripted[key]
            i = self.scripted_i[key]
            self.scripted_i[key] += 1
            self.log.append((self.seg, name, "scripted"))
            return lst[min(i, len(lst) - 1)]
        out = self._sim_call(name, args)
        if out is not None:
            self.log.append((self.seg, name, "sim"))
            return out
        if name == "calculator":                         # suites without a simulator (safety_mixed): the app's calculator
            self.log.append((self.seg, name, "calc"))
            return mod("hermit_fmt").calc_result(str(args.get("expression", "")))
        self.log.append((self.seg, name, "none"))
        return NO_TOOL_ERROR


def make_runner(item, family="4b", use_sim=True):
    return Runner(item, family, use_sim)


# ============================================================================================ fidelity (informational)
def sim_fidelity(item):
    """Replay the reference calls on a fresh simulator alone. -> (matched, total, first mismatch or None)."""
    sim = make_sim(item)
    segs, users = split_reference(item.get("reference"))
    tot = sum(1 for seg in segs for _, r in seg if r is not None)
    if sim is None:
        return 0, tot, "no simulator"
    ok, first_bad = 0, None
    for k, seg in enumerate(segs):
        if k:
            try:
                sim.begin_turn(k, users[k - 1])
            except Exception as e:
                return ok, tot, f"begin_turn: {e}"
        for call, res in seg:
            if res is None:
                continue
            try:
                got = sim.call(call["name"], copy.deepcopy(call["arguments"]))
            except Exception as e:
                got = f"EXC {type(e).__name__}: {e}"
            if got == res:
                ok += 1
            elif first_bad is None:
                first_bad = (call["name"], (got or "")[:160], res[:160])
    return ok, tot, first_bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="")
    ap.add_argument("--show", type=int, default=1)
    a = ap.parse_args()
    ev5 = os.path.join(LAB, "eval5")
    names = a.suites.split(",") if a.suites else [s["suite"] for s in json.load(open(os.path.join(ev5, "suites.json")))["suites"]]
    for n in names:
        rows = [json.loads(l) for l in open(os.path.join(ev5, n + ".jsonl"))]
        okc = totc = 0
        shown = 0
        for it in rows:
            it["_suite5"] = n
            o, t, bad = sim_fidelity(it)
            okc += o; totc += t
            if bad and shown < a.show and t:
                print(f"   {n} {it.get('id')}: {bad}")
                shown += 1
        print(f"{n:20} sim reproduces {okc}/{totc} recorded results")
