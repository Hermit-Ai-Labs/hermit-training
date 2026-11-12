#!/usr/bin/env python3
"""One entry point for the 14 eval3 simulators.

Each gen3 generator ships its own tool simulator (World class or function) and writes eval items whose `world`
field is that simulator's state snapshot. The eval harness has to execute the model's tool calls against the SAME
simulator, with the same app guards (chain dropped when the user never named it, addresses the user never typed
refused, ...). This module wires them together:

    run = make_runner(item)          # item = one eval3 row (or any row with suite/world/context/bio)
    text = run("prepare_send", {...})   # JSON string, exactly what the app would put in the tool message

Rules shared by every adapter:
  * the result is json.dumps(result, ensure_ascii=False [, sort_keys]) exactly like the generator wrote it,
    then "Face ID" is replaced by the example's device word (meta bio), like the app does;
  * the typed user text that guards need is everything the user typed in the context (files excluded where the
    generator excludes them);
  * unknown tools and bad argument shapes return {"error": ...}, never raise.
Generator modules are imported lazily, one per suite. Importing a generator patches gen2's hermit_fmt in memory to the
19-tool registry (gen2 files are untouched), which is what eval_hermit3 wants anyway.
"""
import importlib, json, os, random, sys
sys.dont_write_bytecode = True      # never write .pyc next to the read-only gen2 modules we import

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, os.path.join(LAB, "gen2")):
    if p not in sys.path:
        sys.path.insert(0, p)

SUITES = ["multisend_typed", "multisend_file", "aave", "send_nochain", "route_rounding", "honesty", "trade_view",
          "stocks_price", "file_qa", "knowledge_fix", "buy_jupiter", "close_position", "activity_tool", "open_app_tool"]

_MODS = {}


def _mod(name):
    if name not in _MODS:
        _MODS[name] = importlib.import_module(name)
    return _MODS[name]


def suite_of(item):
    """eval3 rows carry id '<suite>-...'; meta.source or generator also name it."""
    for k in ("suite",):
        if item.get(k):
            return item[k]
    iid = str(item.get("id", ""))
    for s in SUITES:
        if iid.startswith(s + "-") or iid.startswith(s + "_"):
            return s
    m = item.get("meta") or {}
    return m.get("source") or m.get("generator") or item.get("generator")


def user_texts(context):
    return [m["content"] for m in context if m["role"] == "user"]


def bio_of(item):
    if item.get("bio"):
        return item["bio"]
    sysm = item["context"][0]["content"] if item.get("context") and item["context"][0]["role"] == "system" else ""
    return "Touch ID" if "Touch ID" in sysm else "Face ID"


def _dumps(r, sort=False):
    return json.dumps(r, ensure_ascii=False, sort_keys=sort)


def _safe(fn):
    """Call a simulator; shape errors become tool errors."""
    def run(name, args):
        try:
            return fn(name, args if isinstance(args, dict) else {})
        except Exception as e:                      # a model can send anything
            return {"error": f"bad arguments: {type(e).__name__}"}
    return run


def make_runner(item):
    """Return run(name, args) -> JSON string for this item's simulator and world."""
    suite = suite_of(item)
    bio = bio_of(item)
    ctx = item.get("context") or []
    users = user_texts(ctx)
    world = item.get("world")
    sort = False
    if suite == "aave":
        w = _mod("aave").World3.from_state(world)
        fn = w.call
    elif suite == "route_rounding":
        w = _mod("route_rounding").RRWorld.from_state(world)
        fn, sort = w.call, True
    elif suite == "send_nochain":
        m = _mod("send_nochain")
        w = m.World3.from_state(world, seed=0)
        typed = "\n".join(m.without_files(u) for u in users)
        fn = lambda n, a: m.run_tool(w, n, a, typed)
    elif suite == "multisend_typed":
        w = _mod("multisend_typed").MWorld.from_state(world)
        w.user_text = "\n".join(users)
        fn = w.call
    elif suite == "multisend_file":
        w = _mod("multisend_file").MSWorld.from_state(world)
        fn, sort = w.call, True
    elif suite == "close_position":
        w = _mod("close_position").CloseWorld.from_state(world, seed=0)
        fn = w.call
    elif suite == "buy_jupiter":
        w = _mod("buy_jupiter").BuyWorld.from_state(world)
        fn = w.call
    elif suite == "trade_view":
        w = _mod("trade_view").TVWorld.from_state(world)
        fn = w.call
    elif suite == "stocks_price":
        w = _mod("stocks_price").StockWorld.from_state(world)
        fn = w.call
    elif suite == "honesty":
        w = _mod("honesty").HWorld.from_state(world)
        fn = w.call
    elif suite == "open_app_tool":
        w = _mod("open_app_tool").World3.from_state(world)
        fn = w.call
    elif suite == "activity_tool":
        m = _mod("activity_tool")
        wd = json.loads(json.dumps(world))
        fn = lambda n, a: m.call_tool(wd, n, a) if n in m.TOOL_FN else {"error": f"unknown tool {n}"}
    else:                                           # file_qa, knowledge_fix and anything old: gen2 world
        import sim
        w = sim.World.from_state(world) if world else sim.World(random.Random(0))
        fn = w.call
    run = _safe(fn)

    def go(name, args):
        return _dumps(run(name, args), sort).replace("Face ID", bio)
    return go
