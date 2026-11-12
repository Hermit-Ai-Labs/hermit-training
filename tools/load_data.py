#!/usr/bin/env python3
"""Read the Hermit 1.3 training and validation files and print counts.

    python tools/load_data.py

Expected output: train 21361, val 414. The training file is gzipped (jsonl.gz); any tool that reads gzip works, for example
`zcat data/sft5_train.jsonl.gz | head -n 1`.

Each row is {"messages": [...], "tools": bool, "meta": {...}}. Roles are system, user, assistant and tool. An assistant
message may carry "tool_calls": [{"name": ..., "arguments": {...}}]. meta.source says which generator or legacy pool the row
came from, meta.pack is the list of tool names the 4B prompt carries, meta.pack_lite the same for Lite, and meta.family_ok
says which model trains on the row ("4b" only, or "4b" and "lite").
"""
import collections, gzip, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load():
    train = list(read_jsonl(os.path.join(ROOT, "data", "sft5_train.jsonl.gz")))
    val = list(read_jsonl(os.path.join(ROOT, "data", "sft5_val.jsonl")))
    return train, val


def main():
    train, val = load()
    print(f"train {len(train)}")
    print(f"val {len(val)}")
    lite = sum(1 for r in train if "lite" in r["meta"].get("family_ok", []))
    calls = sum(1 for r in train for m in r["messages"] for _ in (m.get("tool_calls") or []))
    print(f"train rows usable by Lite: {lite}")
    print(f"tool calls in train: {calls}")
    print("rows per source (train):")
    for k, v in collections.Counter(r["meta"]["source"] for r in train).most_common():
        print(f"  {k:20s} {v}")
    return 0 if (len(train), len(val)) == (21361, 414) else 1


if __name__ == "__main__":
    sys.exit(main())
