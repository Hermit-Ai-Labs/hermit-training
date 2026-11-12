#!/usr/bin/env python3
"""Build the directory layout the generators, assembler, scorer and trainer expect.

The scripts were written inside a working directory ("the lab") where gen2/, gen3/, gen4/, spec/, eval/, train/, data*/
and eval*/ sit side by side, and each script finds its neighbours with paths like <script dir>/../spec. This repo groups
the same files by purpose (generators/, spec/, evals/, training/, data/). This tool copies them back into the lab layout
under .lab/ so every script runs unchanged:

    python tools/make_lab.py            # creates ./.lab
    cd .lab && python gen4/recovery_errors.py --n 20 --eval-n 5

.lab/ is git-ignored. It holds an uncompressed copy of the training file (about 80 MB). Pool outputs land in .lab/data4/,
.lab/data3/ and .lab/eval4/ (created empty).
"""
import argparse, gzip, os, shutil, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def copy(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(ROOT, ".lab"))
    a = ap.parse_args()
    L = a.dest
    os.makedirs(L, exist_ok=True)
    for g in ("gen", "gen2", "gen3", "gen4"):
        copy(os.path.join(ROOT, "generators", g), os.path.join(L, g))
    copy(os.path.join(ROOT, "generators", "hermit_tools.json"), os.path.join(L, "hermit_tools.json"))
    copy(os.path.join(ROOT, "spec"), os.path.join(L, "spec"))
    copy(os.path.join(ROOT, "evals", "harness"), os.path.join(L, "eval"))
    copy(os.path.join(ROOT, "training"), os.path.join(L, "train"))
    # eval5/: suites at the top level, results/ below
    copy(os.path.join(ROOT, "evals", "suites"), os.path.join(L, "eval5"))
    copy(os.path.join(ROOT, "evals", "results"), os.path.join(L, "eval5", "results"))
    # data5/: the shipped training file is gzipped; the scripts read plain jsonl
    d5 = os.path.join(L, "data5")
    os.makedirs(d5, exist_ok=True)
    with gzip.open(os.path.join(ROOT, "data", "sft5_train.jsonl.gz"), "rb") as fi, open(os.path.join(d5, "sft5_train.jsonl"), "wb") as fo:
        shutil.copyfileobj(fi, fo)
    for f in ("sft5_val.jsonl", "sft5_report.json", "STATS.md"):
        copy(os.path.join(ROOT, "data", f), os.path.join(d5, f))
    for d in ("data2", "data3", "data4", "data5/pools", "eval2", "eval3", "eval4"):
        os.makedirs(os.path.join(L, d), exist_ok=True)
    print("lab layout ready:", L)
    print("run scripts from inside it, for example:  cd %s && python gen4/recovery_errors.py --n 20 --eval-n 5" % os.path.relpath(L))


if __name__ == "__main__":
    main()
