#!/usr/bin/env python3
"""Merge a Hermit 1.2 LoRA adapter into the base, attach the shipped chat template + generation config,
and lay out exactly the files the app downloads (ModelCatalog.wanted: root-level .safetensors/.json/.jinja).

  python merge_export2.py --base <base> --adapter runs/<run>/final --family 4b --out export/Hermit-1.2-4B-merged

Then convert to MLX 4-bit (Linux CPU venv or the Mac):
  ../../mlxenv/bin/python -m mlx_lm convert --hf-path <merged> -q --q-bits 4 --q-group-size 64 --mlx-path <out>-mlx
"""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--family", choices=["4b", "lite"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.base)
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    os.makedirs(a.out, exist_ok=True)
    merged.save_pretrained(a.out, safe_serialization=True)
    tmpl = os.path.join(HERE, "chat_template_lite.jinja" if a.family == "lite" else "chat_template_4b.jinja")
    template = open(tmpl).read()
    tok.chat_template = template
    tok.save_pretrained(a.out)
    shutil.copy(tmpl, os.path.join(a.out, "chat_template.jinja"))
    # generation config = the app's sampling (also capped in LocalEngine, but set sane defaults here)
    gen = {"bos_token_id": tok.bos_token_id, "eos_token_id": tok.convert_tokens_to_ids("<|im_end|>"),
           "pad_token_id": tok.convert_tokens_to_ids("<|endoftext|>"), "do_sample": True,
           "temperature": 0.6, "top_p": 0.9, "top_k": 20, "repetition_penalty": 1.05}
    json.dump(gen, open(os.path.join(a.out, "generation_config.json"), "w"), indent=1)
    # the app stops on <|im_end|>; keep tokenizer eos consistent
    print("merged ->", a.out)
    print("files:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
