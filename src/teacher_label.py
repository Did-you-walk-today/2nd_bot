"""
Qwen3Guard-Gen 으로 교사 라벨을 붙인다 (docs/EXPERIMENT.md E2 · 적합도 실험 1: 교사 일치율).

  python src/teacher_label.py --data data/raw.jsonl --out data/teacher.jsonl --device cuda:1

각 행에 teacher(0/1/null), teacher_safety(Safe/Unsafe/Controversial), teacher_categories 를 붙인다.
Controversial 은 기본으로 null(학습에서 빠짐). --controversial 1 이면 위반으로 친다.

주의: 게시글·댓글(bot=post) 전용. 프롬프트(bot=prompt)는 Qwen3Guard 가 정상 역할 설정 문장을
공격으로 보는 경우가 많아 교사로 쓰지 않는다 — 사람 라벨(label)을 쓴다.
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_hybrid import load_jsonl

SAFETY = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.I)
CATEGORIES = re.compile(r"Categories:\s*(.+)", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3Guard-Gen-4B")
    ap.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    ap.add_argument("--controversial", type=int, choices=[0, 1], default=None)
    ap.add_argument("--include-prompt-bot", action="store_true", help="bot=prompt 행에도 붙인다(비교 실험용)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    rows = load_jsonl(args.data)
    counts = {"Safe": 0, "Unsafe": 0, "Controversial": 0, "unparsed": 0, "skipped": 0}
    with open(args.out, "w", encoding="utf-8") as f:
        for n, r in enumerate(rows, 1):
            if r["bot"] != "post" and not args.include_prompt_bot:
                counts["skipped"] += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            text = f"{r['context']}\n\n{r['content']}" if r.get("context") else r["content"]
            chat = tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            ids = tok(chat, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=64, do_sample=False)
            reply = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

            m = SAFETY.search(reply)
            safety = m.group(1).capitalize() if m else None
            if safety is None:
                counts["unparsed"] += 1
                teacher = None
            else:
                counts[safety] += 1
                teacher = {"Safe": 0, "Unsafe": 1}.get(safety, args.controversial)
            c = CATEGORIES.search(reply)
            r.update(
                teacher=teacher,
                teacher_safety=safety,
                teacher_categories=[x.strip() for x in c.group(1).split(",")] if c else [],
            )
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\r[teacher] {n}/{len(rows)} {counts}", end="", flush=True)
    print(f"\n[teacher] saved → {args.out}")
    if counts["unparsed"]:
        print("[teacher] 출력 형식을 못 읽은 행이 있다. 모델 카드의 출력 형식을 확인하고 SAFETY 정규식을 고친다")


if __name__ == "__main__":
    main()
