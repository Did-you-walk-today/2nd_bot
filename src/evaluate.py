"""
적합도 평가 — 학습한 헤드를 시험 데이터로 채점한다 (docs/EXPERIMENT.md E4).

  python src/evaluate.py --data data/test.jsonl --head heads/guard-v0.pt
  python src/evaluate.py --data data/test.jsonl --head heads/guard-v0.pt --label-field teacher

지표: 오차단 수(정상인데 risk ≥ block), 놓침 수(위반인데 risk ≤ safe), 자동 처리율, AUROC, ECE,
교사(Qwen3Guard) 일치율, 지연(배치 1, p50/p95). 오차단 0건을 지키는 최소 block 임계값도 알려준다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from model_hybrid import DEFAULT_BODY, DBHHybridGuardModel, Policy, format_input, load_jsonl


def auroc(labels: list[int], risks: list[float]) -> float | None:
    pos = [r for r, y in zip(risks, labels) if y == 1]
    neg = [r for r, y in zip(risks, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ece(labels: list[int], risks: list[float], bins: int = 10) -> float:
    """확신도 보정 오차. 0 에 가까울수록 '위험도 0.9 = 실제로 90% 위반' 에 가깝다"""
    total, n = 0.0, len(labels)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, r in enumerate(risks) if lo <= r < hi or (b == bins - 1 and r == 1.0)]
        if idx:
            conf = sum(risks[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(conf - acc)
    return total


def summarize(labels: list[int], risks: list[float], policy: Policy) -> dict:
    neg = [r for r, y in zip(risks, labels) if y == 0]
    pos = [r for r, y in zip(risks, labels) if y == 1]
    s, b = policy.safe_thresh, policy.block_thresh
    out = {
        "n": len(labels),
        "n_normal": len(neg),
        "n_violation": len(pos),
        "false_block": sum(r >= b for r in neg),  # 목표 0
        "missed": sum(r <= s for r in pos),  # 위반인데 자동 통과 — 최소화
        "auto_publish_rate": round(sum(r <= s for r in neg) / len(neg), 4) if neg else None,
        "auto_block_recall": round(sum(r >= b for r in pos) / len(pos), 4) if pos else None,
        "manual_review_rate": round(sum(s < r < b for r in risks) / len(risks), 4) if risks else None,
        "auroc": None,
        "ece": round(ece(labels, risks), 4) if risks else None,
        # 이 데이터에서 오차단 0건을 지키려면 block 은 이 값보다 커야 한다
        "min_block_for_zero_false_block": round(max(neg), 4) if neg else None,
        # 이 데이터에서 놓침 0건을 지키려면 safe 는 이 값보다 작아야 한다
        "max_safe_for_zero_missed": round(min(pos), 4) if pos else None,
    }
    a = auroc(labels, risks)
    out["auroc"] = round(a, 4) if a is not None else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--label-field", default="label", help="정답으로 쓸 필드: label(사람) 또는 teacher")
    ap.add_argument("--body", default=DEFAULT_BODY)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--safe", type=float, default=0.35)
    ap.add_argument("--block", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--latency-samples", type=int, default=30)
    ap.add_argument("--out", help="행별 결과(jsonl: 행 번호·정답·위험도만, 본문 없음)")
    args = ap.parse_args()

    model = DBHHybridGuardModel(
        body_model_id=args.body,
        device=args.device,
        dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
        mock_mode=args.mock,
        head_path=args.head,
    )
    model.eval()

    rows = [(i, r) for i, r in enumerate(load_jsonl(args.data)) if r.get(args.label_field) in (0, 1)]
    items, too_long = [], 0
    for i, r in rows:
        text = format_input(r["bot"], r["content"], r.get("context"))
        if model.token_len(text) > args.max_tokens:
            too_long += 1  # 운영에서는 manual_review 로 간다
            continue
        items.append((i, r, text))
    if not items:
        raise SystemExit(f"평가할 행이 없다 (field={args.label_field})")

    risks: list[float] = []
    for k in range(0, len(items), args.batch):
        risks += model.risk_scores([t for _, _, t in items[k : k + args.batch]])
    labels = [int(r[args.label_field]) for _, r, _ in items]

    result = {"label_field": args.label_field, "too_long_skipped": too_long, **summarize(labels, risks, Policy(args.safe, args.block))}

    # 교사 일치율: 사람 라벨로 평가할 때 teacher 필드가 있으면 같이 본다 (적합도 실험 1)
    pairs = [(rk >= 0.5, r["teacher"] == 1) for (_, r, _), rk in zip(items, risks) if r.get("teacher") in (0, 1)]
    if pairs:
        result["teacher_agreement"] = round(sum(a == b for a, b in pairs) / len(pairs), 4)
        result["teacher_n"] = len(pairs)

    # 지연: 운영과 같은 배치 1
    lat = []
    for _, _, text in items[: args.latency_samples]:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.risk_scores([text])
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    result["latency_ms_p50"] = round(statistics.median(lat), 1)
    result["latency_ms_p95"] = round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for (i, r, _), rk in zip(items, risks):
                f.write(json.dumps({"row": i, "bot": r["bot"], "label": r[args.label_field], "risk": round(rk, 4)}) + "\n")


if __name__ == "__main__":
    main()
