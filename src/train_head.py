"""
가드 헤드 정렬 학습 (docs/EXPERIMENT.md E3).

본체는 얼려 두고, 본체 출력(마지막 실제 토큰 벡터)을 데이터당 **한 번만** 뽑아 둔 뒤 헤드만 학습한다.
본체 통과가 비용의 전부라 몇 분이면 끝나고, 헤드 학습 자체는 수 초다.

  python src/train_head.py --data data/train.jsonl --out heads/guard-post-v0.pt --bot post
  python src/train_head.py --data data/teacher.jsonl --label-field teacher --out heads/guard-post-v0.pt --bot post
"""
from __future__ import annotations

import argparse
import json
import random

import torch
from torch import nn

from evaluate import summarize
from model_hybrid import DEFAULT_BODY, DBHHybridGuardModel, Policy, format_input, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-field", default="label", help="label(사람) 또는 teacher(Qwen3Guard)")
    ap.add_argument("--bot", choices=["post", "prompt"], help="이 봇의 행만 쓴다")
    ap.add_argument("--body", default=DEFAULT_BODY)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--feat-batch", type=int, default=8, help="본체 통과 배치. OOM 이면 줄인다")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [r for r in load_jsonl(args.data) if r.get(args.label_field) in (0, 1)]
    if args.bot:
        rows = [r for r in rows if r["bot"] == args.bot]

    model = DBHHybridGuardModel(
        body_model_id=args.body,
        device=args.device,
        dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
        mock_mode=args.mock,
    )
    print(f"[train] {model.describe()}")

    texts, labels, too_long = [], [], 0
    for r in rows:
        t = format_input(r["bot"], r["content"], r.get("context"))
        if model.token_len(t) > args.max_tokens:
            too_long += 1
            continue
        texts.append(t)
        labels.append(float(r[args.label_field]))
    if len(texts) < 2 or len(set(labels)) < 2:
        raise SystemExit("정상(0)·위반(1) 이 모두 있는 데이터가 2건 이상 필요하다")
    print(f"[train] rows={len(texts)} (위반 {int(sum(labels))}) too_long_skipped={too_long}")

    # 1) 본체 통과 — 길이순으로 묶어 패딩을 줄인다
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    feats = torch.empty(len(texts), model.hidden_size)
    for k in range(0, len(order), args.feat_batch):
        idx = order[k : k + args.feat_batch]
        feats[idx] = model.features([texts[i] for i in idx]).cpu()
        print(f"\r[train] features {min(k + args.feat_batch, len(order))}/{len(order)}", end="", flush=True)
    print()
    y = torch.tensor(labels)

    # 2) 정상·위반 비율을 유지하며 검증셋을 나눈다
    val_idx: list[int] = []
    for cls in (0.0, 1.0):
        ids = [i for i in range(len(y)) if y[i] == cls]
        random.shuffle(ids)
        val_idx += ids[: max(1, round(len(ids) * args.val_ratio))] if len(ids) > 1 else []
    val_set = set(val_idx)
    tr_idx = [i for i in range(len(y)) if i not in val_set]
    xtr, ytr = feats[tr_idx].to(args.device), y[tr_idx].to(args.device)
    xva, yva = feats[val_idx].to(args.device), y[val_idx].to(args.device)

    # 3) 헤드만 학습 (본체는 얼어 있다)
    head = model.head.train()
    pos = ytr.sum().clamp(min=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(ytr) - pos) / pos)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(xtr), ytr)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            vloss = nn.functional.binary_cross_entropy_with_logits(head(xva), yva).item() if len(val_idx) else loss.item()
        if vloss < best - 1e-4:
            best, best_state, wait = vloss, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    print(f"[train] stopped at epoch {epoch + 1}, best val loss {best:.4f}")

    # 4) 검증셋 지표 (참고용 — 합격 판정은 따로 둔 시험셋으로 evaluate.py)
    with torch.no_grad():
        vr = torch.sigmoid(head(xva)).tolist() if len(val_idx) else []
    metrics = summarize([int(v) for v in yva.tolist()], vr, Policy()) if vr else {}
    print(json.dumps({"val": metrics}, ensure_ascii=False, indent=2))

    model.head_trained = True
    model.save_head(
        args.out,
        extra={
            "label_field": args.label_field,
            "bot": args.bot or "all",
            "n_train": len(tr_idx),
            "n_val": len(val_idx),
            "val_loss": round(best, 4),
        },
    )
    print(f"[train] saved head → {args.out}")


if __name__ == "__main__":
    main()
