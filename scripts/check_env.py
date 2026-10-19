"""
학교 서버 환경 점검 (docs/EXPERIMENT.md E0). 가중치는 받지 않고 config 만 읽는다.

  python scripts/check_env.py
"""
import sys

import torch
import transformers
from transformers import AutoConfig

MODELS = sys.argv[1:] or ["Qwen/Qwen3.5-4B", "Qwen/Qwen3Guard-Gen-4B"]

print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__} (cuda {torch.version.cuda})")
print(f"transformers {transformers.__version__}")
print(f"cuda 사용 가능 {torch.cuda.is_available()}, GPU {torch.cuda.device_count()}장")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i} {p.name} {total / 2**30:.1f}GB (여유 {free / 2**30:.1f}GB)")

ok = True
for mid in MODELS:
    try:
        cfg = AutoConfig.from_pretrained(mid)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"\n[X] {mid}: config 를 못 읽었다 — {type(e).__name__}: {e}")
        print("    transformers 가 이 모델을 아직 모르면 최신으로 올린다: pip install -U transformers")
        continue
    text = getattr(cfg, "text_config", None) or cfg
    print(f"\n[O] {mid}")
    print(f"    model_type        {cfg.model_type}")
    print(f"    hidden_size       {getattr(text, 'hidden_size', None)}")
    print(f"    num_hidden_layers {getattr(text, 'num_hidden_layers', None)}")
    print(f"    vocab_size        {getattr(text, 'vocab_size', None)}")
    print(f"    vision            {'있음' if getattr(cfg, 'vision_config', None) else '없음'}")

sys.exit(0 if ok else 1)
