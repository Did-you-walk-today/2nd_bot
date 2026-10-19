"""
DBH 가드 엔진 시험 서버 (FastAPI).

- 127.0.0.1 에만 연다. 학교 서버에 들어오는 포트를 열지 않는다는 DBH 원칙 그대로
- 포트 기본값 8100. 8000 은 DBH 기획상 vLLM 자리다
- GUARD_API_TOKEN 을 지정하면 Bearer 토큰을 요구한다(틀리면 404)
- 헤드 없음·너무 긴 입력·추론 실패는 전부 manual_review (fail-to-review)
- 차단(blinded)은 GUARD_ALLOW_BLOCK=1 일 때만. 적합도 실험 통과 전에는 켜지 않는다
- 로그와 응답에 본문·예외 원문을 남기지 않는다. 사유는 코드만

실행:  python src/server.py
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from model_hybrid import DEFAULT_BODY, DBHHybridGuardModel, Policy, apply_policy, format_input

ENV = os.environ
HOST = ENV.get("GUARD_HOST", "127.0.0.1")
PORT = int(ENV.get("GUARD_PORT", "8100"))
MAX_TOKENS = int(ENV.get("GUARD_MAX_TOKENS", "4096"))
API_TOKEN = ENV.get("GUARD_API_TOKEN", "")
POLICY = Policy(
    safe_thresh=float(ENV.get("GUARD_SAFE", "0.35")),
    block_thresh=float(ENV.get("GUARD_BLOCK", "0.90")),
    allow_block=ENV.get("GUARD_ALLOW_BLOCK") == "1",
)

state: dict = {}
# GPU 하나를 여러 요청이 동시에 쓰지 않게 한다
lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    mock = ENV.get("GUARD_MOCK") == "1"
    device = ENV.get("GUARD_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DBHHybridGuardModel(
        body_model_id=ENV.get("GUARD_BODY", DEFAULT_BODY),
        device=device,
        dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        mock_mode=mock,
        head_path=ENV.get("GUARD_HEAD") or None,
    )
    model.eval()
    state["model"] = model
    print(f"[guard] ready {model.describe()} policy={POLICY}", flush=True)
    if not model.head_trained:
        print("[guard] 헤드가 없다(GUARD_HEAD). 모든 요청이 manual_review 로 나간다", flush=True)
    yield
    state.clear()


app = FastAPI(title="DBH Guard Engine", version="0.1.0", lifespan=lifespan)


class ReviewRequest(BaseModel):
    bot: Literal["post", "prompt"]
    # 원문 그대로 보낸다. <content> 울타리는 서버가 씌운다
    content: str = Field(min_length=1, max_length=50_000)
    # 댓글이면 글 제목·부모 댓글 (DBH task.ts 와 같은 역할)
    context: str | None = Field(default=None, max_length=10_000)


class ReviewResponse(BaseModel):
    status: Literal["published", "manual_review", "blinded"]
    risk: float | None
    reason_code: str
    model: str
    latency_ms: int


def _check_token(authorization: str | None) -> None:
    if not API_TOKEN:
        return
    got = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(got.encode(), API_TOKEN.encode()):
        raise HTTPException(status_code=404)


@app.get("/health")
def health(authorization: str | None = Header(default=None)):
    _check_token(authorization)
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="loading")
    return {"ok": True, **model.describe(), "allow_block": POLICY.allow_block}


@app.post("/api/v1/review", response_model=ReviewResponse)
def review(req: ReviewRequest, authorization: str | None = Header(default=None)):
    _check_token(authorization)
    model: DBHHybridGuardModel | None = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="loading")

    started = time.perf_counter()

    def done(status: str, code: str, risk: float | None = None) -> ReviewResponse:
        ms = int((time.perf_counter() - started) * 1000)
        print(f"[guard] bot={req.bot} status={status} code={code} risk={risk} ms={ms}", flush=True)
        return ReviewResponse(status=status, risk=risk, reason_code=code, model=model.body_model_id, latency_ms=ms)

    try:
        text = format_input(req.bot, req.content, req.context)
        # 자르지 않는다 — 자르면 뒷부분이 무심사가 된다
        if model.token_len(text) > MAX_TOKENS:
            return done("manual_review", "too_long")
        with lock:
            risk = model.risk_scores([text])[0]
        status, code = apply_policy(risk, POLICY, model.head_trained)
        return done(status, code, round(risk, 4))
    except Exception as e:  # noqa: BLE001 — 어떤 실패든 사람 검토로
        print(f"[guard] inference_error {type(e).__name__}", flush=True)
        return done("manual_review", "inference_error")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
