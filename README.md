# dbh-guard-engine

DBH(Destroy the Beaver House) 검수용 **Qwen3.5 Guard** 실험 엔진.
Qwen3.5-4B 본체는 얼려 두고, 그 마지막 토큰 벡터를 읽는 **가드 헤드**만 학습해 위험도(0~1)를 낸다.
임베딩·본체 연산은 한 번만 하고, 학교 서버(RTX 4080 16GB × 2)에서 돌린다.

> [!IMPORTANT]
> 이 저장소는 **시험용**이다. DBH 웹 서버에 아직 연결하지 않는다. 연결은 적합도 실험(docs/EXPERIMENT.md)을
> 통과한 뒤 DBH 쪽 `shadow` 모드부터 한다. 서버의 차단(`blinded`)은 기본으로 꺼져 있다.

설계·실험 기준·초안에서 고친 점은 [docs/EXPERIMENT.md](docs/EXPERIMENT.md).

## 구조

```text
dbh-guard-engine/
├── src/
│   ├── model_hybrid.py   # Qwen3.5 본체(고정) + 가드 헤드, 입력 형식, 판정 정책
│   ├── teacher_label.py  # Qwen3Guard-Gen 으로 교사 라벨 (게시글·댓글 전용)
│   ├── train_head.py     # 헤드만 학습 (본체 출력은 데이터당 1번만 뽑는다)
│   ├── evaluate.py       # 적합도 지표 (오차단·놓침·AUROC·교사 일치율·지연)
│   └── server.py         # FastAPI 시험 서버 (127.0.0.1:8100)
├── scripts/check_env.py  # GPU·라이브러리·모델 config 점검
├── data/sample.jsonl     # 데이터 형식 예시 (실데이터는 커밋하지 않는다)
└── requirements.txt
```

## GPU 배치

| GPU | 쓰임 | 메모리 |
| --- | --- | --- |
| `cuda:0` | Qwen3.5-4B 본체 + 가드 헤드 (학습·평가·서버) | 약 9GB (bf16, 양자화 없음) |
| `cuda:1` | Qwen3Guard-Gen-4B 교사 (라벨 붙일 때만) | 약 8.5GB (bf16) |

## 실행 순서

### 0. 노트북 — GitHub 에 올리기

```bash
cd dbh-guard-engine
git remote add origin https://github.com/<계정>/dbh-guard-engine.git
git push -u origin main
```

### 1. 학교 서버 — 받기

```bash
git clone https://github.com/<계정>/dbh-guard-engine.git
cd dbh-guard-engine
python3 -m venv .venv && source .venv/bin/activate
```

### 2. 설치

```bash
nvidia-smi                       # 드라이버·CUDA 버전 확인
# torch 는 CUDA 버전에 맞는 휠을 먼저 (https://pytorch.org/get-started/locally/ 의 명령)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

모델 파일 위치를 바꾸려면 `export HF_HOME=/data/hf-cache` (4B 두 개 합쳐 약 17GB).

### 3. 환경 점검 (E0)

```bash
python scripts/check_env.py
```

GPU 2장, 두 모델의 `hidden_size`·층 수가 나오면 통과. `[X]` 가 나오면 `pip install -U transformers`.

### 4. 흐름 시험 — 모델 없이 (E1)

가짜 본체(`--mock`)로 학습 → 평가 → 서버가 끝까지 도는지 본다. GPU·다운로드 없이 노트북에서도 된다.

```bash
python src/train_head.py --mock --data data/sample.jsonl --out heads/mock.pt
python src/evaluate.py   --mock --data data/sample.jsonl --head heads/mock.pt
GUARD_MOCK=1 GUARD_HEAD=heads/mock.pt python src/server.py
```

Windows PowerShell 에서는 `$env:GUARD_MOCK="1"; $env:GUARD_HEAD="heads/mock.pt"; python src/server.py`.

### 5. 교사 라벨 (E2, 게시글·댓글만)

```bash
python src/teacher_label.py --data data/raw.jsonl --out data/teacher.jsonl --device cuda:1
```

### 6. 헤드 학습 (E3)

```bash
# 사람 라벨로
python src/train_head.py --data data/train.jsonl --bot post --out heads/guard-post-v0.pt
# 또는 Qwen3Guard 교사 라벨로 (증류)
python src/train_head.py --data data/teacher.jsonl --label-field teacher --bot post --out heads/guard-post-v0.pt
```

GPU 메모리가 모자라면 `--feat-batch 2`.

### 7. 평가 (E4)

학습에 쓰지 않은 시험셋으로 한다.

```bash
python src/evaluate.py --data data/test.jsonl --head heads/guard-post-v0.pt --out results/post-v0.jsonl
```

합격 기준은 [docs/EXPERIMENT.md §4](docs/EXPERIMENT.md#4-합격-기준).

### 8. 서버 (E5)

```bash
GUARD_HEAD=heads/guard-post-v0.pt python src/server.py
# 계속 띄워 두려면
nohup env GUARD_HEAD=heads/guard-post-v0.pt python src/server.py > guard.log 2>&1 &
```

```bash
curl -s http://127.0.0.1:8100/health
curl -s -X POST http://127.0.0.1:8100/api/v1/review \
  -H "Content-Type: application/json" \
  -d '{"bot":"post","content":"부업 하루 50만원 보장!! 오픈채팅 들어오세요","context":"글 제목: 첫 글"}'
```

응답 예 (값은 예시, 차단 스위치가 꺼져 있어 높은 위험도도 `manual_review`):

```json
{"status": "manual_review", "risk": 0.97, "reason_code": "high_risk_hold", "model": "Qwen/Qwen3.5-4B", "latency_ms": 85}
```

`content` 에는 원문만 넣는다. `<content>` 울타리는 서버가 씌운다.

## 환경변수 (server.py)

| 변수 | 기본값 | 뜻 |
| --- | --- | --- |
| `GUARD_HEAD` | (없음) | 학습한 헤드 경로. 없으면 모든 요청이 `manual_review` |
| `GUARD_BODY` | `Qwen/Qwen3.5-4B` | 본체 모델 |
| `GUARD_DEVICE` | `cuda:0` | |
| `GUARD_SAFE` / `GUARD_BLOCK` | `0.35` / `0.90` | 위험도 ≤ safe → `published`, ≥ block → 차단 후보 |
| `GUARD_ALLOW_BLOCK` | `0` | `1` 이어야 `blinded` 를 낸다. 아니면 차단 후보도 `manual_review` |
| `GUARD_MAX_TOKENS` | `4096` | 넘으면 자르지 않고 `manual_review`(`too_long`) |
| `GUARD_API_TOKEN` | (없음) | 지정하면 `Authorization: Bearer <값>` 필수. 틀리면 404 |
| `GUARD_HOST` / `GUARD_PORT` | `127.0.0.1` / `8100` | 바깥에 열지 않는다. 8000 은 vLLM 자리 |
| `GUARD_MOCK` | `0` | `1` 이면 가짜 본체 |

## 데이터 형식 (JSONL)

```json
{"bot": "post", "content": "본문", "context": "글 제목: …\n부모 댓글: …", "label": 0}
```

- `bot`: `post`(게시글·댓글) / `prompt`(프롬프트)
- `label`: 사람 라벨 0(정상)·1(위반). `teacher` 는 `teacher_label.py` 가 붙인다
- `context` 는 댓글일 때만
- 실데이터는 `data/` 에 두되 **커밋하지 않는다**(.gitignore). 개인정보는 합성 값만 쓴다
