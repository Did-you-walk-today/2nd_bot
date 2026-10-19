# Qwen3.5 Guard — 설계와 적합도 실험

작성 2026-10-19 · 관련: DBH 저장소 `docs/AI심사-연동-기획서.md`, `docs/프롬프트-검수-SLM-파인튜닝-기획안.md`,
`docs/클린봇-Qwen서버-기획서.md`

## 1. 목적

Qwen3Guard 의 안전 판정 능력을 **Qwen3.5-4B 본체 위에** 올린다. 한 번의 본체 연산으로 얻은 벡터를
가드 헤드가 읽어 위험도를 낸다. 첫 목표는 **게시글·댓글(bot=post)** 에서 이 구조가 Qwen3Guard 만큼
판정하는지 확인하는 것이다.

## 2. 구조

```text
format_input(bot, content, context)          ← 학습·평가·서버가 같은 문자열
        │
Qwen3.5-4B 본체 (얼림, bf16, cuda:0)         ← 데이터당 1번만 통과
        │  마지막 실제 토큰 벡터 [hidden_size]
가드 헤드 LayerNorm → Linear(hidden, 1)       ← 이것만 학습 (float32, 수 KB)
        │  sigmoid
위험도 0~1 → apply_policy → published / manual_review / blinded(스위치)
```

### Qwen3Guard 헤드를 그대로 옮겨 붙이지 않는 이유

두 모델은 `hidden_size` 가 같아도(2560 예상, `check_env.py` 로 확인) 사전학습이 달라 벡터의 뜻이 다르다.
Qwen3Guard 헤드 가중치를 Qwen3.5 본체에 붙이면 형태만 맞고 판정은 무작위에 가깝다. 그래서
**Qwen3Guard 는 교사로 쓰고**(teacher_label.py), 헤드는 Qwen3.5 벡터 위에서 새로 학습한다(증류).
"Qwen3Guard 헤드를 초기값으로 쓰면 더 빨리 수렴하는가" 는 §6 의 선택 실험으로 남긴다.

### 설계 결정

| 항목 | 결정 | 이유 |
| --- | --- | --- |
| 본체 | Qwen3.5-4B 얼림 | 원래 능력(비전·프롬프트 이해)을 건드리지 않는다. 학습이 싸다 |
| 양자화 | 하지 않음 | 4B bf16 ≈ 9GB 로 16GB 한 장에 들어간다. 교사는 다른 GPU |
| 풀링 | 마지막 실제 토큰 | 디코더 모델은 마지막 토큰이 앞 문맥 전체를 본다 |
| 패딩 | 오른쪽 | 선형 어텐션(Gated DeltaNet)의 순환 상태가 패딩에 오염되지 않는다 |
| 길이 초과 | 자르지 않고 `manual_review` | 자르면 뒷부분이 무심사가 된다 (DBH llm.ts 원칙) |
| 차단 | 기본 꺼짐 (`GUARD_ALLOW_BLOCK`) | 정상 글 오차단 0건. 실험 통과 전에는 사람 검토로만 |
| 헤드 없음 | 전부 `manual_review` | 학습 안 된 헤드의 출력은 무작위다 |

## 3. 실험 단계

| 단계 | 할 일 | 명령 | 통과 기준 |
| --- | --- | --- | --- |
| **E0** 환경 | GPU·라이브러리·config 확인 | `scripts/check_env.py` | GPU 2장, 두 모델 config 로드, `hidden_size` 기록 |
| **E1** 흐름 | 가짜 본체로 학습→평가→서버 | README 4단계 | 오류 없이 끝까지 |
| **E2** 교사 라벨 | 게시글·댓글에 Qwen3Guard 판정 | `teacher_label.py` | 형식 못 읽은 행 0 |
| **E3** 헤드 학습 | 사람 라벨 또는 교사 라벨로 | `train_head.py` | 검증 손실 수렴 |
| **E4** 적합도 | 따로 둔 시험셋 채점 | `evaluate.py` | §4 표 |
| **E5** 서버 | 실제 본체·헤드로 API 시험 | `server.py` + curl | §4 지연 기준, 실패는 전부 `manual_review` |

## 4. 합격 기준

| 지표 (evaluate.py 출력) | 기준 | 비고 |
| --- | --- | --- |
| `false_block` (정상인데 위험도 ≥ block) | **0건** | DBH 제1원칙. 넘으면 `min_block_for_zero_false_block` 보다 block 을 올린다 |
| `teacher_agreement` (Qwen3Guard 와 판정 일치) | 90% 이상 | 적합도 실험 1. 사람 라벨 데이터에 `teacher` 필드가 있을 때 |
| `missed` (위반인데 위험도 ≤ safe) | 0건 목표 | 넘으면 safe 를 `max_safe_for_zero_missed` 아래로 내린다 |
| `auroc` | 0.95 이상 (초안) | 임계값과 무관한 분리 능력 |
| `ece` | 0.1 이하 (초안) | 위험도 0.9 가 실제로 90% 위반에 가까운지 |
| `manual_review_rate` | 기록 | 너무 높으면 자동화 효과가 없다. 임계값과 함께 본다 |
| `latency_ms_p95` (배치 1) | 300ms 이하 | RTX 4080, 4,096 토큰 이하 |

시험셋 구성: DBH 회귀 테스트셋 기준(정상 50 + 위반 50 + 인젝션 30)을 따르되, 첫 목표는 bot=post 이므로
게시글·댓글 정상/위반을 먼저 채운다. **학습에 쓴 행은 시험셋에 넣지 않는다.**

## 5. 주의

- **프롬프트(bot=prompt) 는 Qwen3Guard 를 교사로 쓰지 않는다.** "너는 이제 ~다", "이전 지시를 무시하고" 같은
  정상 프롬프트를 공격으로 보는 경향이 있어, 그대로 증류하면 오탐을 배운다. 프롬프트는 사람 라벨(파인튜닝
  기획안의 오탐 유발형 600건 포함)로 따로 학습한다(`--bot prompt`)
- 이 엔진의 `status` 는 **시험용 권고값**이다. DBH 에서 실제 판정(`decide()`)·상태 전이는 웹 서버 core 가 한다
- 로그·응답에 본문·예외 원문을 남기지 않는다. `evaluate.py --out` 도 행 번호·정답·위험도만 쓴다

## 6. 다음 단계 (이번 범위 밖)

1. **선택 실험** — Qwen3Guard-Stream 헤드 가중치를 초기값으로 썼을 때 수렴 속도·성능 비교(무작위 초기값 대비)
2. **카테고리 헤드** — 위험도 1개 대신 DBH `review_policies` 카테고리별 출력. DBH `Verdict` 와 맞추려면 필요
3. **DBH 연동** — 지금 DBH 워커(`apps/worker/src/review/vllm.ts`)는 vLLM `/v1/chat/completions` + JSON 스키마를
   부른다. 이 엔진을 붙이려면 워커에 분기를 추가하고 위험도 → `Verdict` 변환을 정해야 한다. `shadow` 모드로
   1~2주 불일치를 본 뒤 결정
4. **경로 B 와 비교** — 순정 Qwen3.5-4B + QLoRA(DBH 파인튜닝 기획안) 와 같은 시험셋으로 비교해 하나를 고른다

## 7. 원본 초안에서 고친 점

| 초안 | 고친 것 | 이유 |
| --- | --- | --- |
| `model_hybrid.py` 없음 | 작성 | 서버가 import 하는데 파일이 없었다 |
| `host="0.0.0.0"`, 포트 8000 | `127.0.0.1:8100` | 학교 서버는 바깥에 포트를 열지 않는다. 8000 은 vLLM 자리 |
| 상태값 `approved` | `published` | DBH `posts.status` 값 |
| 학습 안 된 헤드로도 `blinded` 가능 | 헤드 없으면 전부 `manual_review`, 차단은 스위치 | 무작위 출력으로 차단하면 오차단 0건이 깨진다 |
| `truncation=True, max_length=2048` | 자르지 않고 `too_long` → `manual_review` | 잘린 뒷부분이 무심사가 된다 |
| `reason` 에 예외 원문 | `reason_code` 만 | DBH: 원문·내부 정보 비노출 |
| `suggested_community` 를 입력 그대로 반환 | 제거 | 분류하지 않으면서 분류한 것처럼 보인다 |
| `violated_categories` 고정 문자열 | 제거 | 헤드가 카테고리를 내지 않는다(§6-2) |
| `hidden_size=2560` 하드코딩 | config 에서 읽음 | 멀티모달 config 는 `text_config` 안에 있다 |
| `@app.on_event("startup")` | `lifespan` | FastAPI 에서 폐기 예정 |
| `transformers>=4.46.0`, `pyyaml` | 버전은 check_env 로 확인, pyyaml 제거 | Qwen3.5 는 최신 버전이 필요하다. pyyaml 은 쓰지 않았다 |
| `<submission_data>` 태그를 클라이언트가 씌움 | 서버가 `<content>` 로 씌우고 닫는 태그 무력화 | DBH policy.ts 와 같은 울타리 |
| 인증 없음 | `GUARD_API_TOKEN` (선택), 동시 요청은 잠금 | GPU 하나를 여러 요청이 동시에 쓰지 않게 |
| 헤드 학습·평가 코드 없음 | `teacher_label.py`·`train_head.py`·`evaluate.py` | 학습 없이 붙인 헤드는 테스트할 의미가 없다 |
