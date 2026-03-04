# FastAPI + LangGraph 과제 가이드 (단계형)

이 문서는 학습/구현 가이드입니다.  
핵심 목표는 다음 2가지를 순차적으로 완성하는 것입니다.

- SQLite 기반 Chat History CRUD
- BaseChatModel 상속 기반 LLM 호출 + LangGraph 분기 처리

동시에 한 번에 구현하지 말고, 아래 순서대로 구현하세요.

## 0. 현재 코드 상태

현재 저장소 코드는 의도적으로 일부 핵심 함수가 `NotImplementedError` 상태입니다.  
즉, **처음 실행 시 테스트가 실패하는 것이 정상**입니다.

초기 실패 원인:
- CRUD 단계 미구현 함수
- LLM(BaseChatModel) 단계 미구현 함수
- 분기/오케스트레이션 단계 미구현 함수

아래 단계 순서대로 구현하면서 실패 테스트를 줄여가세요.

## 1. 과제 목표

### 최종 기능 목표
- `POST /api/v1/chat/completions`에서 사용자 질문을 받아 처리한다.
- LangGraph에서 `의도 분류` 후:
  - `rag`면 mock RAG 문서(3개)를 컨텍스트로 사용한다.
  - `general`이면 일반 생성 경로로 진행한다.
- 최종 응답 생성은 LLM 호출로 처리한다.
- `stream=true`일 때 SSE로 chunk + final 이벤트를 반환한다.
- Chat History는 SQLite에 저장/조회/수정/삭제/전체삭제 가능해야 한다.

### 학습 목표
- FastAPI 계층 분리 (`api/services/repositories/core/models`)
- SQLite CRUD 쿼리 설계
- 예외 처리와 에러 응답 표준화
- `BaseChatModel` 상속 커스텀 구현
- 통합 테스트와 단위 성격 테스트 분리

## 2. 구현 순서 (중요)

아래 순서를 지키는 것을 권장합니다.

1. `CRUD 먼저`
2. `LLM(BaseChatModel) 다음`
3. `LangGraph 분기 + chat completion 오케스트레이션 마지막`

이 순서가 좋은 이유:
- CRUD를 먼저 고정하면 외부 LLM 이슈와 분리해서 디버깅 가능
- LLM 단계에서 실패해도 DB/API 기본 동작이 이미 보장됨
- 마지막에 분기 흐름만 연결하면 되므로 복잡도가 크게 줄어듦

## 3. 단계별 구현 가이드

## 3-1. 1단계: Chat History CRUD

### 구현 대상
- 저장소: `src/repositories/chat_history_repository.py`
- 서비스: `src/services/chat_history_service.py`
- 라우터: `src/api/histories.py`
- 모델: `src/models/history.py`

### 구현 요구사항
- `session_id + query_id` 복합 PK 유일 제약
- 조회 정렬: `query_id ASC`
- `lt_query_id` 필터 지원 (`query_id < lt_query_id`)
- 기본 `limit=5`
- 전체 삭제 API: `DELETE /api/v1/chat-histories`

### 체크리스트
- [ ] 생성 성공(201)
- [ ] 중복 생성 충돌(409)
- [ ] 목록 조회/필터 조회 동작
- [ ] 수정/단건조회/삭제 동작
- [ ] 전체 삭제 후 건수 반환

## 3-2. 2단계: BaseChatModel 기반 LLM 호출

### 구현 대상
- `src/services/langchain_chat_model.py`
- `src/services/llm_gateway.py`

### 구현 요구사항
- `EndpointChatModel(BaseChatModel)` 구현
- non-stream: `_generate` + HTTP POST
- stream: `_stream` + SSE 파싱
- `llm_gateway`는 위 모델을 사용해:
  - 의도 분류 호출
  - 최종 생성 호출
  - 스트림 생성 호출

### 의도 분류 프롬프트 규칙
- 반드시 JSON 한 줄 출력 강제
- 허용 출력:
  - `{"intent":"rag"}`
  - `{"intent":"general"}`

### 체크리스트
- [ ] HTTP 에러 시 상태코드/응답본문 일부를 로깅
- [ ] JSON 파싱 실패 시 원인 로깅
- [ ] `intent` 파싱 실패 시 안전하게 `general` 폴백

## 3-3. 3단계: LangGraph 분기 + Completion 연결

### 구현 대상
- 그래프: `src/services/rag_graph/*`
- 오케스트레이션: `src/services/chat_completion_service.py`
- 라우터: `src/api/chat.py`

### 분기 흐름
- 사용자 입력
- 의도 분류 노드
- `rag` -> mock 문서 3개 주입
- `general` -> 문서 비움
- 최종 생성 노드(LLM 호출)

### stream 응답 포맷
- chunk:
```text
data: {"chunk":"..."}
```
- final:
```text
data: {"final_answer":"...","documents":[...],"intent":"rag","session_id":1,"query_id":10}
```

## 4. NotImplemented 과제 템플릿 위치

### CRUD 단계
- `chat_history_repository.py`
  - `list_before_query_id`
  - `update`
  - `delete_all`
- `chat_history_service.py`
  - `create_history`
  - `list_histories`
  - `delete_all_histories`
- `api/histories.py`
  - `list_histories`
  - `delete_all_histories`

### LLM 단계
- `langchain_chat_model.py`
  - `_build_payload`
  - `_post_json`
  - `_stream`
- `llm_gateway.py`
  - `classify_intent`
  - `generate_text`
  - `stream_text`

### 분기/오케스트레이션 단계
- `rag_graph/nodes.py`
  - `ClassifyIntentNode.__call__`
  - `route_after_intent`
- `chat_completion_service.py`
  - `create_completion`
  - `stream_completion`
  - `_build_final_messages`

## 5. API 스펙 요약

### Chat Completion
- `POST /api/v1/chat/completions`
- request 예시:
```json
{
  "session_id": 1001,
  "query_id": 10,
  "message": "LangGraph에 대해 RAG로 검색해서 알려줘.",
  "stream": false,
  "temperature": 0.6,
  "max_tokens": 200
}
```

### History CRUD
- `POST /api/v1/chat-histories`
- `GET /api/v1/chat-histories?session_id=1001&lt_query_id=10&limit=5`
- `GET /api/v1/chat-histories/{session_id}/{query_id}`
- `PATCH /api/v1/chat-histories/{session_id}/{query_id}`
- `DELETE /api/v1/chat-histories/{session_id}/{query_id}`
- `DELETE /api/v1/chat-histories` (전체 삭제)

## 6. 테스트 전략 (파일 분리)

### 1) DB 로직 검증
- 파일: `tests/test_chat_history_api.py`
- 목적: CRUD, 중복조건, 전체삭제 검증

### 2) 실제 LLM 통합 검증
- 파일: `tests/test_llm_gateway_integration.py`
- 목적: BaseChatModel 기반 LLM 호출(stream/non-stream) 자체 검증
- 검증 포인트:
  - non-stream: 응답 텍스트가 비어있지 않음
  - stream: chunk 시퀀스가 비어있지 않음

### 3) 분기 + 실제 LLM 통합 검증
- 파일: `tests/test_chat_llm_integration_api.py`
- 목적: 실제 엔드포인트 기준 분기/stream 동작 확인
- 검증 포인트:
  - RAG 질문: `intent=="rag"` + `documents` 존재
  - 일반 질문: `intent=="general"` + `documents==[]`
  - 응답 본문 내용 자체는 검증하지 않음

## 7. 실행 방법

```powershell
uv venv .venv
uv sync
uv run uvicorn src.main:app --reload
```

Swagger:
- `http://127.0.0.1:8000/docs`

## 8. pytest 실행 방법

초기 상태(미구현 상태)에서는 실패가 정상입니다.

1. CRUD 단계 구현 후:
```powershell
uv run pytest -v tests/test_chat_history_api.py
```

2. LLM/분기 단계 구현 후:
```powershell
uv run pytest -v tests/test_llm_gateway_integration.py
```

3. 분기/오케스트레이션 구현 후:
```powershell
uv run pytest -v tests/test_chat_llm_integration_api.py
```

4. 최종:
```powershell
uv run pytest -v
```

## 9. 환경 변수

- `RAG_DB_PATH` (기본: `data/chat_history.db`)
- `LLM_ENDPOINT`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_TIMEOUT_SECONDS`

## 10. 제출 기준 예시

- [ ] API가 스펙대로 동작한다.
- [ ] 에러 응답이 일관된 형태다.
- [ ] 로그에서 DB 조회/LLM 실패 원인을 확인할 수 있다.
- [ ] `pytest` 전체 통과한다. (초기 실패 -> 단계별 구현 후 최종 통과)
- [ ] 파일 길이 200줄 제한을 지킨다.

## 11. 단계별 상세 통과 기준

이 섹션은 “어디까지 되면 다음 단계로 넘어가도 되는지”를 판단하기 위한 기준입니다.

### 11-1. 1단계(CRUD) 통과 기준

### 구현 완료 조건
- `chat_history_repository.py`
  - `list_before_query_id` 구현 완료
  - `update` 구현 완료
  - `delete_all` 구현 완료
- `chat_history_service.py`
  - `create_history`에서 중복 시 `ConflictError(409)` 변환
  - `list_histories`에서 `limit <= 0` 검증
  - `delete_all_histories` 구현 완료
- `api/histories.py`
  - `GET /api/v1/chat-histories` 구현 완료
  - `DELETE /api/v1/chat-histories` 구현 완료

### 동작 검증 조건
- 같은 `(session_id, query_id)` 재생성 시 409 반환
- `lt_query_id` 조회가 `query_id ASC` + `LIMIT`로 동작
- 전체 삭제 API가 `deleted_count` 반환
- 조회 시 `INFO` 로그에 query_ids와 count가 출력

### 테스트 통과 조건
- 명령:
```powershell
uv run pytest -q tests/test_chat_history_api.py
```
- 기대: `tests/test_chat_history_api.py` 전부 통과

### 실패 시 우선 점검 포인트
- SQLite `PRIMARY KEY(session_id, query_id)` 제약 반영 여부
- `row_factory = sqlite3.Row` 설정 여부
- `commit()` 누락 여부

### 11-2. 2단계(LLM/BaseChatModel) 통과 기준

### 구현 완료 조건
- `langchain_chat_model.py`
  - `_build_payload`, `_post_json`, `_stream` 구현
- `llm_gateway.py`
  - `classify_intent`, `generate_text`, `stream_text` 구현

### 동작 검증 조건
- 분류 프롬프트가 JSON 강제 출력 규칙을 갖고 있음
- `parse_intent` 결과가 `rag/general`로 안정 변환
- HTTP status 에러/JSON 파싱 에러 시 로그 + `UpstreamServiceError` 변환
- stream 응답에서 `data:` 라인 파싱 가능

### 테스트 통과 조건
- 명령:
```powershell
uv run pytest -q tests/test_llm_gateway_integration.py
```
- 기대:
  - non-stream 호출이 비어있지 않은 응답 텍스트를 반환
  - stream 호출이 비어있지 않은 chunk 시퀀스를 반환

### 실패 시 우선 점검 포인트
- `LLM_ENDPOINT`, `LLM_MODEL` 값 오타
- `_to_openai_message` role 매핑 오류
- stream 파싱에서 `[DONE]`/빈 라인 처리 누락

### 11-3. 3단계(분기/오케스트레이션) 통과 기준

### 구현 완료 조건
- `rag_graph/nodes.py`
  - `ClassifyIntentNode.__call__` 구현
  - `route_after_intent` 구현
- `chat_completion_service.py`
  - `create_completion`, `stream_completion`, `_build_final_messages` 구현

### 동작 검증 조건
- `query_id < 현재 query_id` 히스토리 5개가 final 프롬프트에 반영
- non-stream: 최종 답변 저장 + JSON 응답 반환
- stream: chunk 이벤트 누적 후 final 이벤트 반환 + 최종 답변 저장
- final 이벤트에 `documents`, `intent`, `session_id`, `query_id` 포함

### 테스트 통과 조건
- 명령:
```powershell
uv run pytest -q tests/test_chat_llm_integration_api.py
```
- 기대:
  - RAG 질문: stream/non-stream 모두 `intent=="rag"` + `documents` 존재
  - 일반 질문: stream/non-stream 모두 `intent=="general"` + `documents==[]`

### 실패 시 우선 점검 포인트
- stream에서 chunk 누적 문자열과 저장값 불일치
- `intent` 정규화 누락(`rag` 외 값 처리)
- history 저장 타이밍 누락(non-stream/stream 종료 후)

### 11-4. 최종 제출 전 체크

- [ ] `uv run pytest -q` 통과
- [ ] Swagger에서 수동 API 확인
- [ ] RAG 질문/일반 질문 각각 stream/non-stream 수동 호출 확인
- [ ] 로그에서 DB 조회 및 LLM 에러 원인 추적 가능
- [ ] `NotImplementedError` 남아있는지 전체 검색으로 확인

최종 확인 명령:
```powershell
rg -n "NotImplementedError" src
```
