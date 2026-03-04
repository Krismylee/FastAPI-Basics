# FastAPI + LangGraph 분기/스트리밍 과제

이 과제의 목표는 `streaming`/`non-streaming` 분기와 LangGraph 기반 흐름을 이해하고, SSE 스트리밍 파싱까지 테스트로 검증하는 것입니다.

## 1. 과제 목표

- `POST /api/v1/chat` 엔드포인트를 통해 채팅 요청을 처리합니다.
- `streaming=False`일 때는 JSON 응답을 반환합니다.
- `streaming=True`일 때는 `text/event-stream`으로 SSE 이벤트를 반환합니다.
- 입력 메시지가 `"123"`이면 `"456"`, 그 외에는 `"정해진 입력값이 아닙니다."`를 반환합니다.
- 스트리밍에서는 문자 단위 `chunk` 이벤트와 마지막 `final` 이벤트를 전송합니다.

## 2. 이번 단계에서 수정할 파일

이번 과제에서는 아래 2개 파일만 구현 대상으로 봅니다.

- `src/services/chat_graph/nodes.py`
- `src/services/chat_graph/run.py`

아래 파일들은 제공 코드로 간주하고 그대로 사용합니다.

- `src/api/chat.py`
- `src/models/chat.py`
- `src/services/chat_service.py`
- `src/services/chat_graph/builder.py`
- `src/services/chat_graph/state.py`
- `src/services/chat_graph/state_keys.py`

## 3. 동작 요구사항

### 3.1 공통 입력

- Endpoint: `/api/v1/chat`
- Method: `POST`
- Request Body:
  - `session_id`: UUID 문자열
  - `query_id`: 정수
  - `streaming`: 불리언
  - `message`: 문자열

### 3.2 분기 규칙

- `message == "123"` -> `"456"`
- 그 외 -> `"정해진 입력값이 아닙니다."`

### 3.3 non-streaming 응답

```json
{"message": "456"}
```

또는

```json
{"message": "정해진 입력값이 아닙니다."}
```

### 3.4 streaming 응답(SSE)

`data:` 라인으로 JSON payload를 전송합니다.

- `{"type":"chunk","delta":"한글자"}`
- `{"type":"final","message":"전체문자열"}`

## 4. 프로젝트 구조

```text
src/
  main.py
  api/
    chat.py
  models/
    chat.py
  services/
    chat_service.py
    chat_graph/
      __init__.py
      builder.py
      nodes.py          # 과제 구현 대상
      run.py            # 과제 구현 대상
      state.py
      state_keys.py
tests/
  test_chat_api.py
```

## 5. 실행 방법

### 5.1 환경 준비

```powershell
uv venv .venv
uv sync
```

### 5.2 서버 실행

```powershell
uv run uvicorn src.main:app --reload
```

Swagger: `http://127.0.0.1:8000/docs`

### 5.3 테스트 실행

```powershell
uv run pytest -q
```

## 6. 테스트 포인트

- non-streaming + `"123"`
- non-streaming + `"다른입력"`
- streaming + `"123"`
- streaming + `"다른입력"`

테스트에서는 streaming 응답을 문자열 전체로 처리하지 않고, `response.iter_lines()`로 한 줄씩 순회하면서 `data:` payload를 파싱합니다.

## 7. 구현 체크리스트

- [ ] `RouteMessageNode`가 분기 규칙을 정확히 반영한다.
- [ ] `EmitChunkNode`가 문자 단위로 `delta`를 생성한다.
- [ ] `should_continue`가 반복 종료 시점을 올바르게 판단한다.
- [ ] `resolve_message`가 최종 메시지를 정확히 반환한다.
- [ ] `stream_sse_events_from_graph`가 `chunk`와 `final` 이벤트를 올바른 순서로 전송한다.
- [ ] pytest 4개 케이스가 모두 통과한다.

## 8. 구현 가이드

### 8.1 `nodes.py` 구현 순서

1. `RouteMessageNode.__call__`
  - `state[KEY_USER_INPUT]` 값을 읽습니다.
  - `"123"`이면 `"456"`, 아니면 `"정해진 입력값이 아닙니다."`를 선택합니다.
  - `KEY_FINAL_MESSAGE`, `KEY_CURSOR`, `KEY_DELTA`를 갱신한 새 상태를 반환합니다.
2. `EmitChunkNode.__call__`
  - `KEY_CURSOR`, `KEY_FINAL_MESSAGE`를 읽습니다.
  - 커서가 문자열 길이 이상이면 `KEY_DELTA`를 빈 문자열로 반환합니다.
  - 아니면 현재 글자를 `KEY_DELTA`로 넣고 커서를 1 증가시킵니다.
3. `should_continue`
  - `KEY_CURSOR < len(state[KEY_FINAL_MESSAGE])`면 `"continue"`, 아니면 `"stop"`을 반환합니다.

### 8.2 `run.py` 구현 순서

1. `resolve_message`
  - 초기 상태를 만들고 `STREAM_GRAPH.invoke(initial_state)`를 호출합니다.
  - 반환된 최종 상태에서 `KEY_FINAL_MESSAGE`를 꺼내 반환합니다.
2. `stream_sse_events_from_graph`
  - 초기 상태를 만들고 `STREAM_GRAPH.stream(..., stream_mode="updates")`를 순회합니다.
  - `route_message` 업데이트에서 최종 메시지를 저장합니다.
  - `emit_chunk` 업데이트에서 `delta`가 있으면 `{"type":"chunk","delta":"..."}`를 `data:` 라인으로 `yield`합니다.
  - 순회가 끝나면 `{"type":"final","message":"..."}`를 `data:` 라인으로 한 번 `yield`합니다.

### 8.3 제출 전 확인

1. `uv run pytest -q` 실행
2. 4개 테스트 통과 확인
3. 수정 파일이 `nodes.py`, `run.py` 범위를 벗어나지 않았는지 확인
