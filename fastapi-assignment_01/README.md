# FastAPI 기초 Mock API 실습

목표는 FastAPI + Pydantic으로 최소 기능의 API를 직접 구현하고, pytest로 검증하는 것입니다.

## 1. 실습 목표

- Pydantic 기반 요청 바디 검증을 적용한다.
- FastAPI 엔드포인트를 구현한다.
- `streaming` 값과 관계없이 동일한 응답을 반환하도록 만든다.
- 테스트 코드로 동작을 검증한다.

## 2. 요구사항

- Endpoint: `/api/v1/chat`
- Method: `POST`
- Request Body:
  - `session_id`: `str` (uuid 형식)
  - `query_id`: `int`
  - `streaming`: `bool`
  - `message`: `str`
- Response (streaming / non-streaming 동일):

```json
{"message": "답변이 생성되었습니다."}
```

## 3. 구현 대상 파일

아래 3개 파일은 현재 `Not Implemented` 상태라고 가정합니다.

- `src/models/chat.py`
- `src/services/chat_service.py`
- `src/api/chat.py`

권장 역할 분리:

- `models`: 요청/응답 Pydantic 스키마
- `services`: 비즈니스 로직(현재는 고정 응답 생성)
- `api`: 라우터 및 HTTP 처리

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
tests/
  test_chat_api.py
```

## 5. 환경 준비

```powershell
uv venv .venv
uv sync
```

가상환경 활성화(선택):

```powershell
.\.venv\Scripts\Activate.ps1
```

## 6. 서버 실행

```powershell
uv run uvicorn src.main:app --reload
```

서버 실행 후 `http://127.0.0.1:8000/docs`에서 스웨거 UI로 확인할 수 있습니다.

## 7. 테스트 실행

```powershell
uv run pytest -v
```

테스트 목표:

- `streaming=False` 요청 시 정상 응답 확인
- `streaming=True` 요청 시 정상 응답 확인
- 두 경우 모두 응답 바디가 동일한지 확인

## 8. 구현 체크리스트

- [ ] `ChatRequest` 모델에 4개 필드가 모두 정의되어 있다.
- [ ] `session_id`가 UUID 형식으로 검증된다.
- [ ] `/api/v1/chat` POST 엔드포인트가 동작한다.
- [ ] 서비스 레이어에서 응답 메시지를 생성한다.
- [ ] 응답이 정확히 `{"message": "답변이 생성되었습니다."}` 이다.
- [ ] pytest 2개(스트리밍/논스트리밍)가 통과한다.

## 9. 힌트

- FastAPI는 함수 파라미터에 Pydantic 모델을 지정하면 요청 바디를 자동 검증합니다.
- 처음에는 하드코딩 응답으로 시작하고, 이후 단계에서 로직을 확장하세요.
- 라우터에서 로직을 직접 처리하지 말고 서비스 함수로 분리하는 습관을 들이세요.
