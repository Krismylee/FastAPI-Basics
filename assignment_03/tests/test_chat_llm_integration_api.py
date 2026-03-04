"""실제 LLM 엔드포인트 기반 채팅 통합 테스트.

이 파일의 목적:
- 실제 LLM 호출로 의도 분기(rag/general)와 stream/non-stream 응답을 검증합니다.

포함 내용:
- RAG 질문 stream/non-stream 테스트
- 일반 질문 stream/non-stream 테스트

사용 시점:
- 외부 엔드포인트 연동 상태를 실제로 점검할 때 사용합니다.
"""

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.config import AppSettings
from src.main import create_app
LLM_ENDPOINT = "http://35.216.126.198:31933/api/infer/IML2026020001/v1/chat/completions"
#LLM_ENDPOINT = "http://35.216.126.198:31178/api/infer/IML2026020001/v1/chat/completions"
LLM_MODEL = "/repository/binary/BIN0016"
RAG_QUESTION = "LangGraph에 대해 RAG로 검색해서 알려줘."
GENERAL_QUESTION = "안녕? 너는 누구니?"


def _create_client(database_path: str) -> TestClient:
    """실제 LLM 설정으로 테스트 앱을 생성합니다."""
    settings = AppSettings(
        database_path=database_path,
        llm_endpoint=LLM_ENDPOINT,
        llm_model=LLM_MODEL,
        llm_temperature=0.6,
        llm_max_tokens=200,
        llm_timeout_seconds=30.0,
    )
    app = create_app(settings=settings)
    return TestClient(app)


def _extract_final_sse_payload(response) -> dict[str, object]:
    """SSE 응답에서 final payload를 추출합니다."""
    payloads: list[dict[str, object]] = []
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        payloads.append(json.loads(line[6:]))
    return [item for item in payloads if "final_answer" in item][0]


def test_rag_question_non_stream_and_stream(tmp_path: Path) -> None:
    """RAG 질문이 stream/non-stream 모두 rag 분기로 처리되는지 검증합니다."""
    client = _create_client(database_path=str(tmp_path / "llm_rag.db"))

    non_stream = client.post(
        "/api/v1/chat/completions",
        json={"session_id": 101, "query_id": 1, "message": RAG_QUESTION, "stream": False},
    )
    assert non_stream.status_code == 200
    non_stream_payload = non_stream.json()["data"]
    assert non_stream_payload["intent"] == "rag"
    assert len(non_stream_payload["documents"]) > 0

    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        json={"session_id": 101, "query_id": 2, "message": RAG_QUESTION, "stream": True},
    ) as stream_response:
        assert stream_response.status_code == 200
        final_payload = _extract_final_sse_payload(stream_response)
    assert final_payload["intent"] == "rag"
    assert len(final_payload["documents"]) > 0


def test_general_question_non_stream_and_stream(tmp_path: Path) -> None:
    """일반 질문이 stream/non-stream 모두 general 분기로 처리되는지 검증합니다."""
    client = _create_client(database_path=str(tmp_path / "llm_general.db"))

    non_stream = client.post(
        "/api/v1/chat/completions",
        json={"session_id": 202, "query_id": 1, "message": GENERAL_QUESTION, "stream": False},
    )
    assert non_stream.status_code == 200
    non_stream_payload = non_stream.json()["data"]
    assert non_stream_payload["intent"] == "general"
    assert non_stream_payload["documents"] == []

    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        json={"session_id": 202, "query_id": 2, "message": GENERAL_QUESTION, "stream": True},
    ) as stream_response:
        assert stream_response.status_code == 200
        final_payload = _extract_final_sse_payload(stream_response)
    assert final_payload["intent"] == "general"
    assert final_payload["documents"] == []

