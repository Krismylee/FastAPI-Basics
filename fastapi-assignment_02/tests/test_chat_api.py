"""채팅 API 동작 검증 테스트.

이 파일의 목적:
- streaming/non-streaming과 입력 분기 규칙이 의도대로 동작하는지 검증합니다.

포함 내용:
- non-streaming 2개 케이스 테스트
- streaming 2개 케이스 테스트

사용 시점:
- 과제 요구사항 충족 여부를 자동으로 확인할 때 사용합니다.
"""

from pathlib import Path
import json
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.main import app

client = TestClient(app)


@pytest.mark.parametrize(
    ("message", "expected_message"),
    [
        ("123", "456"),
        ("다른입력", "정해진 입력값이 아닙니다."),
    ],
)
def test_chat_non_streaming_branch_result(message: str, expected_message: str) -> None:
    """non-streaming에서 입력 분기 결과를 반환해야 합니다."""
    payload = {
        "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "query_id": 1,
        "streaming": False,
        "message": message,
    }

    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    assert response.json() == {"message": expected_message}


@pytest.mark.parametrize(
    ("message", "expected_message"),
    [
        ("123", "456"),
        ("다른입력", "정해진 입력값이 아닙니다."),
    ],
)
def test_chat_streaming_branch_result(message: str, expected_message: str) -> None:
    """streaming에서 chunk와 final 이벤트가 분기 결과와 일치해야 합니다."""
    payload = {
        "session_id": "9c858901-8a57-4791-81fe-4c455b099bc9",
        "query_id": 2,
        "streaming": True,
        "message": message,
    }

    event_payloads = []
    with client.stream("POST", "/api/v1/chat", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Step 1. streaming 응답 라인을 이터레이터로 순회하며 SSE data 필드만 파싱합니다.
        for line in response.iter_lines():
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            event_payloads.append(json.loads(line[6:]))

    chunk_text = "".join(
        event["delta"] for event in event_payloads if event.get("type") == "chunk"
    )
    final_events = [event for event in event_payloads if event.get("type") == "final"]

    assert chunk_text == expected_message
    assert len(final_events) == 1
    assert final_events[0]["message"] == expected_message
