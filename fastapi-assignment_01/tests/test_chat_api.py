"""채팅 API 동작 검증 테스트.

이 파일의 목적:
- streaming/non-streaming 요청 모두 동일한 응답을 반환하는지 검증합니다.

포함 내용:
- non-streaming 테스트
- streaming 테스트

사용 시점:
- 과제 요구사항 충족 여부를 자동으로 확인할 때 사용합니다.
"""

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.main import app

client = TestClient(app)


def test_chat_non_streaming_returns_fixed_message() -> None:
    """streaming=False 요청 시 고정 메시지를 반환해야 합니다."""
    payload = {
        "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "query_id": 1,
        "streaming": False,
        "message": "안녕하세요",
    }

    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    assert response.json() == {"message": "답변이 생성되었습니다."}


def test_chat_streaming_returns_fixed_message() -> None:
    """streaming=True 요청 시에도 동일한 고정 메시지를 반환해야 합니다."""
    payload = {
        "session_id": "9c858901-8a57-4791-81fe-4c455b099bc9",
        "query_id": 2,
        "streaming": True,
        "message": "스트리밍 테스트",
    }

    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    assert response.json() == {"message": "답변이 생성되었습니다."}
