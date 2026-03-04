"""채팅 히스토리 API 테스트.

이 파일의 목적:
- 히스토리 DB CRUD/중복/전체삭제 로직을 API 레벨에서 검증합니다.

포함 내용:
- 전체삭제 후 생성/조회/수정/삭제 시나리오
- 중복 session_id/query_id 충돌 시나리오

사용 시점:
- SQLite 히스토리 로직의 회귀를 확인할 때 사용합니다.
"""

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.config import AppSettings
from src.main import create_app


def _create_client(database_path: str) -> TestClient:
    """DB 검증용 앱 클라이언트를 생성합니다."""
    settings = AppSettings(
        database_path=database_path,
        #llm_endpoint="http://35.216.126.198:31178/api/infer/IML2026020001/v1/chat/completions",
        llm_endpoint="http://35.216.126.198:31933/api/infer/IML2026020001/v1/chat/completions",
        llm_model="/repository/binary/BIN0016",
        llm_temperature=0.6,
        llm_max_tokens=200,
        llm_timeout_seconds=30.0,
    )
    app = create_app(settings=settings)
    return TestClient(app)


def test_history_api_full_flow_and_conflict(tmp_path: Path) -> None:
    """히스토리 API 전체 흐름과 중복 충돌을 검증합니다."""
    client = _create_client(database_path=str(tmp_path / "history_full.db"))

    delete_all_first = client.delete("/api/v1/chat-histories")
    assert delete_all_first.status_code == 200
    assert delete_all_first.json()["data"]["deleted_count"] >= 0

    create_1 = client.post(
        "/api/v1/chat-histories",
        json={
            "session_id": 999,
            "query_id": 1,
            "user_message": "첫 질문",
            "intent": "general",
            "final_answer": "첫 답변",
        },
    )
    assert create_1.status_code == 201

    create_2 = client.post(
        "/api/v1/chat-histories",
        json={
            "session_id": 999,
            "query_id": 2,
            "user_message": "둘째 질문",
            "intent": "rag",
            "final_answer": "둘째 답변",
        },
    )
    assert create_2.status_code == 201

    duplicate = client.post(
        "/api/v1/chat-histories",
        json={
            "session_id": 999,
            "query_id": 2,
            "user_message": "중복 질문",
            "intent": "general",
            "final_answer": "중복 답변",
        },
    )
    assert duplicate.status_code == 409

    list_response = client.get(
        "/api/v1/chat-histories",
        params={"session_id": 999, "limit": 5},
    )
    assert list_response.status_code == 200
    histories = list_response.json()["data"]
    assert len(histories) == 2
    assert [item["query_id"] for item in histories] == [1, 2]

    filtered = client.get(
        "/api/v1/chat-histories",
        params={"session_id": 999, "lt_query_id": 2, "limit": 5},
    )
    assert filtered.status_code == 200
    filtered_items = filtered.json()["data"]
    assert len(filtered_items) == 1
    assert filtered_items[0]["query_id"] == 1

    get_one = client.get("/api/v1/chat-histories/999/1")
    assert get_one.status_code == 200
    assert get_one.json()["data"]["final_answer"] == "첫 답변"

    patch_one = client.patch(
        "/api/v1/chat-histories/999/1",
        json={"final_answer": "수정된 첫 답변"},
    )
    assert patch_one.status_code == 200
    assert patch_one.json()["data"]["final_answer"] == "수정된 첫 답변"

    delete_one = client.delete("/api/v1/chat-histories/999/1")
    assert delete_one.status_code == 200

    not_found = client.get("/api/v1/chat-histories/999/1")
    assert not_found.status_code == 404

    delete_all_end = client.delete("/api/v1/chat-histories")
    assert delete_all_end.status_code == 200
    assert delete_all_end.json()["data"]["deleted_count"] >= 1

