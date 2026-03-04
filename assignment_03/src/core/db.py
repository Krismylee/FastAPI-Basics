"""SQLite 초기화 유틸리티.

이 파일의 목적:
- 서비스 시작 시 채팅 히스토리 스키마를 준비합니다.

포함 내용:
- initialize_database 함수

사용 시점:
- FastAPI 앱 startup 단계에서 테이블을 생성/검증할 때 사용합니다.
"""

from pathlib import Path
import sqlite3


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_histories (
    session_id INTEGER NOT NULL,
    query_id INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    intent TEXT NOT NULL,
    final_answer TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, query_id)
);

CREATE INDEX IF NOT EXISTS idx_chat_histories_session_query
ON chat_histories(session_id, query_id);
"""


def initialize_database(database_path: str) -> None:
    """SQLite 파일과 히스토리 테이블을 준비합니다."""
    # Step 1. DB 파일 경로의 상위 디렉터리를 보장합니다.
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Step 2. 스키마를 적용해 기본 테이블을 준비합니다.
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(SCHEMA_SQL)
        connection.commit()
    finally:
        connection.close()

