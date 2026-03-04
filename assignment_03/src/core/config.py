"""애플리케이션 설정 관리.

이 파일의 목적:
- 채팅 히스토리 + 분기형 RAG 서비스의 런타임 설정을 관리합니다.

포함 내용:
- AppSettings 데이터 클래스
- `.env` 기반 설정 로더

사용 시점:
- DB 경로, LLM 엔드포인트, 모델 옵션이 필요할 때 사용합니다.
"""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppSettings:
    """애플리케이션 런타임 설정 모델."""

    database_path: str
    llm_endpoint: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: float


def load_app_settings() -> AppSettings:
    """`.env`와 환경 변수를 읽어 애플리케이션 설정을 구성합니다."""
    # Step 1. 프로젝트 루트의 `.env`를 먼저 로드합니다.
    load_dotenv(override=False)

    # Step 2. 환경 변수 값을 읽고 기본값과 함께 설정 객체를 만듭니다.
    return AppSettings(
        database_path=os.getenv("RAG_DB_PATH", "data/chat_history.db"),
        llm_endpoint=os.getenv(
            "LLM_ENDPOINT",
            "http://35.216.126.198:31178/api/infer/IML2026020001/v1/chat/completions",
        ),
        llm_model=os.getenv("LLM_MODEL", "/repository/binary/BIN0016"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.6")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    )

