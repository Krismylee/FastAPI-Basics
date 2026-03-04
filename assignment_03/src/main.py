"""채팅 분기형 RAG FastAPI 앱 진입점.

이 파일의 목적:
- 채팅 완성 API와 히스토리 CRUD API를 등록한 앱을 생성합니다.

포함 내용:
- create_app 함수
- FastAPI 앱 객체
- 로컬 실행용 main 함수

사용 시점:
- `uvicorn src.main:app` 실행 또는 테스트 import 시 사용합니다.
"""

import logging

from fastapi import FastAPI
import uvicorn

from src.api.chat import router as chat_router
from src.api.histories import router as histories_router
from src.core.config import AppSettings, load_app_settings
from src.core.db import initialize_database
from src.core.logging_config import configure_logging
from src.core.middleware import register_error_middleware

logger = logging.getLogger(__name__)


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """FastAPI 앱을 생성합니다."""
    configure_logging()
    app_settings = settings or load_app_settings()
    initialize_database(app_settings.database_path)

    app = FastAPI(title="채팅 분기형 RAG 튜토리얼", version="0.3.0")
    app.state.settings = app_settings

    logger.info(
        "App configured with llm_endpoint=%s llm_model=%s db_path=%s",
        app_settings.llm_endpoint,
        app_settings.llm_model,
        app_settings.database_path,
    )

    register_error_middleware(app)
    app.include_router(chat_router)
    app.include_router(histories_router)
    return app


app = create_app()


def main() -> None:
    """로컬 개발 서버를 실행합니다."""
    uvicorn.run("src.main:app", host="127.0.0.1", port=8010, reload=False)


if __name__ == "__main__":
    main()

