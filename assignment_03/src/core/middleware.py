"""전역 에러 처리 미들웨어.

이 파일의 목적:
- 라우터/서비스에서 발생한 예외를 표준 에러 응답 형태로 통일합니다.

포함 내용:
- register_error_middleware 함수

사용 시점:
- FastAPI 앱 생성 시 전역 에러 처리 정책을 등록할 때 사용합니다.
"""

from collections.abc import Awaitable, Callable
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from src.core.errors import AppError
from src.models.common import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


def register_error_middleware(app: FastAPI) -> None:
    """FastAPI 앱에 전역 에러 처리 미들웨어를 등록합니다."""

    @app.middleware("http")
    async def error_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """표준 예외를 JSON 에러 응답으로 변환합니다."""
        try:
            return await call_next(request)
        except AppError as error:
            # Step 1. AppError도 서버 로그에 남겨 디버깅 가능성을 높입니다.
            if error.status_code >= 500:
                logger.exception(
                    "AppError status=%s code=%s path=%s details=%s",
                    error.status_code,
                    error.code,
                    request.url.path,
                    error.details,
                )
            else:
                logger.warning(
                    "AppError status=%s code=%s path=%s details=%s",
                    error.status_code,
                    error.code,
                    request.url.path,
                    error.details,
                )
            payload = ErrorResponse(
                error=ErrorDetail(
                    code=error.code,
                    message=error.message,
                    details=error.details,
                )
            ).model_dump()
            return JSONResponse(status_code=error.status_code, content=payload)
        except Exception as error:  # pylint: disable=broad-except
            logger.exception("예상하지 못한 서버 에러가 발생했습니다: %s", str(error))
            payload = ErrorResponse(
                error=ErrorDetail(
                    code="internal_server_error",
                    message="서버 내부 에러가 발생했습니다.",
                    details={},
                )
            ).model_dump()
            return JSONResponse(status_code=500, content=payload)

