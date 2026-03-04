"""도메인 공통 예외 정의.

이 파일의 목적:
- 서비스 전반에서 재사용할 애플리케이션 예외 타입을 정의합니다.

포함 내용:
- AppError 및 하위 예외 클래스

사용 시점:
- 서비스 계층에서 비즈니스 예외를 발생시키고 HTTP 에러 응답으로 변환할 때 사용합니다.
"""

from typing import Any


class AppError(Exception):
    """애플리케이션 표준 예외."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class BadRequestError(AppError):
    """잘못된 요청을 표현하는 예외."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(400, "bad_request", message, details)


class NotFoundError(AppError):
    """리소스를 찾을 수 없음을 표현하는 예외."""

    def __init__(self, resource_name: str, resource_id: str) -> None:
        super().__init__(
            404,
            "not_found",
            f"{resource_name}을(를) 찾을 수 없습니다.",
            {"resource_id": resource_id},
        )


class ConflictError(AppError):
    """중복 또는 충돌 상태를 표현하는 예외."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(409, "conflict", message, details)


class UpstreamServiceError(AppError):
    """외부 LLM 호출 실패를 표현하는 예외."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(502, "upstream_service_error", message, details)

