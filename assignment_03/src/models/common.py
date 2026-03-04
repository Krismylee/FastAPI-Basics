"""공통 API 스키마.

이 파일의 목적:
- 성공/실패 응답에서 재사용할 공통 모델을 정의합니다.

포함 내용:
- ErrorDetail
- ErrorResponse

사용 시점:
- 전역 에러 처리 미들웨어와 라우터 응답 정의에서 사용합니다.
"""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """표준 에러 상세 정보 모델."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """표준 에러 응답 모델."""

    error: ErrorDetail

