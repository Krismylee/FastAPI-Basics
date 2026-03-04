"""채팅 API 요청/응답 스키마.

이 파일의 목적:
- `/api/v1/chat` 엔드포인트의 요청/응답 형태를 정의합니다.

포함 내용:
- ChatRequest: 요청 바디 모델
- ChatResponse: 응답 바디 모델

사용 시점:
- 라우터와 서비스에서 데이터 검증 및 직렬화가 필요할 때 사용합니다.
"""

from uuid import UUID

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """채팅 요청 바디 모델입니다."""

    session_id: UUID
    query_id: int
    streaming: bool
    message: str


class ChatResponse(BaseModel):
    """채팅 응답 바디 모델입니다."""

    message: str
