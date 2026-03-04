"""채팅 히스토리 CRUD 스키마 정의.

이 파일의 목적:
- 채팅 히스토리 저장/조회/수정/삭제 API에서 사용할 모델을 정의합니다.

포함 내용:
- ChatHistoryCreateRequest
- ChatHistoryUpdateRequest
- ChatHistoryRead 및 응답 래퍼

사용 시점:
- 히스토리 라우터와 서비스/저장소 계층에서 사용합니다.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatHistoryCreateRequest(BaseModel):
    """히스토리 생성 요청 모델."""

    session_id: int
    query_id: int
    user_message: str = Field(min_length=1)
    intent: Literal["rag", "general"]
    final_answer: str = Field(min_length=1)


class ChatHistoryUpdateRequest(BaseModel):
    """히스토리 수정 요청 모델."""

    user_message: str | None = Field(default=None, min_length=1)
    intent: Literal["rag", "general"] | None = None
    final_answer: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_any_field(self) -> "ChatHistoryUpdateRequest":
        """최소 1개 필드는 포함되도록 검증합니다."""
        if self.user_message is None and self.intent is None and self.final_answer is None:
            raise ValueError("수정할 필드를 최소 1개 이상 포함해야 합니다.")
        return self


class ChatHistoryRead(BaseModel):
    """히스토리 조회 모델."""

    session_id: int
    query_id: int
    user_message: str
    intent: Literal["rag", "general"]
    final_answer: str
    created_at: str


class ChatHistoryEnvelope(BaseModel):
    """히스토리 단건 응답 래퍼."""

    data: ChatHistoryRead


class ChatHistoryListEnvelope(BaseModel):
    """히스토리 목록 응답 래퍼."""

    data: list[ChatHistoryRead]


class DeleteHistoryEnvelope(BaseModel):
    """히스토리 삭제 응답 래퍼."""

    data: dict[str, int]


class DeleteAllHistoryEnvelope(BaseModel):
    """히스토리 전체 삭제 응답 래퍼."""

    data: dict[str, int]

