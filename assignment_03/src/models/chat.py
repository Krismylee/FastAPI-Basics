"""채팅 완성 API 스키마 정의.

이 파일의 목적:
- 의도 분류 + RAG 분기 채팅 API에서 사용할 요청/응답 모델을 정의합니다.

포함 내용:
- ChatCompletionRequest
- RagDocument
- ChatCompletionResult 및 응답 래퍼

사용 시점:
- 채팅 라우터, 그래프 노드, 서비스 계층에서 공통 데이터 계약이 필요할 때 사용합니다.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RagDocument(BaseModel):
    """RAG 참고 문서 모델."""

    title: str
    content: str
    page_number: int


class ChatCompletionRequest(BaseModel):
    """채팅 완성 요청 모델."""

    session_id: int
    query_id: int
    message: str = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class ChatCompletionResult(BaseModel):
    """채팅 완성 결과 모델."""

    session_id: int
    query_id: int
    intent: Literal["rag", "general"]
    answer: str
    documents: list[RagDocument]


class ChatCompletionEnvelope(BaseModel):
    """채팅 완성 응답 래퍼."""

    data: ChatCompletionResult

