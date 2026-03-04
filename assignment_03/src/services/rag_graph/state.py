"""RAG 그래프 상태 타입 정의.

이 파일의 목적:
- LangGraph 노드가 공유하는 상태 구조를 명시합니다.

포함 내용:
- ChatGraphState TypedDict

사용 시점:
- 그래프 노드 구현과 실행기에서 상태 타입 안정성을 높일 때 사용합니다.
"""

from typing import TypedDict

from src.models.chat import RagDocument


class ChatGraphState(TypedDict):
    """채팅 분기 그래프 상태 모델."""

    user_message: str
    intent: str
    documents: list[RagDocument]

