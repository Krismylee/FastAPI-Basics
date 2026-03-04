"""RAG 분기 그래프 실행기.

이 파일의 목적:
- 컴파일된 분기 그래프를 실행해 의도와 문서 컨텍스트를 반환합니다.

포함 내용:
- ChatGraphRunner 클래스

사용 시점:
- 채팅 완성 서비스에서 분기 결과가 필요할 때 사용합니다.
"""

from src.services.llm_gateway import LlmGateway
from src.services.rag_graph.builder import build_chat_graph
from src.services.rag_graph.state import ChatGraphState
from src.services.rag_graph.state_keys import KEY_DOCUMENTS, KEY_INTENT, KEY_USER_MESSAGE


class ChatGraphRunner:
    """채팅 분기 그래프 실행 래퍼."""

    def __init__(self, llm_gateway: LlmGateway) -> None:
        self._graph = build_chat_graph(llm_gateway=llm_gateway)

    def run(self, user_message: str) -> ChatGraphState:
        """사용자 메시지를 그래프에 전달해 최종 상태를 반환합니다."""
        initial_state: ChatGraphState = {
            KEY_USER_MESSAGE: user_message,
            KEY_INTENT: "general",
            KEY_DOCUMENTS: [],
        }
        return self._graph.invoke(initial_state)

