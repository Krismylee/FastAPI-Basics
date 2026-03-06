"""RAG 분기 그래프 노드 구현.

이 파일의 목적:
- 의도 분류, RAG 문서 주입, 일반 생성 경로 구성을 노드 단위로 분리합니다.

포함 내용:
- ClassifyIntentNode
- MockRagNode
- GenerateNode
- route_after_intent 함수

사용 시점:
- 그래프 빌더에서 노드를 조합해 분기 워크플로를 만들 때 사용합니다.
"""

from typing import Literal

from src.services.llm_gateway import LlmGateway
from src.services.rag_graph.state import ChatGraphState
from src.services.rag_graph.state_keys import KEY_DOCUMENTS, KEY_INTENT, KEY_USER_MESSAGE
from src.services.retrieval.mock_rag_store import get_mock_rag_documents


class ClassifyIntentNode:
    """사용자 입력 의도를 분류하는 노드."""

    def __init__(self, llm_gateway: LlmGateway) -> None:
        self._llm_gateway = llm_gateway

    def __call__(self, state: ChatGraphState) -> ChatGraphState:
        """LLM 호출 결과를 상태의 intent에 기록합니다."""
        # Step 1. state에서 사용자 질문을 읽으세요.
        question = state.get(KEY_USER_MESSAGE, "")
        # Step 2. llm_gateway.classify_intent를 호출하세요.
        intent = self._llm_gateway.classify_intent(question)
        # Step 3. state[KEY_INTENT]에 결과를 저장해 반환하세요.
        state[KEY_INTENT] = intent
        return state
        #raise NotImplementedError("ClassifyIntentNode.__call__을 구현하세요.")


class MockRagNode:
    """RAG 목업 문서를 상태에 주입하는 노드."""

    def __call__(self, state: ChatGraphState) -> ChatGraphState:
        """고정 문서 3개를 상태에 기록합니다."""
        #state[KEY_DOCUMENTS] = get_mock_rag_documents()
        query = state["user_message"]
        documents = search_rag_documents(query)
        #documents = get_rag_documents(query)
        state[KEY_DOCUMENTS] = documents
        return state


class GenerateNode:
    """최종 생성 이전 공통 정리 노드."""

    def __call__(self, state: ChatGraphState) -> ChatGraphState:
        """일반 경로에서는 문서 목록을 빈 배열로 고정합니다."""
        if state.get(KEY_INTENT, "general") == "general":
            state[KEY_DOCUMENTS] = []
        return state


def route_after_intent(state: ChatGraphState) -> Literal["rag", "general"]:
    """의도 분류 결과에 따라 다음 노드를 결정합니다."""
    # Step 1. state에서 intent를 읽으세요.
    intent = state.get(KEY_INTENT, "general")
    # Step 2. intent가 "rag"면 "rag"를 반환하고, 그 외에는 "general"을 반환하세요.
    if intent == "rag":
        return "rag"
    return "general"
    #raise NotImplementedError("route_after_intent를 구현하세요.")

