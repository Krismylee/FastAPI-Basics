"""RAG 분기 그래프 빌더.

이 파일의 목적:
- 의도 분류 기반 분기 LangGraph를 생성합니다.

포함 내용:
- build_chat_graph 함수

사용 시점:
- 서비스 초기화 시 그래프 컴파일을 수행할 때 사용합니다.
"""

from langgraph.graph import END, START, StateGraph

from src.services.llm_gateway import LlmGateway
from src.services.rag_graph.nodes import (
    ClassifyIntentNode,
    GenerateNode,
    MockRagNode,
    route_after_intent,
)
from src.services.rag_graph.state import ChatGraphState


def build_chat_graph(llm_gateway: LlmGateway):
    """의도 분기 그래프를 생성하고 컴파일합니다."""
    graph = StateGraph(ChatGraphState)
    graph.add_node("classify_intent", ClassifyIntentNode(llm_gateway))
    graph.add_node("rag_node", MockRagNode())
    graph.add_node("generate_node", GenerateNode())

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"rag": "rag_node", "general": "generate_node"},
    )
    graph.add_edge("rag_node", "generate_node")
    graph.add_edge("generate_node", END)
    return graph.compile()

