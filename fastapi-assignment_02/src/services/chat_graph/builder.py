"""채팅 그래프 빌더.

이 파일의 목적:
- 분리된 노드 클래스를 조합해 LangGraph를 생성합니다.

포함 내용:
- build_stream_graph 함수

사용 시점:
- 그래프 구성 규칙을 실행 로직과 분리하고 싶을 때 사용합니다.
"""

from langgraph.graph import END, START, StateGraph

from src.services.chat_graph.nodes import EmitChunkNode, RouteMessageNode, should_continue
from src.services.chat_graph.state import BranchStreamState


def build_stream_graph():
    """분기 + chunk 스트리밍용 LangGraph를 생성합니다."""
    graph = StateGraph(BranchStreamState)
    graph.add_node("route_message", RouteMessageNode())
    graph.add_node("emit_chunk", EmitChunkNode())
    graph.add_edge(START, "route_message")
    graph.add_edge("route_message", "emit_chunk")
    graph.add_conditional_edges(
        "emit_chunk",
        should_continue,
        {"continue": "emit_chunk", "stop": END},
    )
    return graph.compile()
