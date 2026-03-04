"""채팅 그래프 실행기.

이 파일의 목적:
- 그래프 실행 결과를 non-streaming 응답과 SSE 스트리밍으로 변환합니다.

포함 내용:
- resolve_message 함수
- stream_sse_events_from_graph 함수

사용 시점:
- 오케스트레이션 서비스에서 그래프 실행 결과를 소비할 때 사용합니다.
"""

from collections.abc import Iterator
from src.services.chat_graph.builder import build_stream_graph

import json
from src.services.chat_graph.state import BranchStreamState
from src.services.chat_graph.state_keys import (
    KEY_CURSOR,
    KEY_DELTA,
    KEY_FINAL_MESSAGE,
    KEY_USER_INPUT,
)


STREAM_GRAPH = build_stream_graph()


def resolve_message(message: str) -> str:
    """non-streaming용 최종 메시지를 반환합니다."""
    # Step 1. message를 이용해 초기 상태를 구성합니다.
    initial_state : BranchStreamState = {
        KEY_USER_INPUT: message,
        KEY_FINAL_MESSAGE: "",
        KEY_CURSOR: 0,
        KEY_DELTA: ""
        }
    # Step 2. STREAM_GRAPH.invoke(...)로 그래프를 끝까지 실행합니다.
    result = STREAM_GRAPH.invoke(initial_state)
    # Step 3. 최종 상태에서 final_message를 꺼내 반환합니다.
    return result[KEY_FINAL_MESSAGE]

    #_ = message
    #raise NotImplementedError("resolve_message 함수 구현이 필요합니다.")


def stream_sse_events_from_graph(message: str) -> Iterator[str]:
    """그래프 업데이트를 SSE 이벤트로 직렬화해 반환합니다."""
    # Step 1. message를 이용해 초기 상태를 구성합니다.
    initial_state : BranchStreamState = {
        KEY_USER_INPUT: message,
        KEY_FINAL_MESSAGE: "",
        KEY_CURSOR: 0,
        KEY_DELTA: ""
        }
    final_message = ""
    # Step 2. STREAM_GRAPH.stream(..., stream_mode='updates')를 순회합니다.
    for update in STREAM_GRAPH.stream(initial_state, stream_mode="updates"):
        if "route_message" in update:
            final_message = update["route_message"].get(KEY_FINAL_MESSAGE, "")

     # Step 3. emit_chunk 업데이트마다 chunk 이벤트를 yield 합니다.
        if "emit_chunk" in update:
            delta = update["emit_chunk"].get(KEY_DELTA, "")
            if delta:
                chunk_data = json.dumps({"type": "chunk", "delta": delta}, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
    # Step 4. 마지막에 final 이벤트를 1회 yield 합니다.
    final_data = json.dumps({"type": "final", "message": final_message}, ensure_ascii=False)
    yield f"event: final\ndata: {final_data}\n\n"
    
    
    #_ = message
    #raise NotImplementedError("stream_sse_events_from_graph 함수 구현이 필요합니다.")
