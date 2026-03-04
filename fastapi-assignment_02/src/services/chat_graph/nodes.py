"""채팅 그래프 노드 구현.

이 파일의 목적:
- LangGraph에서 실행되는 노드를 클래스로 분리해 정의합니다.

포함 내용:
- RouteMessageNode
- EmitChunkNode
- should_continue 함수

사용 시점:
- 그래프 빌더에서 노드 단위를 독립적으로 조립할 때 사용합니다.
"""

from typing import Literal

from src.services.chat_graph.state import BranchStreamState


class RouteMessageNode:
    """입력값을 최종 메시지로 라우팅하는 노드입니다."""

    def __call__(self, state: BranchStreamState) -> BranchStreamState:
        """분기 규칙을 적용해 상태를 갱신합니다."""
        # Step 1. state에서 사용자 입력을 읽습니다.
        user_input = state.get("user_input", "")

        # Step 2. "123"이면 "456", 아니면 안내 문구를 선택합니다.
        if user_input == "123":
            final_message = "456"
        else:
            final_message = "정해진 입력값이 아닙니다."

        # Step 3. final_message/cursor/delta를 갱신한 상태를 반환합니다.
        state["user_input"] = user_input
        state["final_message"] = final_message
        state["cursor"] = 0
        state["delta"] = ""
        return state
        #raise NotImplementedError("RouteMessageNode.__call__ 구현이 필요합니다.")


class EmitChunkNode:
    """문자 단위 chunk를 하나씩 생성하는 노드입니다."""

    def __call__(self, state: BranchStreamState) -> BranchStreamState:
        """현재 커서 기준으로 1글자 delta를 생성합니다."""
        # Step 1. state에서 final_message와 cursor를 읽습니다.
        final_message = state.get("final_message", "")
        cursor = state.get("cursor", 0)
        
        # Step 2. 종료 조건이면 빈 delta를 반환합니다.
        if cursor >= len(final_message):
            state["delta"] = 0
            return state
        
        # Step 3. 아니라면 현재 문자 delta와 증가된 cursor를 반환합니다.
        state["delta"] = final_message[cursor]
        state["cursor"] = cursor + 1
        return state
        #raise NotImplementedError("EmitChunkNode.__call__ 구현이 필요합니다.")


def should_continue(state: BranchStreamState) -> Literal["continue", "stop"]:
    """chunk 노드 반복 여부를 결정합니다."""
    # Step 1. cursor와 final_message 길이를 비교합니다.
    cursor = state.get("cursor", 0)
    final_message = state.get("final_message", "")

    # Step 2. 반복이 필요하면 "continue", 종료면 "stop"을 반환합니다.
    if cursor < len(final_message):
        return "continue"
    return "stop"
    #raise NotImplementedError("should_continue 함수 구현이 필요합니다.")
