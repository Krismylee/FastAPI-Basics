"""채팅 그래프 상태 정의.

이 파일의 목적:
- LangGraph에서 사용할 상태 스키마를 정의합니다.

포함 내용:
- BranchStreamState 타입

사용 시점:
- 노드, 그래프 빌더, 실행기가 동일한 상태 구조를 공유할 때 사용합니다.
"""

from typing import TypedDict


class BranchStreamState(TypedDict):
    """LangGraph 상태 모델입니다."""

    user_input: str
    final_message: str
    cursor: int
    delta: str
