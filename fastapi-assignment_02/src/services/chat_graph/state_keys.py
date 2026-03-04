"""채팅 그래프 상태 키 상수.

이 파일의 목적:
- 상태 딕셔너리 접근에 사용하는 문자열 키를 한 곳에서 관리합니다.

포함 내용:
- BranchStreamState 키 상수

사용 시점:
- 노드 및 실행기에서 하드코딩 문자열 대신 공통 키를 사용할 때 사용합니다.
"""

KEY_USER_INPUT = "user_input"
KEY_FINAL_MESSAGE = "final_message"
KEY_CURSOR = "cursor"
KEY_DELTA = "delta"
