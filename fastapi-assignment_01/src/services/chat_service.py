"""채팅 도메인 서비스.

이 파일의 목적:
- 채팅 응답 생성 비즈니스 로직을 제공합니다.

포함 내용:
- generate_chat_response 함수 자리

사용 시점:
- 실습에서 서비스 레이어를 직접 구현할 때 사용합니다.
"""

#raise NotImplementedError("TODO: src/services/chat_service.py의 generate_chat_response를 구현하세요.")


from src.models.chat import ChatRequest, ChatResponse

def generate_chat_response(request: ChatRequest) -> ChatResponse:
    return ChatResponse(message="답변이 생성되었습니다.")