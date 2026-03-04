"""채팅 오케스트레이션 서비스.

이 파일의 목적:
- 라우터에서 전달받은 요청을 streaming/non-streaming으로 분기합니다.

포함 내용:
- non-streaming 응답 생성 함수
- streaming 응답 생성 함수

사용 시점:
- HTTP 레이어와 도메인 구현 레이어를 분리하고 싶을 때 사용합니다.
"""

from fastapi.responses import StreamingResponse

from src.models.chat import ChatRequest, ChatResponse
from src.services.chat_graph.run import resolve_message, stream_sse_events_from_graph


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    """non-streaming 응답 모델을 생성합니다."""
    # Step 1. 분기 서비스에서 최종 메시지를 계산합니다.
    final_message = resolve_message(request.message)
    return ChatResponse(message=final_message)


def generate_streaming_response(request: ChatRequest) -> StreamingResponse:
    """streaming 응답을 SSE 형식으로 생성합니다."""
    # Step 2. LangGraph 실행 업데이트를 SSE로 바로 전송합니다.
    return StreamingResponse(
        stream_sse_events_from_graph(request.message),
        media_type="text/event-stream",
    )
