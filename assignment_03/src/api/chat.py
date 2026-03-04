"""채팅 완성 API 라우터.

이 파일의 목적:
- 의도 분류 + RAG 분기를 거쳐 최종 답변을 생성하는 엔드포인트를 제공합니다.

포함 내용:
- `POST /api/v1/chat/completions` 라우터

사용 시점:
- 애플리케이션 클라이언트가 채팅 응답을 요청할 때 사용합니다.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.core.dependencies import get_chat_completion_service
from src.models.chat import ChatCompletionEnvelope, ChatCompletionRequest
from src.services.chat_completion_service import ChatCompletionService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat/completions", response_model=None)
def create_chat_completion(
    request: ChatCompletionRequest,
    service: ChatCompletionService = Depends(get_chat_completion_service),
):
    """채팅 완성 응답을 반환합니다."""
    # Step 1. stream=true면 SSE로 chunk/final 이벤트를 반환합니다.
    if request.stream:
        return StreamingResponse(
            service.stream_completion(request=request),
            media_type="text/event-stream",
        )

    # Step 2. stream=false면 일반 JSON 응답을 반환합니다.
    result = service.create_completion(request=request)
    return ChatCompletionEnvelope(data=result)

