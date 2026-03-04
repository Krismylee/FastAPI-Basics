"""채팅 완성 유스케이스 서비스.

이 파일의 목적:
- 의도 분류 분기 + 최종 LLM 생성 + 히스토리 저장 흐름을 조율합니다.

포함 내용:
- ChatCompletionService 클래스

사용 시점:
- 채팅 완성 라우터에서 non-stream/stream 요청을 처리할 때 사용합니다.
"""

from collections.abc import Iterator
import json
from typing import Literal

from src.models.chat import ChatCompletionRequest, ChatCompletionResult, RagDocument
from src.models.history import ChatHistoryCreateRequest, ChatHistoryRead
from src.services.chat_history_service import ChatHistoryService
from src.services.llm_gateway import LlmGateway
from src.services.rag_graph.run import ChatGraphRunner
from src.services.rag_graph.state_keys import KEY_DOCUMENTS, KEY_INTENT


class ChatCompletionService:
    """의도 분기 기반 채팅 완성 서비스."""

    def __init__(
        self,
        llm_gateway: LlmGateway,
        history_service: ChatHistoryService,
        graph_runner: ChatGraphRunner,
    ) -> None:
        self._llm_gateway = llm_gateway
        self._history_service = history_service
        self._graph_runner = graph_runner

    def create_completion(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        """non-stream 채팅 응답을 생성하고 히스토리를 저장합니다."""
        # Step 1. query_id보다 작은 히스토리(ASC, limit=5)를 조회하세요.
        histories = self._history_service.list_histories(
            session_id=request.session_id,
            lt_query_id=request.query_id,
            limit=5
        )
        # Step 2. graph_runner로 intent/documents를 계산하세요.
        graph_result = self._graph_runner.run(request.message)
        intent = self._normalize_intent(graph_result.get(KEY_INTENT, "general"))
        documents = graph_result.get(KEY_DOCUMENTS, [])
        # Step 3. `_build_final_messages`로 프롬프트를 만들고 generate_text를 호출하세요.
        messages = self._build_final_messages(
            user_message=request.message, 
            histories=histories, documents=documents, intent=intent
        )
        answer = self._llm_gateway.generate_text(
            messages=messages, 
            temperature=request.temperature,# or -1.0, 
            max_tokens=request.max_tokens,# or -1
        )
        # Step 4. 최종 답변을 히스토리에 저장하세요.
        self._history_service.create_history(
            ChatHistoryCreateRequest(
                session_id=request.session_id,
                query_id=request.query_id,
                user_message=request.message,
                intent=intent,
                final_answer=answer
            )
        )
        # Step 5. ChatCompletionResult를 반환하세요.
        return ChatCompletionResult(
            session_id=request.session_id,
            query_id=request.query_id,
            intent=intent,
            answer=answer,
            documents=documents
        )
        #raise NotImplementedError("create_completion을 구현하세요.")

    def stream_completion(self, request: ChatCompletionRequest) -> Iterator[str]:
        """stream 채팅 응답을 SSE 형태로 생성하고 히스토리를 저장합니다."""
        # Step 1. non-stream과 동일하게 히스토리/분기/최종 프롬프트를 준비하세요.
        histories = self._history_service.list_histories(
            session_id=request.session_id,
            lt_query_id=request.query_id,
            limit=5
        )
        graph_result = self._graph_runner.run(request.message)
        intent = self._normalize_intent(graph_result.get(KEY_INTENT, "general"))
        documents = graph_result.get(KEY_DOCUMENTS, [])
        messages = self._build_final_messages(
            user_message=request.message, 
            histories=histories, documents=documents, intent=intent
        )
        # Step 2. stream_text를 순회하며 `data: {"chunk":"..."}` 이벤트를 yield 하세요.
        chunks: list[str] = []
        for token in self._llm_gateway.stream_text(
            messages=messages, temperature=request.temperature, max_tokens=request.max_tokens
        ):
            if not token:
                continue
            chunks.append(token)
            chunk_payload = json.dumps({"chunk": token})
            yield f"data: {chunk_payload}\n\n"
        # Step 3. 모든 chunk를 합쳐 final_answer를 만들고 히스토리에 저장하세요.
        final_answer = "".join(chunks)
        self._history_service.create_history(
            ChatHistoryCreateRequest(
                session_id=request.session_id,
                query_id=request.query_id,
                user_message=request.message,
                intent=intent,
                final_answer=final_answer
            )
        )
        # Step 4. `final_answer/documents/intent/session_id/query_id`를 final 이벤트로 yield 하세요.
        final_payload = json.dumps(
            {
                "final_answer": final_answer,
                "documents": [doc.model_dump() for doc in documents],
                "intent": intent,
                "session_id": request.session_id,
                "query_id": request.query_id
            }
            )
        yield f"data: {final_payload}\n\n"
        #raise NotImplementedError("stream_completion을 구현하세요.")

    def _build_final_messages(
        self,
        user_message: str,
        histories: list[ChatHistoryRead],
        documents: list[RagDocument],
        intent: Literal["rag", "general"],
    ) -> list[dict[str, str]]:
        """최종 생성 호출용 메시지 배열을 구성합니다."""
        # Step 1. histories를 줄 단위 문자열로 직렬화하세요.
        history_lines = []
        for h in histories:
            history_lines.append(f"user: {h.user_message}")
            history_lines.append(f"assistant: {h.final_answer}")
        history_text = "\n".join(history_lines) if history_lines else "없음"
        # Step 2. documents를 줄 단위 문자열로 직렬화하세요.
        documents_text = "\n\n".join(
            f"[문서 {idx+1}]\n]"
            f"제목: {doc.title}\n"
            f"페이지: {doc.page_number}"
            f"내용: {doc.content}"
            for idx, doc in enumerate(documents)
        ) if documents else "없음"
        # Step 3. system/user 메시지 2개를 만들어 list[dict]로 반환하세요.
        if intent == "rag":
            system_prompt = (
                "너는 제공된 문서를 기반으로만 답변하는 AI 어시스턴트다.\n"
                "문서에 없는 내용은 추측하지 말라.\n"
                f"[이전 대화]\n{history_text}\n\n"
                f"[참고 문서]\n{documents_text}"
            )
        else:
            system_prompt = (
                "너는 친절하고 정확하게 답변하는 AI 어시스턴트다.\n"
                f"[이전 대화]\n{history_text}"
            )
        user_prompt = f"사용자 질문:\n{user_message}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        #raise NotImplementedError("_build_final_messages를 구현하세요.")

    def _normalize_intent(self, intent: str) -> Literal["rag", "general"]:
        """의도 문자열을 허용 값으로 정규화합니다."""
        if intent == "rag":
            return "rag"
        return "general"

