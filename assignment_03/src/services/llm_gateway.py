"""외부 LLM 게이트웨이 구현.

이 파일의 목적:
- 의도 분류와 최종 응답 생성을 BaseChatModel 기반으로 호출합니다.

포함 내용:
- LlmGateway 프로토콜
- HttpLlmGateway 구현

사용 시점:
- 채팅 완성 서비스에서 분류/생성 LLM 호출이 필요할 때 사용합니다.
"""

from collections.abc import Iterator
from typing import Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.services.langchain_chat_model import EndpointChatModel
from src.services.llm_parsing import parse_intent


class LlmGateway(Protocol):
    """LLM 호출 인터페이스."""

    def classify_intent(self, user_message: str) -> Literal["rag", "general"]:
        """사용자 메시지의 의도를 분류합니다."""

    def generate_text(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """최종 텍스트를 생성합니다."""

    def stream_text(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """최종 텍스트를 스트리밍으로 생성합니다."""


class HttpLlmGateway:
    """BaseChatModel을 사용해 외부 엔드포인트를 호출하는 게이트웨이."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        default_temperature: float,
        default_max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._chat_model = EndpointChatModel(
            endpoint=endpoint,
            model_name=model,
            request_timeout=timeout_seconds,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )

    def classify_intent(self, user_message: str) -> Literal["rag", "general"]:
        """의도 분류를 수행하고 `rag/general` 중 하나를 반환합니다."""
        # Step 1. 의도 분류용 system/user 메시지를 구성하세요.
        classify_message = [
            SystemMessage(
                content=(
                    "You are an intent classifier. Output EXACTLY one-line JSON and nothing else. "
                    "Valid outputs: {\"intent\":\"rag\"} or {\"intent\":\"general\"}. "
                    "Do not add any explanation, do not wrap in markdown, do not print anything else."
                )
            ),
            HumanMessage(content=f"질문: {user_message}")
        ]
        # Step 2. self._chat_model.invoke로 non-stream 호출하세요.
        response = self._chat_model.invoke(classify_message)
        # Step 3. parse_intent로 결과를 `rag/general`로 정규화하세요.
        intent = parse_intent(str(response.content))
        return intent
        #raise NotImplementedError("classify_intent를 구현하세요.")

    def generate_text(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """non-stream 최종 텍스트를 생성합니다."""
        # Step 1. temperature/max_tokens 기본값 처리하세요.
        #temperature_val = (temperature if temperature >= 0 else self._default_temperature)
        #max_tokens_val = max_tokens if max_tokens > 0 else self._default_max_tokens
        #temperature_val = temperature or self._default_temperature
        #max_tokens_val = max_tokens or self._default_max_tokens
        temperature_val = (
            self._default_temperature
            if temperature is None
            else temperature
        )

        max_tokens_val = (
            self._default_max_tokens
            if max_tokens is None
            else max_tokens
        )
        # Step 2. `_to_langchain_messages`로 변환하세요.
        langchain_messages = self._to_langchain_messages(messages)
        # Step 3. self._chat_model.invoke 결과 content를 문자열로 반환하세요.
        response = self._chat_model.invoke(
            langchain_messages, temperature=temperature_val, max_tokens=max_tokens_val
        )
        return str(response.content)
        #raise NotImplementedError("generate_text를 구현하세요.")

    def stream_text(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """stream 최종 텍스트를 chunk 단위로 생성합니다."""
        # Step 1. temperature/max_tokens 기본값 처리하세요.
        #temperature_val = temperature if temperature >= 0 else self._default_temperature
        #max_tokens_val = max_tokens if max_tokens > 0 else self._default_max_tokens
        temperature_val = (
            self._default_temperature
            if temperature is None
            else temperature
        )

        max_tokens_val = (
            self._default_max_tokens
            if max_tokens is None
            else max_tokens
        )
        # Step 2. `_to_langchain_messages`로 변환 후 self._chat_model.stream을 호출하세요.
        lanchain_messages = self._to_langchain_messages(messages)
        for chunk in self._chat_model.stream(
            lanchain_messages, temperature=temperature_val, max_tokens=max_tokens_val
        ):
        # Step 3. 비어있지 않은 chunk.content만 yield 하세요.
            content = str(chunk.content)
            if content:
                yield content
        #raise NotImplementedError("stream_text를 구현하세요.")

    def _to_langchain_messages(self, messages: list[dict[str, str]]) -> list[BaseMessage]:
        """dict 메시지를 LangChain 메시지로 변환합니다."""
        converted: list[BaseMessage] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

