"""LLM 응답 파싱 유틸리티.

이 파일의 목적:
- completion/stream 응답에서 텍스트와 intent를 안전하게 추출합니다.

포함 내용:
- extract_text_from_completion
- extract_text_from_stream_chunk
- parse_intent

사용 시점:
- HTTP LLM 게이트웨이에서 응답 파싱 로직을 재사용할 때 사용합니다.
"""

import json
import re
from typing import Literal


def extract_text_from_completion(payload: dict[str, object]) -> str:
    """일반 completion 응답에서 텍스트를 추출합니다."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    return ""


def extract_text_from_stream_chunk(payload: dict[str, object]) -> str:
    """스트림 chunk 응답에서 텍스트를 추출합니다."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                return delta["content"]
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def parse_intent(response_text: str) -> Literal["rag", "general"]:
    """분류 응답 문자열에서 intent를 안전하게 파싱합니다."""
    cleaned = response_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    parsed_json: dict[str, object] | None = None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            parsed_json = parsed
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    parsed_json = parsed
            except json.JSONDecodeError:
                parsed_json = None

    if parsed_json is None:
        return "general"

    intent = str(parsed_json.get("intent", "general")).lower()
    if intent == "rag":
        return "rag"
    return "general"

