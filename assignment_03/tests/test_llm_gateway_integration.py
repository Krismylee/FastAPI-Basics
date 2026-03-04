"""LLM 게이트웨이 통합 테스트.

이 파일의 목적:
- BaseChatModel 기반 LLM 호출이 stream/non-stream 모두 정상 동작하는지 검증합니다.

포함 내용:
- HttpLlmGateway non-stream 호출 테스트
- HttpLlmGateway stream 호출 테스트

사용 시점:
- 3-2 단계(LLM 연동) 구현 완료 여부를 독립적으로 점검할 때 사용합니다.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.llm_gateway import HttpLlmGateway

LLM_ENDPOINT = "http://35.216.126.198:31933/api/infer/IML2026020001/v1/chat/completions"
#LLM_ENDPOINT = "http://35.216.126.198:31178/api/infer/IML2026020001/v1/chat/completions"
LLM_MODEL = "/repository/binary/BIN0016"


def _create_gateway() -> HttpLlmGateway:
    """실제 엔드포인트를 사용하는 게이트웨이를 생성합니다."""
    return HttpLlmGateway(
        endpoint=LLM_ENDPOINT,
        model=LLM_MODEL,
        default_temperature=0.6,
        default_max_tokens=120,
        timeout_seconds=30.0,
    )


def test_llm_gateway_non_stream_response() -> None:
    """non-stream 호출이 비어있지 않은 응답 텍스트를 반환하는지 검증합니다."""
    gateway = _create_gateway()
    messages = [
        {"role": "system", "content": "너는 간단히 답하는 도우미다."},
        {"role": "user", "content": "짧게 인사해줘."},
    ]
    answer = gateway.generate_text(messages=messages, temperature=0.6, max_tokens=60)
    assert isinstance(answer, str)
    assert answer.strip() != ""


def test_llm_gateway_stream_response() -> None:
    """stream 호출이 비어있지 않은 chunk 시퀀스를 반환하는지 검증합니다."""
    gateway = _create_gateway()
    messages = [
        {"role": "system", "content": "너는 간단히 답하는 도우미다."},
        {"role": "user", "content": "짧게 인사해줘."},
    ]
    chunks = list(gateway.stream_text(messages=messages, temperature=0.6, max_tokens=60))
    joined = "".join(chunks)
    assert len(chunks) > 0
    assert joined.strip() != ""

