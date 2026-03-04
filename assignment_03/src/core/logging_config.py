"""로깅 설정 유틸리티.

이 파일의 목적:
- 개발/디버깅 시 INFO 로그가 터미널에 출력되도록 로깅을 구성합니다.

포함 내용:
- configure_logging 함수

사용 시점:
- 앱 시작 시 로깅 레벨과 포맷을 초기화할 때 사용합니다.
"""

import logging

_IS_CONFIGURED = False


def configure_logging() -> None:
    """애플리케이션 로깅을 초기화합니다."""
    global _IS_CONFIGURED
    if _IS_CONFIGURED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    _IS_CONFIGURED = True

