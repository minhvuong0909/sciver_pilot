"""Giao diện gọi LLM dùng chung cho M4 và verifier, cùng một model giả lập.

Code thật (vLLM) cài đặt `LLMClient`. Test dùng `ScriptedLLM`: trả lời theo
kịch bản và ghi lại mọi request, để kiểm rò rỉ bằng hash request mà không
cần GPU (SPEC mục 12.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


@dataclass
class LLMRequest:
    prompt: str
    images: list[bytes] = field(default_factory=list)
    temperature: float = 0.0
    seed: int | None = None
    max_tokens: int = 1024
    system: str | None = None


@dataclass
class LLMResponse:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(Protocol):
    model_name: str

    def generate(self, requests: Sequence[LLMRequest]) -> list[LLMResponse]: ...


Responder = Callable[[LLMRequest], str]


class ScriptedLLM:
    """Model giả: trả lời bằng hàm `responder`, hoặc lần lượt theo danh sách `script`
    (lặp lại phần tử cuối khi hết). Ghi lại mọi request trong `self.requests`."""

    model_name = "mock/scripted"

    def __init__(self, script: Sequence[str] | None = None, responder: Responder | None = None):
        if (script is None) == (responder is None):
            raise ValueError("Cần đúng một trong hai: script hoặc responder")
        self._script = list(script or [])
        self._responder = responder
        self._i = 0
        self.requests: list[LLMRequest] = []

    def _next(self, req: LLMRequest) -> str:
        if self._responder is not None:
            return self._responder(req)
        if not self._script:
            raise ValueError("script rỗng")
        text = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return text

    def generate(self, requests: Sequence[LLMRequest]) -> list[LLMResponse]:
        out = []
        for req in requests:
            self.requests.append(req)
            text = self._next(req)
            out.append(LLMResponse(text=text, prompt_tokens=len(req.prompt.split()),
                                   output_tokens=len(text.split())))
        return out
