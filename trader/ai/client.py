"""
trader/ai/client.py
可注入的 LLM 客户端抽象。

默认：OllamaClient（本地，免费，http://localhost:11434）
备选：AnthropicClient（需 ANTHROPIC_API_KEY）
      可扩展：OpenAIClient / GeminiClient

用法：
    client = OllamaClient()          # 本地 Ollama
    client = AnthropicClient()       # Claude（需 API key）

    text = client.chat("你是分析师", "分析 AAPL")
    data = client.json_chat("返回JSON", "给 AAPL 打分")  # -> dict
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.2"
_JSON_RETRY = 3


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """可注入的 LLM 接口；所有 agent 依赖此接口，不依赖具体实现。"""

    def chat(
        self,
        system: str,
        user: str,
        model: str = "",
        temperature: float = 0.3,
    ) -> str: ...

    def json_chat(
        self,
        system: str,
        user: str,
        model: str = "",
        temperature: float = 0.1,
    ) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Ollama（本地，优先）
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    调用本地 Ollama REST API（http://localhost:11434）。
    Ollama 未运行时优雅降级，返回空字符串/空字典，不崩溃。

    安装：https://ollama.com  →  ollama pull llama3.2
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_OLLAMA_URL,
        default_model: str = _DEFAULT_OLLAMA_MODEL,
        timeout: int = 60,
    ) -> None:
        self._url = base_url.rstrip("/")
        self._model = default_model
        self._timeout = timeout

    # ── 公共接口 ────────────────────────────────────────────────────────────

    def chat(
        self,
        system: str,
        user: str,
        model: str = "",
        temperature: float = 0.3,
    ) -> str:
        return self._call(system, user, model or self._model,
                          temperature, json_mode=False)

    def json_chat(
        self,
        system: str,
        user: str,
        model: str = "",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """调用 LLM 并强制解析 JSON。重试最多 _JSON_RETRY 次。"""
        sys_json = (
            system + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown, no explanation, just the JSON object."
        )
        for attempt in range(_JSON_RETRY):
            raw = self._call(sys_json, user, model or self._model,
                             temperature, json_mode=True)
            parsed = _try_parse_json(raw)
            if parsed is not None:
                return parsed
            logger.warning("Ollama JSON parse 失败 attempt=%d raw=%s…",
                           attempt + 1, raw[:80])
            time.sleep(0.5)
        logger.error("Ollama json_chat 多次重试失败，返回 {}")
        return {}

    # ── 内部调用 ─────────────────────────────────────────────────────────────

    def _call(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        json_mode: bool,
    ) -> str:
        import urllib.request
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._url}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
            content = body.get("message", {}).get("content", "")
            logger.debug("Ollama[%s] → %s…", model, content[:60])
            return content
        except OSError:
            logger.warning("Ollama 未运行（%s），返回空响应", self._url)
            return ""
        except Exception as exc:
            logger.error("Ollama 调用失败: %s", exc)
            return ""

    def is_available(self) -> bool:
        """检查 Ollama 是否在线。"""
        import urllib.request
        try:
            urllib.request.urlopen(f"{self._url}/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """列出已安装的模型。"""
        import urllib.request
        try:
            with urllib.request.urlopen(
                f"{self._url}/api/tags", timeout=5
            ) as resp:
                body = json.loads(resp.read())
            return [m["name"] for m in body.get("models", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Anthropic（可选）
# ---------------------------------------------------------------------------

class AnthropicClient:
    """
    调用 Anthropic Claude API（需 ANTHROPIC_API_KEY）。
    依赖 anthropic SDK（pip install anthropic）。
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._model = default_model
        if not self._api_key:
            raise ValueError(
                "AnthropicClient 需要 ANTHROPIC_API_KEY（.env 或环境变量）"
            )

    def chat(
        self,
        system: str,
        user: str,
        model: str = "",
        temperature: float = 0.3,
    ) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            msg = client.messages.create(
                model=model or self._model,
                max_tokens=1024,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        except Exception as exc:
            logger.error("Anthropic 调用失败: %s", exc)
            return ""

    def json_chat(
        self,
        system: str,
        user: str,
        model: str = "",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        sys_json = (
            system + "\n\nRespond ONLY with valid JSON. No markdown."
        )
        raw = self.chat(sys_json, user, model, temperature)
        return _try_parse_json(raw) or {}


# ---------------------------------------------------------------------------
# Stub（离线测试用，不需要任何 API）
# ---------------------------------------------------------------------------

class StubLLMClient:
    """离线测试用 stub，返回预设响应，不调用任何 API。"""

    def __init__(self, default_response: Dict[str, Any] | None = None) -> None:
        self._resp = default_response or {
            "score": 50, "reasoning": "stub response", "confidence": 0.5
        }

    def chat(self, system: str, user: str, model: str = "",
             temperature: float = 0.3) -> str:
        return json.dumps(self._resp)

    def json_chat(self, system: str, user: str, model: str = "",
                  temperature: float = 0.1) -> Dict[str, Any]:
        return dict(self._resp)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def _ollama_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """快速 ping Ollama /api/tags 端点，2s 内无响应视为不可达。"""
    try:
        import urllib.request as _ur
        _ur.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def make_client(provider: str = "", **kwargs) -> LLMClient:
    """
    provider: "ollama" | "anthropic" | "stub"
    默认读 .env LLM_PROVIDER，没有则先尝试 ollama。

    自动降级顺序（当 provider 未指定时）：
      1. Ollama（本地，若可达）
      2. Anthropic（若 ANTHROPIC_API_KEY 存在）
      3. StubLLMClient（离线兜底，会返回 score=50 中性值）
    """
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider == "stub":
        return StubLLMClient(**kwargs)

    # Ollama 路径：先做健康检查再返回
    base_url = kwargs.pop("base_url", None) or os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)
    model = kwargs.pop("default_model", None) or os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)

    if _ollama_reachable(base_url):
        logger.info("LLM: Ollama @ %s  model=%s", base_url, model)
        return OllamaClient(base_url=base_url, default_model=model, **kwargs)

    # Ollama 不可达 → 尝试 Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        logger.warning("Ollama 不可达 → 自动切换 AnthropicClient")
        return AnthropicClient(api_key=anthropic_key)

    # 最终兜底：Stub（所有分数返回 50，会记录明显警告）
    logger.warning(
        "⚠️  Ollama 不可达且无 ANTHROPIC_API_KEY → StubLLMClient（分数均为 50，仅供测试）"
        "  设置 LLM_PROVIDER=anthropic 并配置 ANTHROPIC_API_KEY 可启用真实 LLM。"
    )
    return StubLLMClient()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 响应中提取 JSON（容忍 markdown 代码块）。"""
    if not text:
        return None
    # 去掉 ```json ... ``` 包装
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # 找第一个 { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
