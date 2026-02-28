# -*- coding: utf-8 -*-
"""
统一 LLM 调用：支持 OpenAI 与国内模型（DeepSeek / Kimi-Moonshot / 智谱 GLM）。
高质量（大脑型）用于协议生成、抽样、冲突检测、逻辑回溯；
低成本（劳工型）用于滑动窗口批量提取。
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

# ---------- 国内模型推荐配置 ----------
# 高质量：DeepSeek-R1(推理) / Kimi(长上下文) / 智谱 GLM-4.7(旗舰)
# 低成本：DeepSeek-V3 / GLM-4-Flash / 智谱 GLM-4.7-Flash(免费)
# 智谱 API 文档：https://docs.bigmodel.cn/cn/guide/models/text/glm-4.7#python

PROVIDER_CONFIG = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "models": {
            "deepseek-reasoner": "推理模式，协议/因果/冲突检测",
            "deepseek-chat": "通用对话，批量提取",
        },
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "key_env": "MOONSHOT_API_KEY",
        "models": {
            "moonshot-v1-128k": "128k 长上下文，前 50 章宏观扫描",
            "moonshot-v1-32k": "32k 上下文",
            "kimi-k2.5": "Kimi K2.5 长文本（约 256k token），元协议/整书采样",
            "kimi-k2-turbo-preview": "Kimi K2 Turbo 预览，256k 上下文，元协议/整书采样推荐",
        },
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "ZHIPU_API_KEY",
        "models": {
            "glm-4": "通用",
            "glm-4-flash": "高性价比/免费额度，批量清洗与提取",
            "glm-4.5-air": "GLM-4.5-Air 轻量高性价比，128k 上下文，批量提取推荐",
            "glm-4.7": "GLM-4.7 旗舰版，200K 上下文，思考模式",
            "glm-4.7-flash": "GLM-4.7-Flash 轻量高速/免费，批量提取",
        },
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "models": {"gpt-4o": "高质量", "gpt-4o-mini": "低成本"},
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
        "models": {
            "qwen-max": "文笔生成、正文写作",
            "qwen-plus": "通义千问 Plus",
            "qwen-turbo": "通义千问 Turbo",
        },
    },
}

# 模型 -> (provider, base_url, key_env)
_MODEL_MAP: Dict[str, Tuple[str, Optional[str], str]] = {}
for _prov, _cfg in PROVIDER_CONFIG.items():
    _base = _cfg.get("base_url")
    _key = _cfg["key_env"]
    for _m in _cfg.get("models") or []:
        _MODEL_MAP[_m] = (_prov, _base, _key)
# 别名：方便 env 里写简称
_MODEL_MAP["deepseek-r1"] = ("deepseek", PROVIDER_CONFIG["deepseek"]["base_url"], "DEEPSEEK_API_KEY")
_MODEL_MAP["deepseek-v3"] = ("deepseek", PROVIDER_CONFIG["deepseek"]["base_url"], "DEEPSEEK_API_KEY")
_MODEL_MAP["kimi"] = ("moonshot", PROVIDER_CONFIG["moonshot"]["base_url"], "MOONSHOT_API_KEY")
_moonshot_base = PROVIDER_CONFIG["moonshot"]["base_url"]
_MODEL_MAP["kimk2.5"] = ("moonshot", _moonshot_base, "MOONSHOT_API_KEY")
_MODEL_MAP["Kimi-K2-Turbo-预览"] = ("moonshot", _moonshot_base, "MOONSHOT_API_KEY")
# 智谱 GLM-4.7 系列别名（与官方文档一致）
_zhipu_base = PROVIDER_CONFIG["zhipu"]["base_url"]
_MODEL_MAP["GLM-4.7"] = ("zhipu", _zhipu_base, "ZHIPU_API_KEY")
_MODEL_MAP["GLM-4.7-FlashX"] = ("zhipu", _zhipu_base, "ZHIPU_API_KEY")
_MODEL_MAP["GLM-4.7-Flash"] = ("zhipu", _zhipu_base, "ZHIPU_API_KEY")
# Qwen 系列（阿里云百炼）
_qwen_base = PROVIDER_CONFIG["qwen"]["base_url"]
_MODEL_MAP["qwen-max"] = ("qwen", _qwen_base, "DASHSCOPE_API_KEY")
_MODEL_MAP["qwen-plus"] = ("qwen", _qwen_base, "DASHSCOPE_API_KEY")
_MODEL_MAP["qwen-turbo"] = ("qwen", _qwen_base, "DASHSCOPE_API_KEY")
# 显示名/别名 -> 实际 API 的 model 参数（DeepSeek 接口用 deepseek-chat / deepseek-reasoner）
_API_MODEL_ID_OVERRIDE: Dict[str, str] = {
    "GLM-4.7": "glm-4.7",
    "GLM-4.7-FlashX": "glm-4.7-flash",
    "GLM-4.7-Flash": "glm-4.7-flash",
    "kimi-k2.5": "kimi-k2.5",
    "kimk2.5": "kimi-k2.5",
    "Kimi-K2-Turbo-预览": "kimi-k2-turbo-preview",
    "deepseek-v3": "deepseek-chat",
}


def get_api_key(provider: str = "openai") -> Optional[str]:
    """从环境变量读取 API Key。"""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if provider != "openai":
        key = os.getenv(PROVIDER_CONFIG.get(provider, {}).get("key_env", f"{provider.upper()}_API_KEY")) or key
    return key


def _resolve_model(model: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    解析模型名 -> (base_url, api_key, model_id)。
    base_url 为 None 表示使用 OpenAI 默认。
    """
    model = (model or "").strip()
    if model in _MODEL_MAP:
        prov, base_url, key_env = _MODEL_MAP[model]
        key = os.getenv(key_env) or (os.getenv("OPENAI_API_KEY") if prov == "openai" else None)
        api_model_id = _API_MODEL_ID_OVERRIDE.get(model, model)
        return base_url, key, api_model_id
    # 未命中则当作 OpenAI 模型名
    key = get_api_key("openai")
    return None, key, model


def get_model_config(model: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    解析模型名，返回 (base_url, api_key, model_id)，供异步/多模型场景使用。
    例如：WritingAgent 推理用 DeepSeek R1、文笔用 Qwen-max。
    """
    base_url, key, model_id = _resolve_model(model)
    return base_url, key, model_id


def _call_chat(messages: List[Dict[str, str]], model: str) -> str:
    """
    统一调用：按模型名选择 base_url 与 key，调用 OpenAI 兼容的 chat/completions。
    智谱 GLM-4.7 系列支持 max_tokens、thinking 等参数，见官方文档：
    https://docs.bigmodel.cn/cn/guide/models/text/glm-4.7#python
    """
    try:
        from openai import OpenAI
    except ImportError:
        return ""
    base_url, key, model_id = _resolve_model(model)
    if not key:
        return ""
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    create_kwargs: Dict[str, Any] = {"model": model_id, "messages": messages}
    # 智谱 API 扩展参数（OpenAI 兼容接口透传）
    if base_url and "bigmodel.cn" in (base_url or ""):
        create_kwargs["max_tokens"] = 65536  # 智谱支持最大 128K 输出，默认给足上限内常用值
        if model_id == "glm-4.7":
            create_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    r = client.chat.completions.create(**create_kwargs)
    if r.choices:
        return (r.choices[0].message.content or "").strip()
    return ""


def chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    """同步调用 chat。model 为空时使用 OPENAI_MODEL 或默认。"""
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return _call_chat(messages, model)


def chat_high_quality(messages: List[Dict[str, str]]) -> str:
    """
    高质量模型（大脑型）：协议生成、智能抽样、冲突检测、逻辑回溯。
    推荐：DeepSeek-R1 / Kimi(moonshot-v1-128k)。通过 ANALYZER_HIGH_MODEL 配置。
    """
    model = os.getenv("ANALYZER_HIGH_MODEL", "deepseek-reasoner")
    return _call_chat(messages, model)


def chat_medium_quality(messages: List[Dict[str, str]]) -> str:
    """
    中质量模型（整合型）：逐章提取后的冲突检测与合并、因果树整合。
    通过 ANALYZER_MEDIUM_MODEL 配置；未配置时回退到高质量模型。
    """
    model = os.getenv("ANALYZER_MEDIUM_MODEL", "").strip()
    if not model:
        return chat_high_quality(messages)
    return _call_chat(messages, model)


def chat_low_cost(messages: List[Dict[str, str]]) -> str:
    """
    低成本/极速模型（劳工型）：逐章提取、滑动窗口批量提取、知识卡片 JSON 输出。
    优先使用 ANALYZER_FAST_MODEL（如 glm-4-flash、deepseek-chat）加速 JSON 提取；
    未设置时使用 ANALYZER_LOW_MODEL，默认 glm-4.5-air。
    """
    model = (
        os.getenv("ANALYZER_FAST_MODEL", "").strip()
        or os.getenv("ANALYZER_LOW_MODEL", "glm-4.5-air")
    )
    return _call_chat(messages, model)


def chat_long_context(messages: List[Dict[str, str]]) -> str:
    """
    长上下文模型：整书智能采样、元知识模板设计（Agent/长对话）。
    默认：Kimi K2 Turbo 预览（kimi-k2-turbo-preview，约 256k token）。通过 ANALYZER_LONG_CONTEXT_MODEL 配置。
    """
    model = os.getenv("ANALYZER_LONG_CONTEXT_MODEL", "kimi-k2-turbo-preview")
    return _call_chat(messages, model)


def chat_json(messages: List[Dict[str, str]], model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """调用 LLM 并尝试将回复解析为 JSON。"""
    raw = chat(messages, model=model)
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        for start in ("```json", "```"):
            if raw.startswith(start):
                raw = raw[len(start) :].strip()
                break
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
