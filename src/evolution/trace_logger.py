# -*- coding: utf-8 -*-
"""
黑匣子 (Trace Logger)：记录每次生成过程的 Prompt、Context、模型输出、报错。
单例模式，支持异步写入、按 run_id 查询、追加写入 JSONL。
"""
import asyncio
import functools
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """单条追踪事件。"""
    run_id: str = Field(default="", description="运行 UUID")
    timestamp: str = Field(default="", description="ISO 时间戳")
    module: str = Field(default="", description="来源模块，如 writer / librarian")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="传入的 prompt 或参数")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="模型生成结果")
    context_snapshot: List[Any] = Field(default_factory=list, description="当时检索到的知识卡片摘要")
    error: Optional[str] = Field(default=None, description="错误信息")


class TraceLogger:
    """单例追踪日志记录器。"""
    _instance: Optional["TraceLogger"] = None
    _lock = asyncio.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "TraceLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._log_dir = Path(log_dir) if log_dir else Path(__file__).resolve().parents[2] / "data" / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, List[TraceEvent]] = {}
        self._executor = None

    def _ensure_executor(self) -> None:
        try:
            import concurrent.futures
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        except Exception:
            self._executor = None

    async def log_event(
        self,
        run_id: str,
        module: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        context_snapshot: Optional[List[Any]] = None,
        error: Optional[str] = None,
    ) -> TraceEvent:
        """异步写入一条日志。"""
        event = TraceEvent(
            run_id=run_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            module=module,
            inputs=inputs or {},
            outputs=outputs or {},
            context_snapshot=context_snapshot or [],
            error=error,
        )
        async with TraceLogger._lock:
            if run_id not in self._memory:
                self._memory[run_id] = []
            self._memory[run_id].append(event)
        return event

    def log_event_sync(
        self,
        run_id: str,
        module: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        context_snapshot: Optional[List[Any]] = None,
        error: Optional[str] = None,
    ) -> TraceEvent:
        """同步写入一条日志（供非 async 环境使用）。"""
        event = TraceEvent(
            run_id=run_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            module=module,
            inputs=inputs or {},
            outputs=outputs or {},
            context_snapshot=context_snapshot or [],
            error=error,
        )
        if run_id not in self._memory:
            self._memory[run_id] = []
        self._memory[run_id].append(event)
        return event

    def get_trace(self, run_id: str) -> List[TraceEvent]:
        """获取特定 run_id 的完整日志链。"""
        return list(self._memory.get(run_id, []))

    def save_to_file(self, file_path: Optional[Path] = None) -> Path:
        """将当前内存中的日志追加写入 JSONL 文件。"""
        path = Path(file_path) if file_path else self._log_dir / "execution_trace.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for run_id, events in self._memory.items():
                for ev in events:
                    f.write(ev.model_dump_json(exclude_none=True, ensure_ascii=False) + "\n")
        return path


def trace_execution(module_name: str):
    """装饰器：自动记录被装饰函数的输入、输出与异常。"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = TraceLogger()
            run_id = str(uuid.uuid4())
            inputs = {"args": [str(a)[:500] for a in args], "kwargs": {k: str(v)[:300] for k, v in kwargs.items()}}
            try:
                result = await func(*args, **kwargs)
                out = {"result": str(result)[:2000] if result is not None else None}
                await logger.log_event(run_id, module_name, inputs=inputs, outputs=out)
                return result
            except Exception as e:
                await logger.log_event(run_id, module_name, inputs=inputs, error=str(e))
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = TraceLogger()
            run_id = str(uuid.uuid4())
            inputs = {"args": [str(a)[:500] for a in args], "kwargs": {k: str(v)[:300] for k, v in kwargs.items()}}
            try:
                result = func(*args, **kwargs)
                out = {"result": str(result)[:2000] if result is not None else None}
                logger.log_event_sync(run_id, module_name, inputs=inputs, outputs=out)
                return result
            except Exception as e:
                logger.log_event_sync(run_id, module_name, inputs=inputs, error=str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
