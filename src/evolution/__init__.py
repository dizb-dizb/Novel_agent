# -*- coding: utf-8 -*-
"""
Evolution 模块：生成质量演化闭环。
包含 Trace Logger、Simulator AI、Engineering Diagnostician、Snapshot 与 Runner。
"""
from .trace_logger import TraceLogger, TraceEvent, trace_execution
from .simulator import SimulatorAI, EvaluationResult
from .engineering import EngineeringDiagnostician
from .snapshot import save_snapshot, get_next_version
from .runner import EvolutionLoop
from .file_operator import apply_patch, apply_edits
from .orchestrator import run_evolution_loop, plan_to_edits

__all__ = [
    "TraceLogger",
    "TraceEvent",
    "trace_execution",
    "SimulatorAI",
    "EvaluationResult",
    "EngineeringDiagnostician",
    "save_snapshot",
    "get_next_version",
    "EvolutionLoop",
    "apply_patch",
    "apply_edits",
    "run_evolution_loop",
    "plan_to_edits",
]
