# -*- coding: utf-8 -*-
"""
文件操作员 (File Operator)：接收大模型生成的代码块/补丁，安全地更新项目源码。
仅允许修改 src/ 与 .cursor/rules 下的文件，防止路径穿越。
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 项目根（evolution 的 grandparents[2]）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 允许修改的路径前缀（相对于项目根）
ALLOWED_PREFIXES = ("src/", "src\\", ".cursor/rules/", ".cursor\\rules\\")


def _normalize_path(path: str) -> Path:
    """转为 Path 并统一为相对路径。"""
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(PROJECT_ROOT)
        except ValueError:
            p = Path(path)
    return Path(p.as_posix())


def _is_allowed(relative_path: str) -> bool:
    """是否在允许修改的范围内。"""
    norm = relative_path.replace("\\", "/").strip("/")
    return any(norm.startswith(prefix.rstrip("/")) for prefix in ("src", ".cursor/rules"))


def resolve_safe(project_root: Path, path: str) -> Optional[Path]:
    """
    将 path 解析为项目下的绝对路径，若穿越到项目外或不在允许列表则返回 None。
    """
    root = project_root.resolve()
    p = _normalize_path(path)
    full = (root / p).resolve()
    try:
        rel = full.relative_to(root)
    except ValueError:
        return None
    if ".." in rel.parts:
        return None
    rel_str = rel.as_posix()
    if not _is_allowed(rel_str):
        return None
    return full


def apply_patch(project_root: Path, relative_path: str, new_content: str) -> Tuple[bool, str]:
    """
    将 relative_path 指向的文件覆盖为 new_content。
    返回 (成功, 消息)。
    """
    target = resolve_safe(project_root, relative_path)
    if target is None:
        return False, f"路径不允许或非法: {relative_path}"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        return True, f"已写入 {target.relative_to(project_root)}"
    except Exception as e:
        return False, str(e)


def apply_single_edit(
    project_root: Path,
    relative_path: str,
    old_string: str,
    new_string: str,
) -> Tuple[bool, str]:
    """
    对文件做一次 str 替换（等价于 search_replace）。若 old_string 未找到则失败。
    """
    target = resolve_safe(project_root, relative_path)
    if target is None:
        return False, f"路径不允许或非法: {relative_path}"
    if not target.is_file():
        return False, f"文件不存在: {relative_path}"
    try:
        text = target.read_text(encoding="utf-8")
        if old_string not in text:
            return False, "未找到待替换的 old_string"
        text = text.replace(old_string, new_string, 1)
        target.write_text(text, encoding="utf-8")
        return True, f"已替换 {relative_path}"
    except Exception as e:
        return False, str(e)


def apply_edits(
    project_root: Path,
    edits: List[Dict[str, Any]],
) -> List[Tuple[bool, str]]:
    """
    批量应用编辑。每项为：
    - {"path": "src/writer/xx.py", "new_content": "完整文件内容"} 或
    - {"path": "src/...", "old_string": "...", "new_string": "..."}
    返回每项的 (成功, 消息) 列表。
    """
    root = Path(project_root)
    results = []
    for e in edits:
        path = e.get("path") or ""
        if "new_content" in e:
            ok, msg = apply_patch(root, path, e["new_content"])
        elif "old_string" in e and "new_string" in e:
            ok, msg = apply_single_edit(root, path, e["old_string"], e["new_string"])
        else:
            ok, msg = False, "缺少 new_content 或 old_string/new_string"
        results.append((ok, msg))
    return results


def extract_paths_from_markdown(markdown: str) -> List[str]:
    """
    从 Action Plan 的 Markdown 中提取提到的源码路径（如 src/writer/xxx.py）。
    """
    # 常见写法：src/writer/xxx.py、`src/writer/xxx.py`
    pattern = r"src/[\w/]+\.py"
    found = re.findall(pattern, markdown)
    normalized = []
    for p in found:
        p = p.replace("\\", "/")
        if p not in normalized:
            normalized.append(p)
    return normalized
