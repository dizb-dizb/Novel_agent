# -*- coding: utf-8 -*-
"""
档案员 (Snapshot)：将 src/ 与 .cursor/rules 打包为带版本与评分的快照，便于回滚与对比。
"""
import re
import zipfile
from pathlib import Path
from typing import List, Optional

# 排除目录/文件（仅用于 src 树）
EXCLUDE_NAMES = {"__pycache__", "logs", ".git", ".gitignore", ".env"}


def _should_include(path: Path, root: Path) -> bool:
    rel = path.relative_to(root) if root in path.parents or path == root else path
    parts = rel.parts if hasattr(rel, "parts") else (str(rel),)
    for name in EXCLUDE_NAMES:
        if name in parts or path.name == name:
            return False
    if path.suffix == ".pyc":
        return False
    return True


def get_next_version(snapshots_dir: Path) -> int:
    """根据已有 snapshots 目录下的 zip 文件名 v{N}_ 解析当前最大版本号，返回 N+1。"""
    if not snapshots_dir.is_dir():
        return 1
    max_v = 0
    for f in snapshots_dir.iterdir():
        if f.suffix.lower() != ".zip":
            continue
        m = re.match(r"v(\d+)_", f.stem)
        if m:
            max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def save_snapshot(
    output_dir: Optional[Path] = None,
    version: Optional[int] = None,
    avg_score: Optional[float] = None,
    note: str = "",
) -> Path:
    """
    将 src/ 与 .cursor/rules 压缩到 snapshots/v{version}_score_{avg_score}.zip，
    并在包内生成 version_report.md。
    """
    root = Path(__file__).resolve().parents[2]
    snap_dir = Path(output_dir) if output_dir else root / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    if version is None:
        version = get_next_version(snap_dir)
    score_suffix = f"{avg_score:.1f}" if avg_score is not None else "0"
    zip_name = f"v{version}_score_{score_suffix}.zip"
    zip_path = snap_dir / zip_name

    report_lines = [
        f"# Version Report v{version}",
        f"- average_score: {avg_score}",
        f"- note: {note or 'evolution snapshot'}",
    ]
    report_content = "\n".join(report_lines)

    to_add: List[tuple[Path, str]] = []
    # src/
    src_root = root / "src"
    if src_root.is_dir():
        for f in src_root.rglob("*"):
            if f.is_file() and _should_include(f, src_root) and "__pycache__" not in f.parts and "logs" not in f.parts:
                to_add.append((f, f.relative_to(root).as_posix()))
    # .cursor/rules
    cursor_rules = root / ".cursor" / "rules"
    if cursor_rules.is_dir():
        for f in cursor_rules.rglob("*"):
            if f.is_file():
                to_add.append((f, f.relative_to(root).as_posix()))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in to_add:
            try:
                zf.write(file_path, arcname)
            except Exception:
                pass
        zf.writestr("version_report.md", report_content.encode("utf-8"))
    return zip_path
