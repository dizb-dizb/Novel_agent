# -*- coding: utf-8 -*-
"""
补丁应用入口：读取 JSON 补丁文件并调用 file_operator 更新项目文件。
供 Cursor 或外部进程写入补丁后执行，实现「代行者」式自动改码。
用法：python -m src.evolution.apply_patch [patch_file.json]
      或：python src/evolution/apply_patch.py patches/patch.json
"""
import json
import sys
from pathlib import Path

# 确保可导入 evolution 包
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evolution.file_operator import PROJECT_ROOT, apply_edits


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m src.evolution.apply_patch <patch_file.json>")
        print("  patch_file.json 格式: [ {\"path\": \"src/writer/xx.py\", \"new_content\": \"...\"}, ... ]")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"文件不存在: {path}")
        sys.exit(2)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        print(f"解析 JSON 失败: {e}")
        sys.exit(3)
    if isinstance(data, dict):
        edits = data.get("edits") or data.get("patches") or [data]
    else:
        edits = data
    if not isinstance(edits, list):
        print("JSON 需为编辑列表或包含 edits/patches 的对象")
        sys.exit(4)
    results = apply_edits(PROJECT_ROOT, edits)
    for i, (ok, msg) in enumerate(results):
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {msg}")
    if not all(ok for ok, _ in results):
        sys.exit(5)
    print("补丁已全部应用。")


if __name__ == "__main__":
    main()
