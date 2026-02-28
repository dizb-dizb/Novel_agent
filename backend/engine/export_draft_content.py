# -*- coding: utf-8 -*-
"""
将仿写/基准测试生成的正文从 data/sft_training_dataset.jsonl 导出为可读的 .txt 文件。
每行一个 JSON，其中 draft_content 为单章正文。

用法（项目根目录）：
  python -m backend.engine.export_draft_content
  python -m backend.engine.export_draft_content --out data/仿写正文.txt --last 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSONL = _ROOT / "data" / "sft_training_dataset.jsonl"
DEFAULT_OUT = _ROOT / "data" / "imitation_draft.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="从 sft_training_dataset.jsonl 导出 draft_content 为 txt")
    parser.add_argument("--input", default=str(DEFAULT_JSONL), help="JSONL 文件路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出 txt 路径")
    parser.add_argument("--last", type=int, default=0, help="只导出最后 N 条（0=全部）")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"文件不存在: {path}")
        return
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print("无数据")
        return
    if args.last > 0:
        lines = lines[-args.last:]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines, 1):
            try:
                rec = json.loads(line)
                draft = rec.get("draft_content") or ""
                if draft:
                    f.write(f"\n\n## 第{i}章\n\n")
                    f.write(draft)
            except Exception:
                pass
    print(f"已导出 {len(lines)} 章到: {out_path}")


if __name__ == "__main__":
    main()
