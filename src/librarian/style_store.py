# -*- coding: utf-8 -*-
"""
写作风格指纹库：文笔基因映射 (Stylistic DNA)。
支持多维度风格采样、按场景标签召回原书描写样本；持久化指纹库用于仿写时拟合写作手法与风格习惯。
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 常用场景标签，可与后续 LLM 打标或规则扩展
DEFAULT_TAGS = ["#心理", "#打斗", "#环境", "#对话", "#回忆", "#表白", "#日常", "#悬念"]


@dataclass
class StyleSample:
    """单条风格样本。"""
    text: str
    chapter_index: int
    chapter_title: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class StyleFingerprint:
    """
    全书写作风格指纹：用于风格仿写、拟合写作手法与习惯。
    含数值统计（节奏、对话占比）与 LLM 归纳的写作手法、句式特点等。
    """
    book_id: str = ""
    title: str = ""
    avg_chapter_length: float = 0.0
    avg_paragraph_length: float = 0.0
    dialogue_ratio: float = 0.0  # 对话占比
    keyword_ratios: Dict[str, float] = field(default_factory=dict)

    # 写作手法与风格习惯（由高质量 AI 归纳，供仿写/续写拟合）
    writing_habits: str = ""  # 如：短句为主、多用心理独白、对话简洁、喜用比喻
    sentence_style: str = ""   # 句式特点：短句/长句/长短结合、是否常用反问等
    rhetoric_notes: str = ""   # 修辞与语气：比喻、夸张、口语化等

    # 原文代表性描写片段（高质量 AI 从节选中整理，每条约 80～200 字）
    representative_descriptions: List[str] = field(default_factory=list)
    # 角色代表性说话习惯片段（高质量 AI 整理：角色名/身份 + 典型说话或习惯例句）
    character_speech_samples: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "角色名", "sample": "代表性台词或习惯"}]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "book_id": self.book_id,
            "title": self.title,
            "avg_chapter_length": self.avg_chapter_length,
            "avg_paragraph_length": self.avg_paragraph_length,
            "dialogue_ratio": self.dialogue_ratio,
            "keyword_ratios": self.keyword_ratios,
            "writing_habits": self.writing_habits,
            "sentence_style": self.sentence_style,
            "rhetoric_notes": self.rhetoric_notes,
            "representative_descriptions": list(self.representative_descriptions),
            "character_speech_samples": list(self.character_speech_samples),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StyleFingerprint":
        if not isinstance(d, dict):
            return cls()
        desc = d.get("representative_descriptions") or []
        if not isinstance(desc, list):
            desc = []
        speech = d.get("character_speech_samples") or []
        if not isinstance(speech, list):
            speech = []
        speech = [x for x in speech if isinstance(x, dict) and ("role" in x or "sample" in x)]
        return cls(
            book_id=d.get("book_id", ""),
            title=d.get("title", ""),
            avg_chapter_length=float(d.get("avg_chapter_length", 0) or 0),
            avg_paragraph_length=float(d.get("avg_paragraph_length", 0) or 0),
            dialogue_ratio=float(d.get("dialogue_ratio", 0) or 0),
            keyword_ratios=dict(d.get("keyword_ratios") or {}),
            writing_habits=str(d.get("writing_habits") or ""),
            sentence_style=str(d.get("sentence_style") or ""),
            rhetoric_notes=str(d.get("rhetoric_notes") or ""),
            representative_descriptions=desc,
            character_speech_samples=speech,
        )


class StyleStore:
    """
    写作风格指纹库：根据场景标签或语义召回原书描写样本，并加载/保存风格指纹。
    可从 data/raw/{book_id} 的 JSON 加载章节；指纹库可持久化到 data/cards/{book_id}/style_fingerprint.json。
    """

    def __init__(self, book_id: str = "", title: str = ""):
        self.book_id = book_id
        self.title = title or ""
        self.samples: List[StyleSample] = []
        self.fingerprint: Optional[StyleFingerprint] = None

    def add_sample(self, text: str, chapter_index: int, chapter_title: str = "", tags: Optional[List[str]] = None) -> None:
        if not text or len(text.strip()) < 10:
            return
        self.samples.append(StyleSample(
            text=text.strip(),
            chapter_index=chapter_index,
            chapter_title=chapter_title or "",
            tags=tags or [],
        ))

    def get_samples_by_tags(
        self,
        tags: List[str],
        limit: int = 5,
        chapter_max: Optional[int] = None,
    ) -> List[StyleSample]:
        """按场景标签检索，返回与任一 tag 匹配的样本，最多 limit 条。"""
        tag_set = set(tags)
        out = []
        for s in self.samples:
            if chapter_max is not None and s.chapter_index > chapter_max:
                continue
            if tag_set and not (tag_set & set(s.tags)):
                continue
            out.append(s)
            if len(out) >= limit:
                break
        return out

    def get_samples_by_keywords(
        self,
        keywords: List[str],
        limit: int = 5,
    ) -> List[StyleSample]:
        """按关键词在文本中出现检索（简易语义）。"""
        out = []
        for s in self.samples:
            if any(kw in s.text for kw in keywords):
                out.append(s)
                if len(out) >= limit:
                    break
        return out

    def load_from_book_json(
        self,
        json_path: Path,
        tag_rules: Optional[Dict[str, List[str]]] = None,
        fingerprint_file: Optional[Path] = None,
    ) -> int:
        """
        从书籍 JSON 加载正文，按规则打标并加入样本；若提供 fingerprint_file 则优先加载指纹库（含写作习惯）。
        """
        if not json_path.is_file():
            return 0
        data = json.loads(json_path.read_text(encoding="utf-8"))
        chapters = data.get("chapters") or []
        if not self.title:
            self.title = (data.get("title") or "").strip()
        count = 0
        for ch in chapters:
            content = (ch.get("content") or "").strip()
            if not content:
                continue
            idx = chapters.index(ch)
            title = ch.get("chapter_title") or ""
            tags = []
            if tag_rules:
                for tag, kws in tag_rules.items():
                    if any(kw in content for kw in kws):
                        tags.append(tag)
            # 按段落切分，每段作为一条样本（避免单条过长）
            for para in re.split(r"\n\n+", content):
                if len(para.strip()) < 20:
                    continue
                self.add_sample(para, idx, title, tags)
                count += 1
        self._compute_fingerprint(chapters)
        if fingerprint_file and fingerprint_file.is_file():
            try:
                data = json.loads(Path(fingerprint_file).read_text(encoding="utf-8"))
                loaded = StyleFingerprint.from_dict(data)
                if loaded and self.fingerprint:
                    self.fingerprint.writing_habits = loaded.writing_habits or self.fingerprint.writing_habits
                    self.fingerprint.sentence_style = loaded.sentence_style or self.fingerprint.sentence_style
                    self.fingerprint.rhetoric_notes = loaded.rhetoric_notes or self.fingerprint.rhetoric_notes
                    self.fingerprint.title = loaded.title or self.fingerprint.title
                    if getattr(loaded, "representative_descriptions", None):
                        self.fingerprint.representative_descriptions = loaded.representative_descriptions
                    if getattr(loaded, "character_speech_samples", None):
                        self.fingerprint.character_speech_samples = loaded.character_speech_samples
            except Exception:
                pass
        return count

    def _compute_fingerprint(self, chapters: list) -> None:
        """根据已加载样本与章节列表计算数值风格指纹。"""
        if not chapters:
            return
        total_len = sum(len((c.get("content") or "")) for c in chapters)
        num_ch = len(chapters)
        paras = []
        for c in chapters:
            content = (c.get("content") or "")
            paras.extend([p for p in re.split(r"\n\n+", content) if len(p.strip()) > 1])
        dialogue_approx = 0
        for p in paras:
            if re.search(r'[「『].*[」』]', p) or "说道" in p or "道：" in p:
                dialogue_approx += len(p)
        self.fingerprint = StyleFingerprint(
            book_id=self.book_id,
            title=self.title,
            avg_chapter_length=total_len / num_ch if num_ch else 0,
            avg_paragraph_length=total_len / len(paras) if paras else 0,
            dialogue_ratio=dialogue_approx / total_len if total_len else 0,
        )

    def save_fingerprint_to_file(self, path: Path) -> None:
        """将当前风格指纹（含写作习惯）持久化到指纹库文件。"""
        if not self.fingerprint:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.fingerprint.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_fingerprint_from_file(self, path: Path) -> Optional[StyleFingerprint]:
        """从指纹库文件加载风格指纹（含写作手法与习惯）。"""
        path = Path(path)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.fingerprint = StyleFingerprint.from_dict(data)
            return self.fingerprint
        except Exception:
            return None

    def get_fingerprint(self) -> Optional[StyleFingerprint]:
        return self.fingerprint


def build_style_fingerprint_library(
    book_id: str,
    title: str,
    json_path: Path,
    out_path: Path,
    max_chars_for_llm: int = 25000,
) -> Optional[StyleFingerprint]:
    """
    构建写作风格指纹库：统计数值指纹 + LLM 归纳写作手法与风格习惯，持久化到 out_path。
    供续写/仿写时拟合写作手法和风格习惯使用。
    """
    store = StyleStore(book_id=book_id, title=title)
    n = store.load_from_book_json(json_path)
    if n == 0 or not store.fingerprint:
        return None
    fp = store.fingerprint

    # 采样正文供 LLM 归纳写作手法与习惯
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chapters = data.get("chapters") or []
    if not chapters:
        store.save_fingerprint_to_file(out_path)
        return fp
    indices = [0, len(chapters) // 4, len(chapters) // 2, 3 * len(chapters) // 4, len(chapters) - 1]
    indices = sorted(set(i for i in indices if 0 <= i < len(chapters)))[:5]
    parts = []
    total = 0
    for i in indices:
        ch = chapters[i]
        content = (ch.get("content") or "").strip()[:6000]
        if not content:
            continue
        parts.append(f"## 第{i+1}章\n{content}")
        total += len(content)
        if total >= max_chars_for_llm:
            break
    sample_text = "\n\n---\n\n".join(parts)

    try:
        from src.utils.llm_client import chat_high_quality
        user = f"""请基于以下小说节选，归纳该书的「写作手法与风格习惯」，并整理「原文代表性描写片段」与「角色代表性说话习惯」。
用于后续仿写/续写时拟合文风与口吻。

书名：{title}
book_id：{book_id}

## 节选正文

{sample_text}

---

请严格按以下 JSON 输出（仅此 JSON，不要 markdown 包裹）：
{{
  "writing_habits": "3～5 条写作习惯，如：短句为主、多用心理独白、对话简洁、节奏明快、喜用比喻 等",
  "sentence_style": "句式特点：短句/长句/长短结合；是否常用反问、设问；句号与逗号的使用节奏",
  "rhetoric_notes": "修辞与语气：比喻/夸张/口语化/书面语/幽默/冷峻 等",
  "representative_descriptions": [
    "从节选中摘录或概括的第 1 段代表性描写（叙述/环境/动作等，80～200 字）",
    "第 2 段代表性描写",
    "第 3 段（至少 3 段，最多 5 段）"
  ],
  "character_speech_samples": [
    {{ "role": "主角或角色名/身份", "sample": "该角色的一句或几句代表性台词，体现说话习惯、口吻、用词" }},
    {{ "role": "另一角色", "sample": "其代表性说话片段" }}
  ]
}}

要求：
- representative_descriptions：必须是能代表本书描写风格的原文或高度贴近原文的概括，每段 80～200 字。
- character_speech_samples：至少 2 个、最多 4 个角色的代表性说话习惯片段，sample 为书中典型台词或口吻示例。
"""
        raw = chat_high_quality([
            {"role": "system", "content": "你是文风分析专家，负责整理写作手法、代表性描写与角色说话习惯，只输出上述 JSON。"},
            {"role": "user", "content": user},
        ])
        raw = (raw or "").strip()
        for p in ("```json", "```"):
            if raw.startswith(p):
                raw = raw[len(p):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        llm_data = json.loads(raw)
        if isinstance(llm_data, dict):
            fp.writing_habits = str(llm_data.get("writing_habits") or "").strip()
            fp.sentence_style = str(llm_data.get("sentence_style") or "").strip()
            fp.rhetoric_notes = str(llm_data.get("rhetoric_notes") or "").strip()
            fp.title = title
            desc = llm_data.get("representative_descriptions")
            if isinstance(desc, list):
                fp.representative_descriptions = [str(x).strip() for x in desc if str(x).strip()][:5]
            speech = llm_data.get("character_speech_samples")
            if isinstance(speech, list):
                fp.character_speech_samples = []
                for x in speech:
                    if isinstance(x, dict) and (x.get("role") or x.get("sample")):
                        fp.character_speech_samples.append({
                            "role": str(x.get("role") or "").strip(),
                            "sample": str(x.get("sample") or "").strip(),
                        })
                fp.character_speech_samples = fp.character_speech_samples[:4]
    except Exception:
        pass
    store.save_fingerprint_to_file(out_path)
    return fp
