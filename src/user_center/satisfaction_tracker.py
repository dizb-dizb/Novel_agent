# -*- coding: utf-8 -*-
"""
满意度追踪（Satisfaction Tracker）：记录用户对已生成内容的赞/踩，动态修正用户画像。
实现「读者模块与用户专家联动」中的反馈闭环。
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import FeedbackRecord, UserProfile, TropePreference


class SatisfactionTracker:
    """
    记录用户对段落/章节的点赞或踩，支持持久化与基于反馈的画像微调。
    """

    def __init__(self, storage_path: Optional[Path] = None, user_id: str = ""):
        self.user_id = user_id
        self.storage_path = Path(storage_path) if storage_path else None
        self._records: List[FeedbackRecord] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.is_file():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self._records = [FeedbackRecord(**r) for r in (data.get("records") or [])]
        except Exception:
            self._records = []

    def save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(
                {"user_id": self.user_id, "records": [r.model_dump() for r in self._records]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def record(
        self,
        segment_id: str,
        book_id: str = "",
        chapter_index: int = -1,
        rating: int = 0,
        comment: str = "",
    ) -> None:
        """记录一条反馈：rating 1 赞 / -1 踩 / 0 取消。"""
        r = FeedbackRecord(
            segment_id=segment_id,
            book_id=book_id,
            chapter_index=chapter_index,
            rating=rating,
            comment=comment,
        )
        # 同 segment 覆盖
        self._records = [x for x in self._records if x.segment_id != segment_id]
        if rating != 0:
            self._records.append(r)
        if self.storage_path:
            self.save()

    def get_recent_feedback(self, book_id: str = "", limit: int = 20) -> List[FeedbackRecord]:
        """取最近 limit 条反馈，可选按 book_id 过滤。"""
        out = list(self._records)
        if book_id:
            out = [x for x in out if x.book_id == book_id]
        return out[-limit:]

    def get_rating_summary(self, book_id: str = "") -> Dict[str, Any]:
        """汇总赞/踩数量。"""
        subset = self.get_recent_feedback(book_id=book_id, limit=500)
        likes = sum(1 for x in subset if x.rating == 1)
        dislikes = sum(1 for x in subset if x.rating == -1)
        return {"likes": likes, "dislikes": dislikes, "total": len(subset)}


def update_profile_from_feedback(
    profile: UserProfile,
    records: List[FeedbackRecord],
    strength_delta: float = 0.1,
) -> UserProfile:
    """
    根据反馈记录微调用户画像：踩多的方向减弱偏好，赞多的方向增强。
    简化实现：仅根据整体赞/踩比例微调 payoff_urgency 与 preferred_pacing。
    """
    if not records:
        return profile
    likes = sum(1 for r in records if r.rating == 1)
    dislikes = sum(1 for r in records if r.rating == -1)
    total = likes + dislikes
    if total == 0:
        return profile
    ratio = likes / total
    # 赞多 -> 略提高 payoff_urgency（用户更喜欢「有爆点」）；踩多 -> 略降低
    delta = (ratio - 0.5) * 2 * strength_delta
    new_urgency = max(0.0, min(1.0, profile.payoff_urgency + delta))
    return profile.model_copy(update={"payoff_urgency": new_urgency})
