"""Hybrid mood inference: VADER sentiment + TF-IDF similarity to mood profiles + keyword cues."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Representative text per mood for lexical similarity (TF-IDF / cosine).
MOOD_PROFILES: dict[str, str] = {
    "happy": (
        "joyful elated cheerful upbeat grateful smiling laugh fun party celebrate "
        "great day sunshine positive energy content blessed wonderful excited good mood"
    ),
    "sad": (
        "crying tears lonely empty grief heartbroken depressed hopeless blue down "
        "melancholy loss mourning hurt devastated miss them sorrow exhausted emotionally drained"
    ),
    "calm": (
        "peaceful relaxed zen meditative breathe slow quiet unwind cozy soft gentle "
        "stillness resting low key mellow chill peaceful evening need to decompress"
    ),
    "energetic": (
        "pumped hype workout dance run fast loud adrenaline motivated ready to go "
        "festive active explosive bass festival midnight drive unstoppable fierce"
    ),
    "anxious": (
        "worried nervous panic stressed overthinking cant sleep racing heart deadline "
        "uneasy on edge jittery overwhelmed what if scared meeting tomorrow stomach knots"
    ),
    "angry": (
        "furious mad rage irritated hate unfair resent bitter annoyed livid frustrated "
        "seething betrayal done with yelling stewing heated wronged"
    ),
    "romantic": (
        "love crush date kiss longing partner soulmate butterflies wedding intimate "
        "holding hands roses valentine moonlight tender affection falling for them"
    ),
    "nostalgic": (
        "remember childhood memories old times hometown years ago miss those days "
        "throwback grandpa summer 90s wish i could go back photo album simpler times"
    ),
}

EMOJI_HINTS: dict[str, list[str]] = {
    "happy": ["😊", "😁", "🎉", "✨", "💃", "🙂"],
    "sad": ["😢", "😭", "💔", "😞", "🥺"],
    "calm": ["😌", "🧘", "🌙", "☕", "🌿"],
    "energetic": ["🔥", "⚡", "💪", "🎸", "🕺"],
    "anxious": ["😰", "😟", "😨", "🫠"],
    "angry": ["😡", "🤬", "💢", "😤"],
    "romantic": ["❤️", "💕", "🌹", "😍"],
    "nostalgic": ["🕰️", "📼", "🌅", "🧸"],
}

KEYWORD_WEIGHT = 0.35
EMOJI_WEIGHT = 0.2
TFIDF_WEIGHT = 0.45


@dataclass
class MoodResult:
    mood: str
    label_display: str
    confidence: float
    vader: dict[str, float]
    scores_by_mood: dict[str, float]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "label_display": self.label_display,
            "confidence": round(self.confidence, 4),
            "vader": {k: round(v, 4) for k, v in self.vader.items()},
            "scores_by_mood": {k: round(v, 4) for k, v in self.scores_by_mood.items()},
            "rationale": self.rationale,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _keyword_hits(text: str) -> dict[str, float]:
    text_l = _normalize(text)
    scores = {m: 0.0 for m in MOOD_PROFILES}
    for mood, profile in MOOD_PROFILES.items():
        for token in profile.split():
            if len(token) < 3:
                continue
            if token in text_l:
                scores[mood] += 1.0
    max_hits = max(scores.values()) or 1.0
    return {m: scores[m] / max_hits for m in scores}


def _emoji_scores(text: str) -> dict[str, float]:
    scores = {m: 0.0 for m in MOOD_PROFILES}
    for mood, emojis in EMOJI_HINTS.items():
        for e in emojis:
            if e in text:
                scores[mood] += 1.0
    m = max(scores.values()) or 1.0
    return {k: scores[k] / m for k in scores}


def _vader_alignment(compound: float) -> dict[str, float]:
    """Soft prior: valence steers compatible moods (not decisive alone)."""
    c = max(-1.0, min(1.0, compound))
    return {
        "happy": max(0.0, c),
        "sad": max(0.0, -c) * 0.95,
        "calm": max(0.0, 0.35 - abs(c) * 0.25),
        "energetic": max(0.0, c) * 0.65 + max(0.0, abs(c) - 0.2) * 0.25,
        "anxious": max(0.0, 0.55 - c) * 0.5 + max(0.0, -c) * 0.15,
        "angry": max(0.0, -c) * 0.45 + max(0.0, 0.2 - c) * 0.35,
        "romantic": max(0.0, c) * 0.5 + 0.08,
        "nostalgic": max(0.0, -c * 0.35) + 0.12,
    }


class MoodEngine:
    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self._moods = list(MOOD_PROFILES.keys())
        corpus = [MOOD_PROFILES[m] for m in self._moods]
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
        )
        self._mood_matrix = self._vectorizer.fit_transform(corpus)

    def infer(self, text: str) -> MoodResult:
        raw = text or ""
        norm = _normalize(raw)
        if not norm:
            return MoodResult(
                mood="calm",
                label_display="Calm / neutral",
                confidence=0.0,
                vader={"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0},
                scores_by_mood={m: 1.0 / len(self._moods) for m in self._moods},
                rationale="No text provided; defaulting to calm picks.",
            )

        v = self._vader.polarity_scores(raw)
        user_vec = self._vectorizer.transform([norm])
        sims = cosine_similarity(user_vec, self._mood_matrix).flatten()
        tfidf_map = {self._moods[i]: float(sims[i]) for i in range(len(self._moods))}

        kw = _keyword_hits(raw)
        em = _emoji_scores(raw)
        va = _vader_alignment(v["compound"])

        combined: dict[str, float] = {}
        for m in self._moods:
            combined[m] = (
                TFIDF_WEIGHT * tfidf_map[m]
                + KEYWORD_WEIGHT * kw[m]
                + EMOJI_WEIGHT * em[m]
                + 0.25 * va[m]
            )

        best_mood = max(combined, key=combined.get)  # type: ignore[arg-type]
        sorted_scores = sorted(combined.items(), key=lambda x: -x[1])
        top = sorted_scores[0][1]
        second = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        confidence = float(np.clip((top - second + 0.05) / (top + 0.15), 0.0, 1.0))

        rationale_parts = [
            f"VADER compound={v['compound']:.2f} (valence hint).",
            f"Closest lexical profile: {max(tfidf_map, key=tfidf_map.get)}.",
        ]
        if max(kw.values()) >= 1.0:
            rationale_parts.append("Keyword overlap boosted matching moods.")
        if max(em.values()) >= 1.0:
            rationale_parts.append("Emoji cues detected.")

        display_names = {
            "happy": "Happy / uplifted",
            "sad": "Sad / low",
            "calm": "Calm / grounded",
            "energetic": "Energetic / pumped",
            "anxious": "Anxious / tense",
            "angry": "Angry / frustrated",
            "romantic": "Romantic / soft",
            "nostalgic": "Nostalgic / reflective",
        }

        return MoodResult(
            mood=best_mood,
            label_display=display_names.get(best_mood, best_mood.title()),
            confidence=confidence,
            vader=v,
            scores_by_mood=combined,
            rationale=" ".join(rationale_parts),
        )


_engine: MoodEngine | None = None


def get_engine() -> MoodEngine:
    global _engine
    if _engine is None:
        _engine = MoodEngine()
    return _engine
