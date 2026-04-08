"""Load curated catalog and sample recommendations per mood."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_CATALOG: dict[str, Any] | None = None


def _data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "catalog.json"


def load_catalog() -> dict[str, Any]:
    global _CATALOG
    if _CATALOG is None:
        with open(_data_path(), encoding="utf-8") as f:
            _CATALOG = json.load(f)
    return _CATALOG


def recommend(
    mood: str,
    *,
    per_category: int = 5,
    seed: int | None = None,
) -> dict[str, Any]:
    catalog = load_catalog()
    key = mood if mood in catalog else "calm"
    block = catalog[key]
    rng = random.Random(seed)

    def pick(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        if len(items) <= n:
            return list(items)
        return rng.sample(items, n)

    return {
        "mood": key,
        "songs": pick(block.get("songs", []), per_category),
        "movies": pick(block.get("movies", []), per_category),
        "series": pick(block.get("series", []), per_category),
    }
