"""FastAPI app: mood analysis and entertainment recommendations."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.mood_engine import get_engine
from app.recommender import load_catalog, recommend

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Moodify", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    per_category: int = Field(5, ge=1, le=10)
    recommendation_seed: int | None = Field(None, description="Optional RNG seed for stable picks.")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/moods")
def list_moods() -> dict[str, list[str]]:
    return {"moods": sorted(load_catalog().keys())}


@app.post("/api/analyze")
def analyze(body: AnalyzeRequest) -> dict:
    engine = get_engine()
    result = engine.infer(body.text)
    picks = recommend(
        result.mood,
        per_category=body.per_category,
        seed=body.recommendation_seed,
    )
    return {
        "analysis": result.as_dict(),
        "recommendations": picks,
    }


@app.get("/api/recommend")
def recommend_only(mood: str, per_category: int = 5, seed: int | None = None) -> dict:
    moods = load_catalog()
    key = mood.strip().lower()
    if key not in moods:
        raise HTTPException(status_code=404, detail=f"Unknown mood '{mood}'.")
    return recommend(key, per_category=per_category, seed=seed)


if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def serve_index() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(index)
