"""FastAPI server for Prompt Studio: JSON API + the static frontend in web/."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.combined import EXAMPLES, analyze_prompt, batch_analyze, get_enhancer
from src.preferences import record, summary
from src.rubric import run_llm

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

app = FastAPI(title="Prompt Studio", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=WEB), name="static")


class AnalyzeIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    mode: str = "auto"
    level: str = "developer"


class BatchIn(BaseModel):
    prompts: list[str] = Field(min_length=1, max_length=100)
    mode: str = "auto"
    level: str = "developer"


class PreferenceIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    merged: str = Field(min_length=1, max_length=20000)
    choice: str
    score_before: int = 0
    score_after: int = 0
    engine: str = ""


class OllamaIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    optimized: str = Field(min_length=1, max_length=20000)
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/how-it-works")
def how_it_works():
    return FileResponse(WEB / "how-it-works.html")


@app.get("/api/health")
def health():
    try:
        get_enhancer()
    except Exception as exc:  # missing artifacts / torch
        raise HTTPException(503, f"Neural model unavailable: {exc}")
    return {"ok": True, "model": "BiLSTM + attention"}


@app.get("/api/examples")
def examples():
    return [{"name": name, "text": text} for name, text in EXAMPLES]


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    if body.mode not in ("auto", "basic", "standard", "advanced"):
        raise HTTPException(422, "mode must be auto, basic, standard or advanced")
    if body.level not in ("developer", "beginner"):
        raise HTTPException(422, "level must be developer or beginner")
    if not body.prompt.strip():
        raise HTTPException(422, "prompt is empty")
    return analyze_prompt(body.prompt, body.mode, body.level)


@app.post("/api/batch")
def batch(body: BatchIn):
    prompts = [p.strip() for p in body.prompts if p and p.strip()]
    if not prompts:
        raise HTTPException(422, "no prompts given")
    rows = batch_analyze(prompts, body.mode, body.level)
    numeric = lambda key: [r[key] for r in rows]
    return {
        "rows": rows,
        "summary": {
            "avg_before": round(sum(numeric("before")) / len(rows), 1),
            "avg_after": round(sum(numeric("after")) / len(rows), 1),
            "avg_gain": round(sum(numeric("gain")) / len(rows), 1),
            "intent_preserved": round(
                sum(1 for r in rows if r["intent_preserved"]) / len(rows) * 100),
        },
    }


@app.get("/api/preferences")
def preferences():
    return summary()


@app.post("/api/preference")
def save_preference(body: PreferenceIn):
    if body.choice not in ("original", "merged"):
        raise HTTPException(422, "choice must be 'original' or 'merged'")
    return record({
        "prompt": body.prompt,
        "merged": body.merged,
        "choice": body.choice,
        "score_before": body.score_before,
        "score_after": body.score_after,
        "engine": body.engine,
    })


@app.post("/api/ollama")
def ollama(body: OllamaIn):
    """Optional: run original vs merged prompt on a local Ollama model."""
    try:
        original = run_llm(body.prompt, body.model, base_url=body.base_url)
        optimized = run_llm(body.optimized, body.model, base_url=body.base_url)
    except Exception as exc:
        raise HTTPException(502, f"Could not reach Ollama or the model: {exc}")
    return {"original": original, "optimized": optimized}
