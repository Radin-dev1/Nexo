import argparse
import os
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .runtime import GenerationSettings, NexoRuntime

app = FastAPI(title="Nexo AI", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16000)
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=0.5, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    model: str


@lru_cache(maxsize=1)
def get_runtime():
    return NexoRuntime(os.getenv("NEXO_MODEL", "outputs/nexo"))


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("NEXO_MODEL", "outputs/nexo")}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        runtime = get_runtime()
        text = runtime.generate(
            request.prompt,
            GenerationSettings(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            ),
        )
        return GenerateResponse(text=text, model=runtime.model_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description="Serve Nexo over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("nexo.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
