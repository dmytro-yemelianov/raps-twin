"""HTTP server for Tier 1 LLM inference service."""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn

app = FastAPI(title="EDDT Tier 1 LLM Service")

# Load model on startup
model_path = os.getenv("TIER1_MODEL_PATH", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
n_ctx = int(os.getenv("TIER1_N_CTX", "2048"))
n_gpu_layers = int(os.getenv("TIER1_N_GPU_LAYERS", "-1"))

print(f"Loading Tier 1 model from {model_path}...", file=sys.stderr)
llm = Llama(
    model_path=model_path,
    n_ctx=n_ctx,
    n_gpu_layers=n_gpu_layers,
    verbose=False,
)
print("Tier 1 model loaded successfully", file=sys.stderr)


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.1
    top_p: float = 0.9
    stop: list[str] | None = None


class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Run inference on the Tier 1 model."""
    try:
        output = llm.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop or ["\n", "Action:", "Decision:"],
        )
        return InferenceResponse(
            text=output["choices"][0]["text"].strip(),
            tokens_generated=output["usage"]["completion_tokens"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": model_path}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
