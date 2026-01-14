"""llama.cpp client for local LLM inference."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import asyncio
from llama_cpp import Llama
from .inference import InferenceInterface

logger = logging.getLogger(__name__)


class LlamaCppClient(InferenceInterface):
    """Client for llama.cpp-based local LLM inference.

    Thread-safe: uses a lock to serialize access to the Llama instance
    since llama-cpp-python is not thread-safe by default.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
    ):
        """
        Initialize llama.cpp client.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_batch: Batch size for processing
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self._lock = threading.Lock()
        # Dedicated executor with single worker to serialize LLM calls
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama_cpp")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=False,
        )

    async def decide(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Run inference using llama.cpp (thread-safe)."""
        loop = asyncio.get_running_loop()

        def _call():
            with self._lock:
                try:
                    return self.llm.create_completion(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop or ["\n", "Action:", "Decision:"],
                    )
                except Exception as exc:
                    logger.error("llama.cpp completion failed: %s", exc)
                    raise

        output = await loop.run_in_executor(self._executor, _call)
        return output["choices"][0]["text"].strip()

    def close(self):
        """Shutdown the executor and release resources."""
        self._executor.shutdown(wait=False)

    async def health_check(self) -> bool:
        """Check if model is loaded and ready."""
        return self.llm is not None


class LlamaCppHttpClient(InferenceInterface):
    """HTTP client for llama.cpp service running in separate container."""

    def __init__(self, base_url: str):
        """
        Initialize HTTP client for remote llama.cpp service.

        Args:
            base_url: Base URL of the LLM service (e.g., "http://tier1:8001")
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for HTTP client. Install with: pip install httpx")

        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def decide(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Run inference via HTTP API."""
        response = await self.client.post(
            "/infer",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["text"]

    async def health_check(self) -> bool:
        """Check if remote service is healthy."""
        try:
            response = await self.client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
