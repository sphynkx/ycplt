"""Loading and calling the GGUF model via llama-cpp-python.

Keeps a single Llama instance per process (module-level singleton),
available through get_llm() after load_llm() has been called at app startup.
"""
import os
import asyncio
from typing import Optional

from llama_cpp import Llama

from utils import config

_llm: Optional[Llama] = None


def load_llm() -> Llama:
    """Loads the model from config.MODEL_PATH. Raises RuntimeError with a
    clear message if the file is missing or not in GGUF format."""
    global _llm

    if not os.path.exists(config.MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {config.MODEL_PATH}. "
            f"Download a GGUF model and place it at this path (see README)."
        )
    if not config.MODEL_PATH.lower().endswith(".gguf"):
        raise RuntimeError(
            f"Expected a GGUF (.gguf) file, got: {config.MODEL_PATH}. "
            f"Old .bin/.ggml files (ggmlv2/ggmlv3) are not supported by modern llama.cpp."
        )
    try:
        _llm = Llama(
            model_path=config.MODEL_PATH,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            n_gpu_layers=config.N_GPU_LAYERS,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{config.MODEL_PATH}': {e}")

    print("Model loaded successfully.")
    return _llm


def get_llm() -> Optional[Llama]:
    return _llm


def generate_sync(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
    """Runs one generation. max_tokens=None (the default) means "no artificial
    cap" — llama-cpp-python then generates until the model emits a stop token
    or the context window (config.N_CTX) is full, whichever comes first.
    Pass an explicit max_tokens only when a caller genuinely wants a shorter
    answer (e.g. utils/intent.py's one-word classifier)."""
    if _llm is None:
        raise RuntimeError("Model is not loaded")
    out = _llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return out["choices"][0]["message"]["content"]


async def generate_async(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, generate_sync, prompt, max_tokens, temperature)
