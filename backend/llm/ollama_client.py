from typing import List, Dict

from backend.config.settings import OLLAMA_MODEL, OLLAMA_NUM_GPU_LAYERS


def _ollama():
    try:
        import ollama
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'ollama' Python package is required for local LLM calls. "
            "Install project requirements before running live dialogue."
        ) from exc
    return ollama


class OllamaClient:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        num_gpu_layers: int = OLLAMA_NUM_GPU_LAYERS,
    ):
        self.model = model
        self.num_gpu_layers = num_gpu_layers

    def chat(self, messages: List[Dict[str, str]], system: str = "") -> str:
        payload: List[Dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        options = {"num_gpu": self.num_gpu_layers} if self.num_gpu_layers >= 0 else None
        response = _ollama().chat(model=self.model, messages=payload, options=options)
        return response["message"]["content"].strip()

    def generate(self, prompt: str, system: str = "") -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system)
