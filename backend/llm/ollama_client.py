from typing import List, Dict

from backend.config.settings import OLLAMA_MODEL


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
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def chat(self, messages: List[Dict[str, str]], system: str = "") -> str:
        payload: List[Dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        response = _ollama().chat(model=self.model, messages=payload)
        return response["message"]["content"].strip()

    def generate(self, prompt: str, system: str = "") -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system)
