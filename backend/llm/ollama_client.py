from typing import List, Dict
import ollama

from backend.config.settings import OLLAMA_MODEL


class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def chat(self, messages: List[Dict[str, str]], system: str = "") -> str:
        payload: List[Dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        response = ollama.chat(model=self.model, messages=payload)
        return response["message"]["content"].strip()

    def generate(self, prompt: str, system: str = "") -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system)
