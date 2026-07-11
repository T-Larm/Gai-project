from typing import Dict, Iterator, List

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
        payload = self._payload(messages, system)

        response = _ollama().chat(model=self.model, messages=payload)
        return response["message"]["content"].strip()

    def chat_stream(
        self, messages: List[Dict[str, str]], system: str = ""
    ) -> Iterator[str]:
        """Yield Ollama response text as soon as each token chunk arrives."""
        response = _ollama().chat(
            model=self.model,
            messages=self._payload(messages, system),
            stream=True,
        )
        for chunk in response:
            message = chunk["message"]
            content = message.get("content", "") if hasattr(message, "get") else message["content"]
            if content:
                yield content

    def generate(self, prompt: str, system: str = "") -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system)

    @staticmethod
    def _payload(
        messages: List[Dict[str, str]], system: str
    ) -> List[Dict[str, str]]:
        payload: List[Dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)
        return payload
