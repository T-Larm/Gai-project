import backend.llm.ollama_client as ollama_module
from backend.llm.ollama_client import OllamaClient


class _FakeOllamaModule:
    def __init__(self):
        self.chat_calls = []

    def chat(self, model, messages, options=None):
        self.chat_calls.append({"model": model, "messages": messages, "options": options})
        return {"message": {"content": " hi there "}}


def test_chat_passes_num_gpu_option(monkeypatch):
    fake = _FakeOllamaModule()
    monkeypatch.setattr(ollama_module, "_ollama", lambda: fake)

    client = OllamaClient(num_gpu_layers=20)
    reply = client.chat([{"role": "user", "content": "hello"}], system="be brief")

    assert reply == "hi there"
    call = fake.chat_calls[0]
    assert call["options"] == {"num_gpu": 20}
    assert call["messages"][0] == {"role": "system", "content": "be brief"}


def test_chat_omits_options_when_layers_unlimited(monkeypatch):
    fake = _FakeOllamaModule()
    monkeypatch.setattr(ollama_module, "_ollama", lambda: fake)

    client = OllamaClient(num_gpu_layers=-1)
    client.chat([{"role": "user", "content": "hello"}])

    assert fake.chat_calls[0]["options"] is None
