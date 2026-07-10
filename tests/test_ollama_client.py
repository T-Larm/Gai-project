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


class _FakeStreamingOllamaModule:
    """chat(stream=True) yields dict chunks like the real ollama package."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.chat_calls = []

    def chat(self, model, messages, options=None, stream=False):
        self.chat_calls.append(
            {"model": model, "messages": messages, "options": options, "stream": stream}
        )
        assert stream is True
        return ({"message": {"content": t}} for t in self.tokens)


def test_chat_stream_yields_tokens_in_order(monkeypatch):
    fake = _FakeStreamingOllamaModule(["Hel", "lo", " there", "."])
    monkeypatch.setattr(ollama_module, "_ollama", lambda: fake)

    client = OllamaClient(num_gpu_layers=20)
    tokens = list(client.chat_stream([{"role": "user", "content": "hi"}], system="be brief"))

    assert tokens == ["Hel", "lo", " there", "."]
    call = fake.chat_calls[0]
    assert call["stream"] is True
    assert call["options"] == {"num_gpu": 20}
    assert call["messages"][0] == {"role": "system", "content": "be brief"}


def test_chat_stream_skips_empty_tokens(monkeypatch):
    fake = _FakeStreamingOllamaModule(["Hi", "", ".", ""])
    monkeypatch.setattr(ollama_module, "_ollama", lambda: fake)

    client = OllamaClient(num_gpu_layers=-1)
    tokens = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert tokens == ["Hi", "."]
    assert fake.chat_calls[0]["options"] is None
