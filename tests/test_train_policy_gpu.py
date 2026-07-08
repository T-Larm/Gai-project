import json

import pytest

from evaluation.train_policy import resolve_device, train_policy


class _FakeDevice:
    def __init__(self, name):
        self.name = name
        self.type = name.split(":", 1)[0]

    def __str__(self):
        return self.name


class _FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeBackends:
    class cuda:
        class matmul:
            allow_tf32 = False

    class cudnn:
        allow_tf32 = False


class _FakeTorch:
    backends = _FakeBackends()

    def __init__(self, cuda_available=False):
        self.cuda = _FakeCuda(cuda_available)

    def device(self, name):
        return _FakeDevice(name)

    def set_float32_matmul_precision(self, value):
        self.matmul_precision = value


def test_resolve_device_rejects_cpu_by_default():
    with pytest.raises(RuntimeError, match="CPU training is disabled"):
        resolve_device(_FakeTorch(), "cpu")


def test_resolve_device_rejects_missing_cuda():
    with pytest.raises(RuntimeError, match="no CUDA GPU is available"):
        resolve_device(_FakeTorch(cuda_available=False), "cuda")


def test_resolve_device_allows_cuda_when_available():
    device = resolve_device(_FakeTorch(cuda_available=True), "auto")

    assert str(device) == "cuda"


def _v2_sample(action_id, mood, thi):
    return {
        "features": {
            "categorical": {"occ": "king", "arch": "aggressive"},
            "multi": {"traits": ["aggressive"], "inv": ["water"]},
            "continuous": {"thi": thi, "hun": 0.2, "max_threat": 0.0},
        },
        "label": {"action_id": action_id, "zone": "no_threat"},
        "aux": {"mood": mood},
    }


def test_train_policy_cpu_smoke_on_v2_data(tmp_path):
    pytest.importorskip("torch")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples = [_v2_sample("drink", "calm", 0.9), _v2_sample("work", "happy", 0.1)] * 4
    for split in ("train", "valid", "test"):
        (data_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(sample) for sample in samples),
            encoding="utf-8",
        )

    metrics = train_policy(
        data_dir=data_dir,
        out_dir=tmp_path / "ckpt",
        epochs=2,
        batch_size=4,
        hidden_dim=8,
        device="cpu",
        allow_cpu=True,
        amp=False,
    )

    assert "action_id" in metrics["test"]["accuracy"]
    assert "action_id" in metrics["test"]["macro_f1"]
    assert (tmp_path / "ckpt" / "model.pt").exists()
    assert (tmp_path / "ckpt" / "metadata.json").exists()
