import pytest

from evaluation.train_policy import resolve_device


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
