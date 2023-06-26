import time
from pathlib import Path


class MockGPUEnergyMeter:
    def __init__(self, device_index=0) -> None:
        self.device_index = device_index
        self.energy = 1_000

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MockWhisperModel:
    available_model_sizes = [
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "largev2",
    ]

    def __init__(self, model_size) -> None:
        assert model_size in self.available_model_sizes
        self.model_size = model_size

    def transcribe(self, audio):
        time.sleep(0.15)
        if self.model_size == "tiny":
            return ""
        else:
            return Path("benchmarks/examples/10-min-talk-reference.txt").read_text()


def mock_load_model(model_size: str):
    return MockWhisperModel(model_size)
