import time
from pathlib import Path
from typing import Any


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
        elif self.model_size == "medium":
            return {"text": "transcript"}
        else:
            return Path("benchmarks/examples/10-min-talk-reference.txt").read_text()


def mock_load_model(model_size: str):
    return MockWhisperModel(model_size)


class MockPipeline:
    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.model = model

    def __call__(self, inputs: str) -> list[dict[str, str]]:
        return [{"summary_text": "summary"}]


def load_mock_pipeline(name, model):
    return MockPipeline(name=name, model=model)


def mock_load_audio(path: str):
    return None
