import time
from pathlib import Path
from typing import Any

from torchmetrics.functional import word_error_rate


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

    def __init__(self, model_size, return_strategy) -> None:
        assert model_size in self.available_model_sizes
        self.model_size = model_size
        self.return_strategy = return_strategy

    def transcribe(self, audio):
        time.sleep(0.15)
        if self.return_strategy == "empty":
            return {"text": ""}
        elif self.return_strategy == "dict":
            return {"text": "transcript"}
        elif self.return_strategy == "reference":
            return {
                "text": Path(
                    "benchmarks/examples/10-min-talk-reference.txt"
                ).read_text()
            }
        return {"text": ""}


class MockModelLoader:
    def __init__(self, return_strategy=None) -> None:
        self.model_size = None
        self.return_strategy = return_strategy

    def __call__(self, model_size: str) -> Any:
        self.model_size = model_size
        return MockWhisperModel(model_size, self.return_strategy)


class MockPipeline:
    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.model = model
        self.inputs = None
        self.min_length = None
        self.max_length = None

    def __call__(
        self, inputs: str, min_length: int = None, max_length: int = None
    ) -> list[dict[str, str]]:
        self.inputs = inputs
        self.min_length = min_length
        self.max_length = max_length
        return [{"summary_text": "summary"}]


class MockPipelineLoader:
    def __init__(self) -> None:
        self.pipeline = None

    def __call__(
        self,
        name: str,
        model: str,
    ) -> Any:
        self.pipeline = MockPipeline(name=name, model=model)
        return self.pipeline


def mock_load_audio(path: str):
    return None


def mock_word_error_rate(preds: str, target: str):
    assert isinstance(preds, str), "Prediction should be a string!"
    assert isinstance(target, str), "Target should be a string!"
    return word_error_rate(preds=preds, target=target)


def mock_wer_class():
    return mock_word_error_rate


class MockResult:
    def __init__(self):
        self.summary = None
        self.transcript = None

    def __call__(self, summary, transcript):
        self.summary = summary
        self.transcript = transcript
