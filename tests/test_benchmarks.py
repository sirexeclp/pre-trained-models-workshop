from pathlib import Path

import torchmetrics
import whisper

import benchmarks.whisper_benchmark
from benchmarks import utils
from benchmarks.whisper_benchmark import ModelSize, wer_benchmark

from .mocks import (
    MockGPUEnergyMeter,
    MockModelLoader,
    mock_wer_class,
    mock_word_error_rate,
)


def patch_wer(monkeypatch):
    """Try to patch all possible word error rate functions and import locations."""
    try:
        monkeypatch.setattr(
            torchmetrics.functional, "word_error_rate", mock_word_error_rate
        )
    except:
        pass
    try:
        monkeypatch.setattr(
            benchmarks.whisper_benchmark, "word_error_rate", mock_word_error_rate
        )
    except:
        pass
    try:
        monkeypatch.setattr(torchmetrics, "WordErrorRate", mock_wer_class)
    except:
        pass
    try:
        monkeypatch.setattr(
            benchmarks.whisper_benchmark, "WordErrorRate", mock_wer_class
        )
    except:
        pass


def test_wer_tiny(monkeypatch):
    monkeypatch.setattr(whisper, "load_model", MockModelLoader(return_strategy="empty"))
    monkeypatch.setattr(utils, "GPUEnergyMeter", MockGPUEnergyMeter)
    patch_wer(monkeypatch)

    wer, runtime, energy = wer_benchmark(
        ModelSize.TINY,
        Path("benchmarks/examples/10-min-talk.mp3"),
        Path("benchmarks/examples/10-min-talk-reference.txt"),
    )
    assert wer is not None, "Word Error Rate need's to be implemented!"
    assert isinstance(
        wer, float
    ), f"Word Error Rate must be of type Float but was {type(wer)}!"

    # the mocked tiny model will have a wer of 1
    assert wer == 1, "Word Error Rate is incorrect!"


def test_wer_base(monkeypatch):
    monkeypatch.setattr(
        whisper, "load_model", MockModelLoader(return_strategy="reference")
    )
    monkeypatch.setattr(utils, "GPUEnergyMeter", MockGPUEnergyMeter)
    patch_wer(monkeypatch)
    wer, runtime, energy = wer_benchmark(
        ModelSize.BASE,
        Path("benchmarks/examples/10-min-talk.mp3"),
        Path("benchmarks/examples/10-min-talk-reference.txt"),
    )
    assert wer is not None, "Word Error Rate need's to be implemented!"
    assert isinstance(
        wer, float
    ), f"Word Error Rate must be of type Float but was {type(wer)}!"
    # the mocked base model will have a wer of 0
    assert wer == 0, "Word Error Rate is incorrect!"


def test_runtime(monkeypatch):
    monkeypatch.setattr(whisper, "load_model", MockModelLoader(return_strategy="empty"))
    monkeypatch.setattr(utils, "GPUEnergyMeter", MockGPUEnergyMeter)
    patch_wer(monkeypatch)
    wer, runtime, energy = wer_benchmark(
        ModelSize.TINY,
        Path("benchmarks/examples/10-min-talk.mp3"),
        Path("benchmarks/examples/10-min-talk-reference.txt"),
    )
    assert runtime is not None, "Runtime measurement needs to be implemented!"
    assert (
        0.1 < runtime < 0.2
    ), f"Runtime measurement is incorrect! Expected t between 0.1s and 0.2s but got {runtime}."


def test_energy(monkeypatch):
    monkeypatch.setattr(whisper, "load_model", MockModelLoader(return_strategy="empty"))
    monkeypatch.setattr(utils, "GPUEnergyMeter", MockGPUEnergyMeter)
    patch_wer(monkeypatch)
    wer, runtime, energy = wer_benchmark(
        ModelSize.TINY,
        Path("benchmarks/examples/10-min-talk.mp3"),
        Path("benchmarks/examples/10-min-talk-reference.txt"),
    )
    assert (
        energy == 1_000
    ), "Energy measurement is incorrect! Should be 1000!"  # this should pass by default
