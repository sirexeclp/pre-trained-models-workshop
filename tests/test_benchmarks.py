from pathlib import Path

import whisper
from mocks import MockGPUEnergyMeter, mock_load_model

from benchmarks import utils
from benchmarks.whisper_benchmark import ModelSize, wer_benchmark


def test_wer_tiny(monkeypatch):
    monkeypatch.setattr(whisper, "load_model", mock_load_model)
    monkeypatch.setattr(utils, "GPUEnergyMeter", MockGPUEnergyMeter)
    wer, runtime, energy = wer_benchmark(
        ModelSize.TINY,
        Path("benchmarks/examples/10-min-talk.mp3"),
        Path("benchmarks/examples/10-min-talk-reference.txt"),
    )
    assert wer is not None, "Word Error Rate need's to be implemented!"
    # the mocked tiny model will have a wer of 1
    assert wer == 1, "Word Error Rate is incorrect!"


def test_wer_base(monkeypatch):
    monkeypatch.setattr(whisper, "load_model", mock_load_model)
    monkeypatch.setattr(utils, "GPUEnergyMeter", MockGPUEnergyMeter)
    wer, runtime, energy = wer_benchmark(
        ModelSize.BASE,
        Path("benchmarks/examples/10-min-talk.mp3"),
        Path("benchmarks/examples/10-min-talk-reference.txt"),
    )
    assert wer is not None, "Word Error Rate need's to be implemented!"
    # the mocked base model will have a wer of 0
    assert wer == 0, "Word Error Rate is incorrect!"


def test_runtime(monkeypatch):
    monkeypatch.setattr(whisper, "load_model", mock_load_model)
    monkeypatch.setattr(utils, "GPUEnergyMeter", MockGPUEnergyMeter)
    wer, runtime, energy = wer_benchmark(
        ModelSize.TINY,
        Path("benchmarks/examples/10-min-talk.mp3"),
        Path("benchmarks/examples/10-min-talk-reference.txt"),
    )
    assert runtime is not None, "Runtime measurement needs to be implemented!"
    assert 0.1 < runtime < 0.2, "Runtime measurement is incorrect!"


def test_energy(monkeypatch):
    monkeypatch.setattr(whisper, "load_model", mock_load_model)
    monkeypatch.setattr(utils, "GPUEnergyMeter", MockGPUEnergyMeter)
    wer, runtime, energy = wer_benchmark(
        ModelSize.TINY,
        Path("benchmarks/examples/10-min-talk.mp3"),
        Path("benchmarks/examples/10-min-talk-reference.txt"),
    )
    assert (
        energy == 1_000
    ), "Energy measurement is incorrect!"  # this should pass by default
