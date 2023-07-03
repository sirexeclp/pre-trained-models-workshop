"""Some simple benchmarks for the whisper model."""
import time
from enum import Enum
from pathlib import Path
from typing import Tuple

import torchmetrics.functional
import whisper

from . import utils


class ModelSize(Enum):
    """This contains all of the available model sizes of whisper."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"

    # We will skip the larger models, to save some time.
    # If you still have time left, you can uncomment these
    # and include them in your benchmark.

    # MEDIUM = "medium"
    # LARGE = "large"


def wer_benchmark(
    model_size: ModelSize, input_file: str, reference_file: Path
) -> Tuple[float, float, float]:
    """Benchmark the given model size, by measuring time, energy, and WER."""
    model = whisper.load_model(model_size.value)
    gpu_energy = utils.GPUEnergyMeter()

    start = time.time()
    with gpu_energy:
        prediction = model.transcribe(input_file)["text"]
    end = time.time()
    runtime = end - start
    target = reference_file.read_text()
    wer = torchmetrics.functional.word_error_rate(
        preds=prediction, target=target
    ).item()
    return (
        wer,
        runtime,
        gpu_energy.energy,
    )
