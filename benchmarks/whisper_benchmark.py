"""Some simple benchmarks for the whisper model."""
import time
from enum import Enum
from pathlib import Path
from typing import Tuple

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
    model_size: ModelSize, input_file: Path, reference_file: Path
) -> Tuple[float, float, float]:
    """Benchmark the given model size, by measuring time, energy, and WER."""
    model = whisper.load_model(model_size.value)
    gpu_energy = utils.GPUEnergyMeter()

    start = None  # TODO: save the timestamp of the start of this benchmark
    with gpu_energy:
        prediction = model.transcribe(str(input_file))
    end = None  # TODO: save the timestamp of the bend of this benchmark
    runtime = None  # TODO: compute the runtime

    target = reference_file.read_text()
    wer = None  # TODO: insert function to compute word error rate
    # Hint: maybe you can find a library function
    # torch metrics: https://torchmetrics.readthedocs.io/en/stable/
    return (
        wer,
        runtime,
        gpu_energy.energy,
    )
