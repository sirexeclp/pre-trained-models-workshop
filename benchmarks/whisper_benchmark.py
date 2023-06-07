"""Some simple benchmarks for the whisper model."""
import time
from enum import Enum
from pathlib import Path

import pandas as pd
import whisper
from pynvml3.pynvml import NVMLLib
from torchmetrics.functional import word_error_rate


class ModelSize(Enum):
    """This contains all of the available model sizes of whisper."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"


class Stopwatch:
    """A context manager based stopwatch."""

    def __init__(self) -> None:
        self.start = None
        self.end = None

    @property
    def duration(self) -> float:
        """Return the elapsed duration in seconds."""
        return self.end - self.start

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()


class GPUEnergyMeter:
    """A context manager based GPU energy meter."""

    def __init__(self, device_index=0) -> None:
        self.start = None
        self.end = None
        self.lib = NVMLLib()
        self.device_index = device_index
        self.device = None

    @property
    def energy(self) -> float:
        """Return the total amount of consumed energy in J."""
        return (self.end - self.start) / 1_000

    def __enter__(self):
        self.lib.open()
        self.device = self.lib.device[self.device_index]
        self.start = self.device.get_total_energy_consumption()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = self.device.get_total_energy_consumption()
        self.lib.close()


def wer_benchmark(
    model_size: ModelSize, input_file: Path, reference_file: Path
) -> float:
    """Benchmark the given model size, by measuring time, energy, and WER."""
    model = whisper.load_model(model_size.value)
    with GPUEnergyMeter() as energy_meter:
        with Stopwatch() as stopwatch:
            prediction = model.transcribe(str(input_file))
    target = reference_file.read_text()
    return (
        word_error_rate(preds=prediction["text"], target=target).item(),
        stopwatch.duration,
        energy_meter.energy,
    )


def benchmark_wer_model_sizes(
    input_file: Path, reference_file: Path, output_file: Path
):
    """Run the wer vs model size benchmark on all available sizes."""
    results = []
    for model_size in ModelSize:
        wer, inference_time, energy = wer_benchmark(
            model_size, input_file, reference_file
        )
        result = {
            "model_size": model_size.value,
            "time": inference_time,
            "wer": wer,
            "energy": energy,
        }
        print(result)
        results.append(result)
    pd.DataFrame(results).to_csv(output_file, index=False)

def get_gpu_name() -> str:
    """Return the name of the first GPU."""
    with NVMLLib() as lib:
        return lib.device[0].get_name()


def main():
    """The main function."""
    examples_path = Path("benchmarks", "examples")
    results_path = Path("benchmarks", "results")
    input_file = examples_path / "10-min-talk.mp3"
    reference_file = examples_path / "10-min-talk-reference.txt"
    results_file = results_path / f"{get_gpu_name()}.csv"
    benchmark_wer_model_sizes(input_file, reference_file, results_file)


if __name__ == "__main__":
    main()
