from pathlib import Path

import pandas as pd

from benchmarks.utils import get_gpu_name
from benchmarks.whisper_benchmark import ModelSize, wer_benchmark


def benchmark_wer_model_sizes(
    input_file: Path, reference_file: Path, output_file: Path
):
    """Run the wer vs model size benchmark on all available sizes."""
    results = []
    for model_size in ModelSize:
        wer, inference_time, energy = wer_benchmark(
            model_size, str(input_file), reference_file
        )
        result = {
            "model_size": model_size.value,
            "time": inference_time,
            "wer": wer,
            "energy": energy,
        }
        print(result)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def main():
    """The main function."""
    examples_path = Path("benchmarks", "examples")
    results_path = Path("benchmarks", "results", "whisper_benchmark")
    results_path.mkdir(exist_ok=True, parents=True)
    input_file = examples_path / "10-min-talk.mp3"
    reference_file = examples_path / "10-min-talk-reference.txt"
    results_file = results_path / f"{get_gpu_name()}.csv"
    benchmark_wer_model_sizes(input_file, reference_file, results_file)


if __name__ == "__main__":
    main()
