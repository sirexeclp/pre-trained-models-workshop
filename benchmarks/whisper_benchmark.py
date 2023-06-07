from pathlib import Path
from torchmetrics.functional import word_error_rate
from enum import Enum
import whisper
import time
import pandas as pd

input_file = "examples/10-min-talk.mp3"


class ModelSize(Enum):
    """This contains all of the available model sizes of whisper."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"

def wer_benchmark(model_size: ModelSize) -> float:
    model = whisper.load_model(model_size.value)
    start = time.time()
    prediction = model.transcribe(input_file)
    end = time.time()
    target = Path("examples/10-min-talk-reference.txt").read_text()
    return word_error_rate(preds=prediction["text"], target=target).item(), end - start


def benchmark_wer_model_sizes():
    results = []
    for model_size in ModelSize:
        wer, inference_time = wer_benchmark(model_size)
        result = dict(model_size=model_size.value, time=inference_time, wer=wer)
        print(result)
        results.append(result)
    pd.DataFrame(results).to_csv("model_sizes_benchmark.csv", index=False)


if __name__ == "__main__":
    benchmark_wer_model_sizes()
