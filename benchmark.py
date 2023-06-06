from pathlib import Path
from torchmetrics.functional import word_error_rate
from enum import Enum
import whisper
import time
import pandas as pd

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
input_file = "examples/10-min-talk.mp3"


class ModelSize(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"


# def batch_time_benchmark(batch_size: int, model_size: ModelSize) -> float:
#     model = whisper.load_model(model_size.value)
#     start = time.time()
#     results = model.transcribe([input_file]*batch_size)
#     end = time.time()
#     return end - start


# def benchmark_batch_sizes():
#     results = []
#     for model_size in ModelSize:
#         for batch_size in batch_sizes:
#             inference_time = batch_time_benchmark(batch_size, model_size)
#             result = dict(
#                     model_size=model_size.value,
#                     batch_size=batch_size,
#                     time=inference_time,
#                 )
#             print(result)
#             results.append(result)
#     pd.DataFrame(results).to_csv("batch_sizes_benchmark.csv",index=False)


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
