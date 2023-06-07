"""This module implements the transcribe and summarization logic."""
import base64
import os

import ffmpeg
import numpy as np
import whisper
from celery import Celery
from transformers import pipeline
from whisper.audio import SAMPLE_RATE

mq_host = os.getenv("MQ_HOST")

app = Celery("tasks", backend=f"rpc://{mq_host}", broker=f"pyamqp://{mq_host}")


def audio_from_base64(base64_str: str, sample_rate: int = SAMPLE_RATE):
    """Load audio from a base64 string and return numpy array."""
    try:
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
            .run(
                cmd=["ffmpeg", "-nostdin"],
                input=base64.b64decode(base64_str),
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as error:
        raise RuntimeError(f"Failed to load audio: {error.stderr.decode()}") from error

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@app.task
def transcribe(base64_audio: str, model_size: str):
    """Return a transcription of the audio, generated with whisper."""
    model = whisper.load_model(
        model_size, download_root=os.environ.get("TRANSFORMERS_CACHE")
    )
    audio = audio_from_base64(base64_audio)
    return model.transcribe(audio)["text"]


@app.task
def summarize(text: str, min_length=30, max_length=100):
    """Return a summary of the input text."""
    summarizer = pipeline(
        "summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary"
    )
    return summarizer(text, min_length=min_length, max_length=max_length)[0][
        "summary_text"
    ]

def transcribe_and_summarize(base64_audio: str, model_size: str, min_length: int, max_length:int):
    return transcribe.s(base64_audio, model_size) | summarize.s(min_length=min_length, max_length=max_length)
