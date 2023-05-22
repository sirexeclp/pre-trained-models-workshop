import base64
import os
from celery import Celery, chain
from fastapi import UploadFile
import whisper
from whisper.audio import SAMPLE_RATE
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import ffmpeg
from transformers import pipeline


app = Celery("tasks", backend="rpc://rabbitmq", broker="pyamqp://rabbitmq")


def load_audio(file: str, sr: int = SAMPLE_RATE):
    try:
        out, _ = (
            ffmpeg.input("pipe:", threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(
                cmd=["ffmpeg", "-nostdin"],
                input=base64.b64decode(file),
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@app.task
def transcribe(audio_file: UploadFile, model_size: str):
    model = whisper.load_model(
        model_size, download_root=os.environ.get("TRANSFORMERS_CACHE")
    )
    audio = load_audio(audio_file)
    result = model.transcribe(audio)
    return result["text"]
    # text = result["text"]
    # print(text)


@app.task
def summarize(text: str, max_length=130, min_length=30):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text)[0]["summary_text"]


@app.task
def transcribe_and_summarize(audio_file, model_size: str):
    return (transcribe.s(audio_file, model_size) | summarize.s())()
