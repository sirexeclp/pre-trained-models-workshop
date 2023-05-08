import base64
from celery import Celery
from fastapi import UploadFile
import whisper
from whisper.audio import SAMPLE_RATE
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import ffmpeg


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
    model = whisper.load_model(model_size)
    audio = load_audio(audio_file)
    result = model.transcribe(audio)
    return {"text": result["text"]}
