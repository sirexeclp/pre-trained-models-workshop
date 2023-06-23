"""This is our fastapi backend."""
import os

import transformers
import whisper
from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from .load_audio import load_audio
from pydantic import BaseModel

app = FastAPI()


class Result(BaseModel):
    summary: str
    transcript: str


@app.post("/summarize")
async def summarize(
    file: UploadFile,
    model_size: str = Form(),
    min_length: int = Form(),
    max_length: int = Form(),
):
    """Summarize the uploaded file."""
    audio = load_audio(file)
    model = whisper.load_model(
        model_size,
        download_root=os.environ.get("TRANSFORMERS_CACHE")
    )
    transcript = model.transcribe(audio)["text"]
    summarizer = transformers.pipeline(
        "summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary"
    )
    summary = summarizer(transcript, min_length=min_length, max_length=max_length)[0][
        "summary_text"
    ]
    return Result(summary=summary, transcript=transcript)


# serve static html, css and js files
app.mount("/", StaticFiles(directory="condensor/static", html=True), name="static")
