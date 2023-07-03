"""This is our fastapi backend."""
import os

import transformers
import whisper
from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import utils

app = FastAPI()

import transformers


class Result(BaseModel):
    summary: str
    transcript: str


@app.post("/summarize")
async def summarize(
    file: UploadFile,
    whisper_model_size: str = Form(),
    min_summary_length: int = Form(),
    max_summary_length: int = Form(),
) -> Result:
    """Summarize the uploaded file.

    1. use the load_audio function to load the audio file
    2. load the requested model_size of whisper
    3. transcribe the audio using whisper
    4. load a summarization pipeline
    4. summarize the transcript (make sure to set min and max length)
    5. return the Result object back to the client
    """
    audio = utils.load_audio(file)
    model = whisper.load_model(whisper_model_size)
    transcript = model.transcribe(audio)["text"]

    summarizer = transformers.pipeline(
        "summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary"
    )

    summary = summarizer(
        transcript, min_length=min_summary_length, max_length=max_summary_length
    )[0]["summary_text"]
    return Result(summary=summary, transcript=transcript)


# serve static html, css and js files
app.mount("/", StaticFiles(directory="condensor/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0")
