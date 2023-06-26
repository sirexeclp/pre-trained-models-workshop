"""This is our fastapi backend."""
import os

import transformers
import whisper
from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import utils

app = FastAPI()


class Result(BaseModel):
    summary: str
    transcript: str


@app.post("/summarize")
async def summarize(
    file: UploadFile,
    whisper_model_size: str = Form(),
    min_length: int = Form(),
    max_length: int = Form(),
):
    """Summarize the uploaded file.

    1. use the load_audio function to load the audio file
    2. load the requested model_size of whisper
    3. transcribe the audio using whisper
    4. load a summarization pipeline
    4. summarize the transcript
    5. return the Result object back to the client
    """
    audio = utils.load_audio(file)
    model = None  # TODO: load the requested size of whisper
    transcript = model.transcribe(audio)["text"]
    summarizer = transformers.pipeline(
        "summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary"
    )
    summary = None  # TODO: use the summarizer to summarize the transcript
    # Hint: infos about the summarization pipeline can be found here:
    # https://huggingface.co/docs/transformers/tasks/summarization
    return Result(summary=summary, transcript=transcript)


# serve static html, css and js files
app.mount("/", StaticFiles(directory="condensor/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0")
