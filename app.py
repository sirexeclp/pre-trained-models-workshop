import base64
import whisper
from whisper.audio import SAMPLE_RATE

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles

from transcribe import transcribe

app = FastAPI()


@app.post("/upload")
async def create_upload_file(file: UploadFile, model_size: str = Form()):
    with file.file as the_actual_file:
        result = transcribe.delay(base64.b64encode(the_actual_file.read()).decode("UTF-8"), model_size)
    return result.wait()


app.mount("/", StaticFiles(directory="static", html=True), name="static")
