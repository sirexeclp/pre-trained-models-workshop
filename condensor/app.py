"""This is our fastapi backend."""
import base64

from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from .condense import transcribe_and_summarize

app = FastAPI()

class Summary(BaseModel):
    summary: str

@app.post("/upload")
async def create_upload_file(
    file: UploadFile,
    model_size: str = Form(),
    min_length: int = Form(),
    max_length: int = Form(),
):
    """File upload handler."""
    with file.file as the_actual_file:
        result = transcribe_and_summarize(
            base64.b64encode(the_actual_file.read()).decode("UTF-8"),
            model_size,
            min_length,
            max_length,
        ).delay()
    summary = result.wait()
    return # TODO: return a Summary object to the client


# serve static html, css and js files
app.mount("/", StaticFiles(directory="condensor/static", html=True), name="static")
