"""This is our fastapi backend."""
import base64

from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles

from .condense import transcribe_and_summarize

app = FastAPI()


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
    return {"text": result.wait()}


# serve static html, css and js files
app.mount("/", StaticFiles(directory="condensor/static", html=True), name="static")
