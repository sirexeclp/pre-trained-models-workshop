"""This is our fastapi backend."""
import ffmpeg
import numpy as np
from fastapi import UploadFile
from whisper.audio import SAMPLE_RATE


def load_audio(uploaded_file: UploadFile, sample_rate: int = SAMPLE_RATE):
    """Load audio from a base64 string and return numpy array."""
    with uploaded_file.file as the_actual_file:
        try:
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
                .run(
                    cmd=["ffmpeg", "-nostdin"],
                    input=the_actual_file.read(),
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
        except ffmpeg.Error as error:
            raise RuntimeError(
                f"Failed to load audio: {error.stderr.decode()}"
            ) from error
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
