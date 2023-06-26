import transformers
import whisper
from fastapi.testclient import TestClient
from mocks import load_mock_pipeline, mock_load_audio, mock_load_model

from condensor import utils
from condensor.app import app


def test_summarize_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(whisper, "load_model", mock_load_model)
    monkeypatch.setattr(transformers, "pipeline", load_mock_pipeline)
    monkeypatch.setattr(utils, "load_audio", mock_load_audio)

    tmp_file = tmp_path / "dummy.mp3"
    tmp_file.touch()

    client = TestClient(app)
    response = client.post(
        "/summarize",
        files={
            "file": ("dummy.mp3", b""),
        },
        data={
            "whisper_model_size": "medium",
            "min_length": 10,
            "max_length": 20,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"summary": "summary", "transcript": "transcript"}
