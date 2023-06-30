import asyncio

import pytest
import transformers
import whisper
from fastapi.testclient import TestClient
from mocks import MockModelLoader, MockPipelineLoader, MockResult, mock_load_audio

import condensor.app
from condensor import utils
from condensor.app import app, summarize


def apply_pipeline_patch(monkeypatch, pipeline_loader):
    monkeypatch.setattr(transformers, "pipeline", pipeline_loader)
    try:
        monkeypatch.setattr(condensor.app, "pipeline", pipeline_loader)
    except:
        pass


def test_summarize_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(whisper, "load_model", MockModelLoader(return_strategy="dict"))
    apply_pipeline_patch(monkeypatch, MockPipelineLoader())
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
            "min_summary_length": 10,
            "max_summary_length": 20,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"summary": "summary", "transcript": "transcript"}


@pytest.mark.parametrize("model_size", ["tiny", "base", "small", "medium", "large"])
def test_model_loading(monkeypatch, model_size):
    model_loader = MockModelLoader()
    pipeline_loader = MockPipelineLoader()
    monkeypatch.setattr(whisper, "load_model", model_loader)
    apply_pipeline_patch(monkeypatch, pipeline_loader)
    monkeypatch.setattr(utils, "load_audio", mock_load_audio)

    asyncio.run(summarize(None, model_size, 10, 15))
    assert model_loader.model_size == model_size, "Incorrect model size was used!"


def test_pipeline_loading(monkeypatch):
    model_size = "base"
    model_loader = MockModelLoader()
    pipeline_loader = MockPipelineLoader()
    monkeypatch.setattr(whisper, "load_model", model_loader)
    apply_pipeline_patch(monkeypatch, pipeline_loader)
    monkeypatch.setattr(utils, "load_audio", mock_load_audio)

    asyncio.run(summarize(None, model_size, 10, 15))

    assert (
        pipeline_loader.pipeline.name == "summarization"
    ), "Incorrect pipeline was used!"
    assert (
        pipeline_loader.pipeline.model
        == "pszemraj/long-t5-tglobal-base-16384-book-summary"
    ), "Incorrect model was used!"


def test_pipeline_execution(monkeypatch):
    model_size = "base"
    model_loader = MockModelLoader(return_strategy="dict")
    pipeline_loader = MockPipelineLoader()
    monkeypatch.setattr(whisper, "load_model", model_loader)
    apply_pipeline_patch(monkeypatch, pipeline_loader)
    monkeypatch.setattr(utils, "load_audio", mock_load_audio)

    asyncio.run(summarize(None, model_size, 10, 15))

    assert isinstance(
        pipeline_loader.pipeline.inputs, str
    ), "Incorrect input type to summarization pipeline! Expected string!"
    assert (
        pipeline_loader.pipeline.inputs == "transcript"
    ), "Incorrect input to summarization pipeline!"


def test_pipeline_length(monkeypatch):
    model_size = "base"
    min_length = 69
    max_length = 420
    model_loader = MockModelLoader(return_strategy="dict")
    pipeline_loader = MockPipelineLoader()
    monkeypatch.setattr(whisper, "load_model", model_loader)
    apply_pipeline_patch(monkeypatch, pipeline_loader)
    monkeypatch.setattr(utils, "load_audio", mock_load_audio)

    asyncio.run(summarize(None, model_size, min_length, max_length))

    assert (
        pipeline_loader.pipeline.min_length == min_length
    ), "Minimum summary length was not set to the correct value!"
    assert (
        pipeline_loader.pipeline.max_length == max_length
    ), "Maximum summary length was not set to the correct value!"


def test_pipeline_output(monkeypatch):
    model_size = "base"
    model_loader = MockModelLoader(return_strategy="dict")
    pipeline_loader = MockPipelineLoader()
    monkeypatch.setattr(whisper, "load_model", model_loader)
    apply_pipeline_patch(monkeypatch, pipeline_loader)
    monkeypatch.setattr(utils, "load_audio", mock_load_audio)
    mock_result = MockResult()
    monkeypatch.setattr(condensor.app, "Result", mock_result)

    asyncio.run(summarize(None, model_size, 10, 15))

    assert isinstance(mock_result.transcript, str), "Transcript should be a string!"
    assert mock_result.transcript == "transcript", "Unexpected value for transcript!"

    assert isinstance(mock_result.summary, str), "Summary should be a string!"
    assert mock_result.summary == "summary", "Unexpected value for summary!"
