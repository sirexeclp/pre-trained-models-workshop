# KISZ CONDENSOR (Meeting Summarizer) DEMO

The KISZ CONDENSOR is a web based tool that can summarize audio files in text form.
It uses OpenAI Whisper to transcribe the audio and the Huggingface `summary` pipeline with `long-t5` to create a summary.
The frontend is implemented in HTML5 with bootstrap css and jquery.
The backend uses fastapi and celery with the rabitmq task queue.
Everythin is orchestrated using docker-compose.

## Getting started

### Docker Compose

The easiest way to run the app is using docker-compose.

You need docker, docker-compose and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed.

Build/pull all necessary containers and start them with

~~~bash
docker-compose up --always-recreate-deps --build   
~~~

### Bare Metal

You can also run the app without docker-compose.
However it is still recommended to run rabitmq in a docker-container.
If you want to use a GPU, make sure to have NVIDIA drivers and CUDA installed.

1. Install the python requirements: `pip install -r requirements.txt`
2. Start rabiitmq `docker run -d -p 5672:5672 rabbitmq`
3. Start the celery worker `MQ_HOST=localhost celery -A transcribe worker --loglevel=INFO`
4. Start the fastapi server `uvicorn condensor.app:app --reload`