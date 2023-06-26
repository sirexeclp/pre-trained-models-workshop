# KISZ CONDENSOR (Meeting Summarizer) Workshop


## Part 1: Benchmarks

### Step 1: Write the benchmark

All files for the benchmarks are contained in the [benchmarks](benchmarks) directory.

To get everything working, you only need to make edits in  [whisper_benchmarks.py](benchmarks/whisper_benchmark.py).
All lines of interest in this file are marked with a `#TODO` comment.

### Step 2: Run some tests

You can run the tests located in [tests/test_benchmarks.py](tests/test_benchmarks.py) with the following command:

~~~bash 
pytest tests/test_benchmarks.py 
~~~

### Step3: Run the benchmark

Once all the tests are green, you should be good to go.
Now you can run the benchmarks by invoking:

~~~bash
python3 -m benchmarks
~~~

This should take 2-3 minutes.

### Step4: Visualize the results
Once the benchmarks are done, you can visualize the results.

Open the notebook in [benchmark_analysis.ipynb](benchmarks/benchmark_analysis.ipynb) and execute all cells.

You should now see some plots with your results.


## Part 2: Demo

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