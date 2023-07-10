# KISZ CONDENSOR (Meeting Summarizer) Workshop

NOTE: All commands listed in this readme assume your working directory is the project root.

## Part 1: Benchmarks

### Step 1: Write the benchmark

All files for the benchmarks are contained in the [benchmarks](benchmarks) directory.

To get everything working, you only need to make edits in  [whisper_benchmarks.py](benchmarks/whisper_benchmark.py).
All lines of interest in this file are marked with a `#TODO` comment.

### Step 2: Run some tests

You can run the tests located in [tests/test_benchmarks.py](tests/test_benchmarks.py) with the following command:

~~~bash
pytest
~~~

### Step 3: Run the benchmark

Once all the tests are green, you should be good to go.
Now you can run the benchmarks by invoking:

~~~bash
python3 -m benchmarks
~~~

This should take 2-3 minutes.

### Step 4: Visualize the results
Once the benchmarks are done, you can visualize the results.

Open the notebook in [benchmark_analysis.ipynb](benchmarks/benchmark_analysis.ipynb) and execute all cells.

You should now see some plots with your results.


## Part 2: Demo

> Make sure to checkout the `solutions/benchmark` branch before you continue.
> ~~~bash
> git stash
> git checkout solutions/benchmark
> ~~~
>

The KISZ CONDENSOR is a web based tool that can summarize audio files in text form.
It uses OpenAI Whisper to transcribe the audio and the Huggingface `summary` pipeline with `long-t5` to create a summary.
The frontend is implemented in HTML5 with bootstrap CSS and jquery.
The backend uses fastapi.
Everything is orchestrated using docker compose.

All files for this part of the exercise can be found in [condensor](condensor).

## Step 1: Write the backend

Open [app.py](condensor/app.py).
In this file you will find the `summarize` function, which handles the POST requests made from the frontend.
As in the previous exercise, your task is to fill in the missing TODOs.

### Step 2: Run some tests

You can run the tests located in [tests/test_app.py](tests/test_app.py) with the following command:

~~~bash
pytest
~~~

### Step 3: Run the benchmark

Once all the tests are green, you should be good to go.
Now you can run the app by invoking:

~~~bash
uvicorn condensor.app:app
~~~

This should start up a web-server on port 8000.
(If you are working on one of the vms, VS-Code should automatically forward this port to your local machine.)

### Step 4: Test the app

Now you can open your browser and navigate to: [http://localhost:8000](http://localhost:8000)

You can test the app, by using the example file provided in [benchmarks/examples/10-min-talk.mp3](benchmarks/examples/10-min-talk.mp3).

## Part 3: Docker

We have preinstalled docker, docker compose and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on the laptops and vms.
So for now we will just focus on the configuration and usage of docker and docker compose.

### Step 1: Fix the docker compose file

We have provided a complete docker-file for you, but the [docker-compose.yaml](docker-compose.yaml) 
is missing a few important lines.

### Step 2: Test the docker container

Use the following command to build or pull all necessary containers and start them:

~~~bash
docker compose up --always-recreate-deps --build
~~~

If you exposed the correct port, you should now be able to access the app on port 8000, as before:
[http://localhost:8000](http://localhost:8000)

Make sure to shut down your local development server first, to free the port for use with docker.

### Step 3: Double check for GPU usage

If you did not configure the GPU everything will still work, except much slower.
To check that the GPU is in fact used in your container, you can run the following command:

~~~bash
nvidia-smi
~~~

or 

~~~bash
watch -n 1 nvidia-smi
~~~

to continuously monitor the GPU.

Alternatively you can also use

~~~bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format csv --loop=1
~~~

which will just print GPU utilization and memory usage.

Now, upload a file and keep a look on  Memory-Usage and GPU-Util.
These will tell you if something (your app) is using the GPU.
If nothing happens, double check the configuration in your docker compose file.

### Step 4: Configure GPU usage

As you should have seen in Step 3, docker does not automatically pass a GPU to your container. You need to tell docker to use the nvidia runtime and which gpus you want to pass to the container.

Go to https://docs.docker.com/compose/gpu-support/ to learn how to pass a GPU to a docker container in docker compose.

Repeat steps 2 and 3 to restart the container and check that it now actually used the GPU.

~~~bash
docker compose up --always-recreate-deps --build
~~~

### Step 5: Run multiple containers

> Before you continue checkout the solutions/docker branch
> ~~~bash
> git stash
> git checkout solutions/docker
> ~~~

Alternatively make sure you have specified a port range in your docker-compose file, such as:

~~~yaml
ports:
  - "8000-8010:8000"
~~~

You can start multiple containers with 

~~~bash
docker compose up --scale app=3
~~~

and access them at port 8000, 8001, ...

- [http://localhost:8000](http://localhost:8000)
- [http://localhost:8001](http://localhost:8001)
- [http://localhost:8002](http://localhost:8002)
