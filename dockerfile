FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt update
WORKDIR /app
ADD requirements.txt .
RUN pip3 install -r requirements.txt
ADD condensor condensor
CMD [ "uvicorn", "--host", "0.0.0.0", "--workers", "8", "condensor.app:app" ]