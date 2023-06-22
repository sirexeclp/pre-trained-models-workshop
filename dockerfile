FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
RUN apt update
WORKDIR /app
ADD requirements.txt .
# TODO: install pip requirements
# TODO: add condensor directory
CMD [ "uvicorn", "--host", "0.0.0.0", "--workers", "8", "condensor.app:app" ]