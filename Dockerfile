FROM continuumio/miniconda3

WORKDIR /app
COPY . .
RUN conda install python=3.10
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install -r requirements.txt
RUN python ./scripts/docker_build_script.py

EXPOSE 7860