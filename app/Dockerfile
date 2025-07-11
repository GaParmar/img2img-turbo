# Dockerfile-gpu
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

# Fix protobuf conflict with ONNX
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python

# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY ./requirements.txt ./

# Install production dependencies.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y libgomp1
RUN apt-get install -y libusb-1.0-0
RUN apt-get install -y libgl1-mesa-glx

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./app ./
COPY ./src ./src

ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY
ENV NCCL_P2P_DISABLE=1
ENTRYPOINT ["python", "main.py"]