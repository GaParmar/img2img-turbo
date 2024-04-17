FROM continuumio/miniconda3

WORKDIR /app
COPY . .
RUN conda env create -f environment.yaml
EXPOSE 7860