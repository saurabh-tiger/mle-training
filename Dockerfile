FROM continuumio/miniconda3
LABEL maintainer="Saurabh Zinjad"
COPY deploy/conda/env.yml env.yml
RUN conda env create -f env.yml
COPY src src
RUN mkdir artifacts data logs && mkdir data/processed data/raw
SHELL ["conda", "run", "-n", "testcase-dev", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "testcase-dev", "python", "src/main.py"]