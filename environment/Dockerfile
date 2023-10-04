# hash:sha256:b920a516e14ef6e4b0e0d8b0e2556bec0b081d66cc395de699a05c3061636dd1
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    matplotlib==3.8.0 \
    numpy==1.26.0 \
    pandas==2.1.1 \
    pyswarms==1.3.0 \
    scikit-learn==1.3.1 \
    scipy==1.11.3 \
    tensorflow==2.14.0
