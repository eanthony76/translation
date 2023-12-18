# Start with a base image that includes CUDA, for example, the official CUDA 11.2 runtime Ubuntu image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set a label to describe the Dockerfile
LABEL maintainer="eanthony"

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    python3 \
    pip

#Copy requirements
COPY requirements.txt requirements.txt

#Install requirements
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt

COPY nllb-translation /home/nllb-translation

WORKDIR /home/nllb-translation

EXPOSE 7860

CMD ["python3", "-u", "app.py"]