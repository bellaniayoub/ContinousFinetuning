# Start with the official Airflow image
FROM apache/airflow:2.9.1-python3.10

USER root
# Install git (needed for some pip packages)
# Install netcat (needed for docker-compose healthcheck/wait-for-it)
# Install docker.io (needed to control vllm container)
RUN apt-get update && apt-get install -y git netcat-openbsd docker.io && apt-get clean

USER airflow
# Install your Training & Deployment dependencies
# We do NOT use a venv here; we install directly into the container's python
RUN pip install --no-cache-dir \
    torch \
    transformers \
    peft \
    trl \
    bitsandbytes \
    accelerate \
    pandas \
    datasets \
    requests \
    vllm==0.5.0 \
    evaluate \
    bert_score \
    absl-py \
    rouge_score