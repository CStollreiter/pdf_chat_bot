FROM python:3.9-slim

WORKDIR /llm_app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* 

COPY . .

RUN pip3 install -r requirements.txt

ENV OLLAMA_API_BASE_URL='http://host.docker.internal:11434'

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
