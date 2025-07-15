# Post Similarity & Tagging Microservice

## Overview
A FastAPI microservice for duplicate post detection and auto-tagging using NLP models. Fully offline, configurable, and Docker-ready. Duplicate detection is powered by Sentence Transformers for semantic embeddings and FAISS for fast similarity search, while tag generation uses KeyBERT for extracting relevant keywords.

## Features
- Duplicate detection using sentence embeddings (Sentence Transformers) + FAISS for fast similarity search
- Auto-tag generation using KeyBERT
- Configurable thresholds and tag rules
- REST API with FastAPI
- Docker support

## Requirements
- Python 3.10+
- See requirements.txt for Python dependencies (now includes pydantic)

## Installation
sh
pip install -r requirements.txt


## Running the API (Locally)
sh
uvicorn main:app --reload --port 8000

- The FastAPI app is defined in main.py as app.
- By default, this binds to 127.0.0.1 (localhost). For Docker, see below.

## API Endpoints

### 1. Check Duplicate
- *POST /check-duplicate*
- *Input:*
json
{
  "user_id": "user_1234",
  "title": "Binary search trees explained",
  "body": "Can someone explain how binary search trees work and their use cases?"
}

- *Response:*
json
{
  "is_duplicate": true,
  "similar_post_id": "1",
  "similarity_score": 1.0,
  "status": "blocked"
}


### 2. Generate Tags
- *POST /generate-tags*
- *Input:*
json
{
  "post_id": "1",
  "post_body": "Can someone explain how binary search trees work and their use cases?"
}

- *Response:*
json
{
  "post_id": "1",
  "tags": ["binary search tree", "data structures", "algorithms"]
}


### 3. Health Check
- *GET /health*
- *Response:* { "status": "ok" }

### 4. Version
- *GET /version*
- *Response:* { "model_version": "1.0", "config_version": "1.0" }

## Docker Usage

### 1. Build the Docker image
sh
docker build -t post-tagger .


### 2. Run the container
sh
docker run -p 8000:8000 post-tagger

- The Dockerfile uses uvicorn main:app --host 0.0.0.0 --port 8000 so the API is accessible from outside the container.
- If you use 127.0.0.1, the API will not be accessible from your host machine.

## Dockerfile
A sample Dockerfile is included:
dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


## Notes
- All models and configs are local; no external API calls.
- See config.json for threshold and tag settings.
- Dataset: posts_dataset.json (200+ posts, tags, duplicates)
