FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt /app/requirements.txt

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir -r /app/requirements.txt

# Do NOT copy app files — we mount them
EXPOSE 8000
CMD ["uvicorn","service.app:app","--host","0.0.0.0","--port","8000"]
