FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling tracking packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source directories into working folder bounds
COPY src/ /app/src/

EXPOSE 8000

ENV PYTHONPATH="/app"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]