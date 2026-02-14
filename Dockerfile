FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (added curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Remove the HEALTHCHECK from here so it doesn't break MLflow/Streamlit
EXPOSE 8000 8501 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]