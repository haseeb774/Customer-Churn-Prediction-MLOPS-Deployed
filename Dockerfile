FROM python:3.9-slim

WORKDIR /app

# Use a faster mirror and only install the bare minimum for OpenCV/Images
RUN sed -i 's/deb.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh
EXPOSE 8501 8000

CMD ["./entrypoint.sh"]