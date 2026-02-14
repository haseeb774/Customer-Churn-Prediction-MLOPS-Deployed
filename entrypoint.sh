#!/bin/bash

# Stop execution if any command fails
set -e

echo "--- Step 1: Running Training Pipeline ---"
# This generates model.pkl and your analysis images
python pipeline/run_pipeline.py

echo "--- Step 2: Starting FastAPI (Background) ---"
# We use '&' to run this in the background so the script continues
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Give FastAPI a moment to start up
sleep 5

echo "--- Step 3: Starting Streamlit (Foreground) ---"
# This runs in the foreground and keeps the container alive
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0