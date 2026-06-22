
#!/bin/bash

# ==========================================
# Configuration Section
# ==========================================

# 1. OpenAI API Key
# The key is read from the git-ignored .env file by llmcer/config.py, so it is
# NOT hard-coded here (committing it would leak the secret). To override for a
# single run you may still `export OPENAI_API_KEY=sk-...` before calling run.sh.
# Only export it here if it is non-empty, so we never clobber the .env value
# with an empty string.
# export OPENAI_API_KEY="sk-..."

# 2. Dataset Paths
# Relative paths from project root
export DATASET_PATH="datasets/cora/cora.csv"
export GROUND_TRUTH_PATH="datasets/cora/gt.csv"

# 3. Model Configuration
# Path to local embedding model or Hugging Face model name
# As per request: "all-MiniLM-L6-v2" folder is at the same level as "llmcer"
export EMBEDDING_MODEL_PATH="all-MiniLM-L6-v2"
export OPENAI_MODEL="gpt-4o-mini"

# ==========================================
# Execution Section
# ==========================================

# Install dependencies if not already installed (optional check)
# pip install -r requirements.txt

echo "Starting LLMCER Pipeline..."
echo "Dataset: $DATASET_PATH"
echo "Ground Truth: $GROUND_TRUTH_PATH"
echo "Embedding Model: $EMBEDDING_MODEL_PATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Extract dataset name (filename without extension)
DATASET_NAME=$(basename "$DATASET_PATH" | cut -d. -f1)

# Generate timestamped log filename with dataset name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${DATASET_NAME}_run_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"

# Run the pipeline script and pipe output to both console and log file
# using unbuffered output for python (-u) to ensure logs appear in real-time
python -u scripts/run_pipeline.py | tee "$LOG_FILE"
