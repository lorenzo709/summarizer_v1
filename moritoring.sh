#!/bin/bash

TOPIC="Retrieval-Augmented_Generation_for_Legacy_Code_Refactoring"
MODEL="gpt-oss:120b"
# 1. Setup unique log folder
# TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LOG_DIR="logs_${TOPIC}_${MODEL}"
mkdir -p "$LOG_DIR"

# 2. Define the Cleanup Function
cleanup() {
    # We use a broader kill here to ensure the while loops also stop
    echo -e "\n--- Cleaning up all background processes ---"
    jobs -p | xargs kill 2>/dev/null
    echo "Done. Logs are safe in $LOG_DIR"
}

trap cleanup EXIT SIGINT SIGTERM

# 3. Start Monitors (Polling every 60 seconds)

# GPU: Using a loop to poll every 1s
(
  while true; do
    rocm-smi --showpower --showmemuse --csv 2>/dev/null >> "$LOG_DIR/gpu_stats.csv"
    sleep 1
  done
) &

# CPU: top -d 1 sets the delay to 1 seconds
top -b -d 1 | grep --line-buffered "Cpu(s)" >> "$LOG_DIR/cpu_stats.txt" &

# # 4. Run Two Python Pipelines Simultaneously
# echo "Starting pipelines... (Press Ctrl+C to stop manually)"

# python summarize.py
# PY_PID_1=$!

# python SOTA_overview.py
# PY_PID_2=$!

# echo "Pipelines running with PIDs: $PY_PID_1, $PY_PID_2"

python main.py
# PY_PID_1=$!

# # 5. Wait for both Python scripts to finish
# wait $PY_PID_1 $PY_PID_2

echo "FINISHED"
