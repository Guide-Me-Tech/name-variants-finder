#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure strict shell settings for robust error handling
# ------------------------------------------------------------------------------
# Explanation:
#   - 'set -e' stops the script at the first sign of an error.
#   - 'set -u' treats unset variables as errors.
#   - 'set -o pipefail' ensures that the entire pipeline fails if any command fails.
set -euo pipefail

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Define a trap to log unexpected errors with context
# ------------------------------------------------------------------------------
trap 'echo "[ERROR] A fatal error occurred at line $LINENO. Exiting..." >&2' ERR

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Redirect output (stdout & stderr) to both console and a logfile
# ------------------------------------------------------------------------------
exec > >(tee -i deploy_names.log)
exec 2>&1

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Helper function to measure and log execution time
# ------------------------------------------------------------------------------
timer_start() {
  STEP_START_TIME=$(date +%s)
}

timer_end() {
  local STEP_END_TIME
  STEP_END_TIME=$(date +%s)
  local ELAPSED=$((STEP_END_TIME - STEP_START_TIME))
  echo "Step took ${ELAPSED} seconds."
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Function to optionally log CPU usage via 'ps'
# ------------------------------------------------------------------------------
# Explanation:
#   - This can provide a snapshot of CPU usage after a command finishes.
#   - For deeper insights, consider external tools like 'top', 'htop', or Prometheus exporters.
log_cpu_usage() {
  echo "Current CPU usage by top processes:"
  ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 6
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 6: Execute the Python embedding script
# ------------------------------------------------------------------------------
echo "---- Running embed_milvus.py ----"
timer_start
python embed_milvus.py
timer_end
log_cpu_usage

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 7: Start the FastAPI server
# ------------------------------------------------------------------------------
echo "---- Starting FastAPI server ----"
timer_start
fastapi run /app/server_search_names.py --port 8000
timer_end
log_cpu_usage

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 8: Completion message
# ------------------------------------------------------------------------------
echo "Deployment script completed successfully."
