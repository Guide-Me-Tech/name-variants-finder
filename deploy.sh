#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure strict shell settings for robust error handling
# ------------------------------------------------------------------------------
# Explanation: 
#   - 'set -e' stops the script on the first error.
#   - 'set -u' treats unset variables as an error.
#   - 'set -o pipefail' makes piped commands fail if any step fails.
set -euo pipefail

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Trap to log any unexpected errors with context
# ------------------------------------------------------------------------------
trap 'echo "[ERROR] A fatal error occurred at line $LINENO. Exiting..." | tee -a deploy.log; exit 1;' ERR

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Redirect output to both console and a log file
# ------------------------------------------------------------------------------
# Explanation:
#   - tee -i deploy.log writes output to both the console and deploy.log.
#   - 2>&1 merges standard error into standard output for unified logging.
exec > >(tee -i deploy.log)
exec 2>&1

# ------------------------------------------------------------------------------
# ### Janis Rubins - Helper function to measure elapsed time for steps
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
# ### Janis Rubins - Step 4: Start the standalone_embed.sh script
# ------------------------------------------------------------------------------
timer_start
echo "---- Starting standalone_embed.sh with 'start' argument ----"
sh standalone_embed.sh start
timer_end

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Build the Docker image
# ------------------------------------------------------------------------------
timer_start
echo "---- Building Docker image 'name_variants_image' using dockerfile.server ----"
docker build --file dockerfile.server -t name_variants_image .
timer_end

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 6: Stop and remove existing container if running
# ------------------------------------------------------------------------------
timer_start
echo "---- Stopping and removing any existing 'name_variants' container ----"
# Explanation:
#   - docker stop name_variants  -> stops the container if it exists, otherwise errors
#   - docker rm name_variants    -> removes the container
# We use || true to gracefully handle if there's no existing container to stop/remove
docker stop name_variants || true
docker rm name_variants || true
timer_end

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 7: Run the Docker container in detached mode
# ------------------------------------------------------------------------------
timer_start
echo "---- Running new 'name_variants' container in detached mode ----"
docker run -d \
  --name name_variants \
  --net consultant_ai \
  -p 8000:8000 \
  name_variants_image
timer_end

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 8: Final message
# ------------------------------------------------------------------------------
echo "Deployment script completed successfully."
