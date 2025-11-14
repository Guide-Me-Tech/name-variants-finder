#!/usr/bin/env bash

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Set strict shell options for robust error handling
# ------------------------------------------------------------------------------
# Explanation:
#   - set -e: Exit immediately on any command failure.
#   - set -u: Treat references to unset variables as errors.
#   - set -o pipefail: Propagate errors in a pipeline correctly.
set -euo pipefail

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Define a trap to log unexpected errors with context
# ------------------------------------------------------------------------------
trap 'echo "[ERROR] A fatal error occurred at line $LINENO. Exiting..." >&2' ERR

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Optional CPU usage snapshot function
# ------------------------------------------------------------------------------
# Explanation:
#   - This function gives a quick CPU usage overview after certain operations.
#   - For advanced monitoring, consider tools like top, htop, or Docker metrics.
log_cpu_usage() {
  echo "[INFO] CPU usage snapshot:"
  ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 6
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Run Embed function to create embedded etcd configurations
# ------------------------------------------------------------------------------
run_embed() {
  local start_time
  start_time=$(date +%s)
  echo "[INFO] Entering run_embed() at $(date)."

  # ### Janis Rubins - Step 4.1: Generate etcd config (embedEtcd.yaml)
  cat << EOF > embedEtcd.yaml
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF
  echo "[INFO] embedEtcd.yaml file created."

  # ### Janis Rubins - Step 4.2: Generate user override config (user.yaml)
  cat << EOF > user.yaml
# Extra config to override default milvus.yaml
EOF
  echo "[INFO] user.yaml file created."

  # ### Janis Rubins - Step 4.3: Launch milvus-standalone container
  sudo docker run -d \
    --name milvus-standalone \
    --net consultant_ai \
    -d \
    --security-opt seccomp:unconfined \
    -e ETCD_USE_EMBED=true \
    -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
    -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
    -e COMMON_STORAGETYPE=local \
    -v "$(pwd)/volumes/milvus":/var/lib/milvus \
    -v "$(pwd)/embedEtcd.yaml":/milvus/configs/embedEtcd.yaml \
    -v "$(pwd)/user.yaml":/milvus/configs/user.yaml \
    -p 19530:19530 \
    -p 9091:9091 \
    -p 2379:2379 \
    --health-cmd="curl -f http://localhost:9091/healthz" \
    --health-interval=30s \
    --health-start-period=90s \
    --health-timeout=20s \
    --health-retries=3 \
    milvusdb/milvus:v2.4.5 \
    milvus run standalone  1> /dev/null

  echo "[INFO] milvus-standalone container started with embedded etcd."

  local end_time
  end_time=$(date +%s)
  echo "[INFO] Exiting run_embed(). Duration: $((end_time - start_time)) seconds."
  log_cpu_usage
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Wait for Milvus to become healthy
# ------------------------------------------------------------------------------
wait_for_milvus_running() {
  local start_time
  start_time=$(date +%s)
  echo "[INFO] Entering wait_for_milvus_running() at $(date). Waiting for Milvus to become healthy..."

  while true
  do
    local res
    res=$(sudo docker ps | grep milvus-standalone | grep healthy | wc -l || true)
    if [ "$res" -eq 1 ]; then
      echo "[INFO] Milvus started successfully."
      echo "[INFO] To change the default Milvus configuration, modify user.yaml and restart the service."
      break
    fi
    sleep 1
  done

  local end_time
  end_time=$(date +%s)
  echo "[INFO] Exiting wait_for_milvus_running(). Duration: $((end_time - start_time)) seconds."
  log_cpu_usage
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 6: Start Milvus
# ------------------------------------------------------------------------------
start() {
  local start_time
  start_time=$(date +%s)
  echo "[INFO] Entering start() at $(date). Checking if Milvus is already running..."

  local res
  res=$(sudo docker ps | grep milvus-standalone | grep healthy | wc -l || true)
  if [ "$res" -eq 1 ]; then
    echo "[INFO] Milvus is already running."
    exit 0
  fi

  res=$(sudo docker ps -a | grep milvus-standalone | wc -l || true)
  if [ "$res" -eq 1 ]; then
    echo "[INFO] Container exists. Starting existing milvus-standalone container..."
    sudo docker start milvus-standalone 1> /dev/null
  else
    echo "[INFO] No existing container found. Running run_embed()..."
    run_embed
  fi

  if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to start Milvus."
    exit 1
  fi

  wait_for_milvus_running
  local end_time
  end_time=$(date +%s)
  echo "[INFO] Exiting start(). Duration: $((end_time - start_time)) seconds."
  log_cpu_usage
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 7: Stop Milvus
# ------------------------------------------------------------------------------
stop() {
  local start_time
  start_time=$(date +%s)
  echo "[INFO] Entering stop() at $(date). Stopping milvus-standalone container..."

  # Explanation: We redirect stdout to /dev/null, but if an error occurs,
  # it will trigger our trap from set -e or be detected by $?.
  sudo docker stop milvus-standalone 1> /dev/null

  if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to stop Milvus."
    exit 1
  fi
  echo "[INFO] Milvus stopped successfully."

  local end_time
  end_time=$(date +%s)
  echo "[INFO] Exiting stop(). Duration: $((end_time - start_time)) seconds."
  log_cpu_usage
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 8: Delete Milvus
# ------------------------------------------------------------------------------
delete() {
  local start_time
  start_time=$(date +%s)
  echo "[INFO] Entering delete() at $(date). Deleting milvus-standalone container..."

  local res
  res=$(sudo docker ps | grep milvus-standalone | wc -l || true)
  if [ "$res" -eq 1 ]; then
    echo "[ERROR] Milvus container is running. Please stop Milvus before deleting."
    exit 1
  fi

  sudo docker rm milvus-standalone 1> /dev/null
  if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to remove milvus-standalone container."
    exit 1
  fi

  # Explanation: Remove volumes and config files to fully clean up.
  sudo rm -rf "$(pwd)/volumes"
  sudo rm -rf "$(pwd)/embedEtcd.yaml"
  sudo rm -rf "$(pwd)/user.yaml"
  echo "[INFO] Milvus container and related files deleted successfully."

  local end_time
  end_time=$(date +%s)
  echo "[INFO] Exiting delete(). Duration: $((end_time - start_time)) seconds."
  log_cpu_usage
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 9: Command-line arguments handling
# ------------------------------------------------------------------------------
# Explanation: We take one argument ($1) to decide which function to run.
case "${1:-}" in
  restart)
    stop
    start
    ;;
  start)
    start
    ;;
  stop)
    stop
    ;;
  delete)
    delete
    ;;
  *)
    echo "Usage: $0 {restart|start|stop|delete}"
    ;;
esac
