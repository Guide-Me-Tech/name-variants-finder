#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Define logging functions
# ------------------------------------------------------------------------------
log_info() {
  echo "$(date +'%Y-%m-%d %H:%M:%S') - INFO - $*"
}

log_error() {
  echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR - $*" >&2
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Optionally log resource usage (CPU, memory, etc.)
# ------------------------------------------------------------------------------
log_resource_usage() {
  log_info "System memory usage:"
  free -h || true
  log_info "CPU and memory usage by processes:"
  ps -eo pid,comm,pcpu,pmem --sort=-pcpu | head -n 5 || true
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Build Docker image with error handling
# ------------------------------------------------------------------------------
build_docker_image() {
  local dockerfile="dockerfiles/dockerfile.classifier_server"
  local tag="aslon1213/classifier_server"
  
  # ### Janis Rubins - Step 3.1: Log entry
  log_info "Building Docker image using Dockerfile: $dockerfile with tag: $tag"
  local start_time
  start_time=$(date +%s)

  # Try building the image, capturing errors
  if ! docker build --file "$dockerfile" -t "$tag" .; then
    log_error "Failed to build Docker image $tag"
    exit 1
  fi

  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - start_time))
  log_info "Docker build completed in ${duration}s"
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Stop and remove existing container gracefully
# ------------------------------------------------------------------------------
stop_and_remove_container() {
  local container_name="classifier_server"
  # ### Janis Rubins - Step 4.1: Log entry
  log_info "Stopping and removing container: $container_name (if running)"
  local start_time
  start_time=$(date +%s)

  # 'docker container stop' may fail if container not running, so allow errors
  set +e
  docker container stop "$container_name" >/dev/null 2>&1
  docker container rm "$container_name" >/dev/null 2>&1
  set -e
  
  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - start_time))
  log_info "Stopping/removing container took ${duration}s"
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Run the new container
# ------------------------------------------------------------------------------
run_container() {
  local container_name="classifier_server"
  local image_name="aslon1213/classifier_server:latest"
  # ### Janis Rubins - Step 5.1: Log entry
  log_info "Running Docker container: $container_name from image: $image_name"
  local start_time
  start_time=$(date +%s)

  # Attempt to run container
  if ! docker run -d -p 50052:50050 --net consultant_ai --name "$container_name" "$image_name"; then
    log_error "Failed to start Docker container: $container_name"
    exit 1
  fi
  
  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - start_time))
  log_info "Docker container $container_name started successfully in ${duration}s"
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 6: Main script flow
# ------------------------------------------------------------------------------
main() {
  # ### Janis Rubins - Step 6.1: Log script start
  log_info "Script execution started. Building, stopping/removing container, and running new container."
  local script_start
  script_start=$(date +%s)

  # Optional resource usage log
  log_resource_usage

  build_docker_image
  stop_and_remove_container
  run_container

  # Log resource usage again (optional)
  log_resource_usage

  # ### Janis Rubins - Step 6.2: Final performance log
  local script_end
  script_end=$(date +%s)
  local total_duration=$((script_end - script_start))
  log_info "Script completed successfully in ${total_duration}s"
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 7: Execute main function
# ------------------------------------------------------------------------------
main
