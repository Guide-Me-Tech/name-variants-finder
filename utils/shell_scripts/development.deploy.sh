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
# ### Janis Rubins - Step 2: Optionally log system resource usage
# ------------------------------------------------------------------------------
log_resource_usage() {
  log_info "Logging current memory usage:"
  free -h || true  # 'free' may not exist on all systems

  log_info "Logging top CPU-consuming processes:"
  ps -eo pid,comm,pcpu,pmem --sort=-pcpu | head -n 5 || true
}

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Build Docker image with error handling
# ------------------------------------------------------------------------------
build_docker_image() {
  local dockerfile="dockerfiles/dockerfile.development"
  local tag="aslon1213/classifier_development"

  # ### Janis Rubins - Step 3.1: Log entry
  log_info "Building Docker image with Dockerfile: $dockerfile and tag: $tag"
  local start_time
  start_time=$(date +%s)

  # Attempt to build the image
  if ! docker build --file "$dockerfile" -t "$tag" .; then
    log_error "Failed to build Docker image: $tag"
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
  local container_name="classifier"
  
  # ### Janis Rubins - Step 4.1: Log entry
  log_info "Stopping and removing container: $container_name (if running)"
  local start_time
  start_time=$(date +%s)

  # 'docker container stop' and 'rm' might fail if container is not running
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
  local container_name="classifier"
  local image_name="aslon1213/classifier_development"
  local host_port1="50051"
  local container_port1="50051"
  local host_port2="50050"
  local container_port2="50050"
  local mount_source="/home/sbadmin/aslon/consultant_ai_sentence_classifier/"
  local mount_target="/usr/src/app/"
  local docker_net="consultant_ai"

  # ### Janis Rubins - Step 5.1: Log entry with parameters
  log_info "Running Docker container: $container_name from image: $image_name"
  log_info "Ports mapped: $host_port1->$container_port1, $host_port2->$container_port2"
  log_info "Mounting: $mount_source -> $mount_target"
  local start_time
  start_time=$(date +%s)

  # Attempt to run container
  if ! docker run -d \
    -p "${host_port1}:${container_port1}" \
    -p "${host_port2}:${container_port2}" \
    --mount type=bind,source="${mount_source}",target="${mount_target}" \
    --net "${docker_net}" \
    --name "${container_name}" \
    "${image_name}"
  then
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
  log_info "Script execution started: building image, stopping/removing container, and running new container."
  local script_start
  script_start=$(date +%s)

  # Optional resource usage logging at the beginning
  log_resource_usage

  build_docker_image
  stop_and_remove_container
  run_container

  # Optional resource usage logging at the end
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
