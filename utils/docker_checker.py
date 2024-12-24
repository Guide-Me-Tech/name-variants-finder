import logging
import time
import os
import traceback

try:
    import psutil  # For optional resource usage
except ImportError:
    psutil = None

import docker

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Optional function to log CPU/Memory usage
# ------------------------------------------------------------------------------
def log_resource_usage():
    """
    Logs system resource usage if psutil is available.
    """
    if psutil:
        mem_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        logging.info(f"Memory Usage: {mem_info.used / (1024*1024):.2f} MB / {mem_info.total / (1024*1024):.2f} MB")
        logging.info(f"CPU Usage: {cpu_percent:.2f}%")
    else:
        logging.info("psutil not installed. Skipping resource usage logging.")

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Main function to monitor containers
# ------------------------------------------------------------------------------
def main():
    """
    Continuously checks all Docker containers, and if 'classifier' is stopped,
    it restarts it by calling an external shell script.
    """
    # ### Janis Rubins - Step 3.1: Log entry into main function
    logging.info("Starting Docker container monitor script.")
    start_time = time.time()

    # Optional log of initial resource usage
    log_resource_usage()

    # Initialize Docker client
    try:
        logging.info("Creating Docker client from environment.")
        client = docker.from_env()
        logging.info("Docker client created successfully.")
    except Exception as e:
        logging.error(f"Failed to create Docker client: {e}")
        traceback.print_exc()
        return  # Cannot proceed without a valid client

    # Continuous monitoring loop
    while True:
        loop_start_time = time.time()

        try:
            # ### Janis Rubins - Step 3.2: Fetch container list
            containers = client.containers.list(all=True)
            logging.info(f"Found {len(containers)} containers in total.")

            # Print summary line
            container_names = []
            for container in containers:
                container_names.append(container.attrs["Name"])
            logging.info(f"Container names: {container_names}")

            # ### Janis Rubins - Step 3.3: Check if 'classifier' is not running
            for container in containers:
                if container.attrs["Name"] == "/classifier":
                    status = container.attrs["State"]["Status"]
                    logging.info(f"Classifier status: {status}")

                    if status != "running":
                        logging.warning("Classifier is stopped. Attempting to restart.")
                        # Attempt to call the shell script
                        try:
                            os.system("sh ~/aslon/consultant_ai_sentence_classifier/utils/shell_scripts/restart.sh")
                            logging.info("Restart script executed successfully.")
                        except Exception as ex:
                            logging.error(f"Failed to run restart script: {ex}")
                            traceback.print_exc()

        except docker.errors.APIError as api_err:
            # ### Janis Rubins - Step 3.4: Log Docker API errors
            logging.error(f"Docker API error: {api_err}")
            traceback.print_exc()

        except Exception as ex:
            # ### Janis Rubins - Step 3.5: Log unexpected errors
            logging.error(f"Unexpected error in monitor loop: {ex}")
            traceback.print_exc()

        # ### Janis Rubins - Step 3.6: Performance logging
        loop_end_time = time.time()
        loop_duration = loop_end_time - loop_start_time
        logging.info(f"Monitor loop iteration took {loop_duration:.2f} seconds.")

        # Optional resource usage each iteration
        log_resource_usage()

        # Sleep 10 seconds before next iteration
        time.sleep(10)


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # ### Janis Rubins - Step 4.1: Start main and measure overall performance
    script_start = time.time()
    main()
    script_end = time.time()
    total_duration = script_end - script_start
    logging.info(f"Script exited after {total_duration:.2f} seconds.")
