import logging
import time
import traceback
import os

try:
    import psutil  # For optional resource usage tracking
except ImportError:
    psutil = None

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Optional function to log CPU and memory usage
# ------------------------------------------------------------------------------
def log_resource_usage():
    """
    Logs system resource usage if 'psutil' is installed.
    """
    if psutil:
        mem_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        logging.info(
            f"Memory Usage: {mem_info.used / (1024 * 1024):.2f} MB / "
            f"{mem_info.total / (1024 * 1024):.2f} MB, CPU Usage: {cpu_percent:.2f}%"
        )
    else:
        logging.info("psutil not installed. Skipping resource usage logging.")

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: List of dependencies to install
# ------------------------------------------------------------------------------
dependencies_list = [
    "langchain",
    "openai",
    "python-dotenv",
    "chromadb",
    "orjson",
    "grpcio-tools",
    "tensorflow",
    "bs4",
    "nltk",
    "sentence_transformers",
    "langchain_openai",
    "langchainhub",  # From your commented references
]

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Function to install dependencies
# ------------------------------------------------------------------------------
def install_dependencies(dependencies):
    """
    Installs a list of Python packages using pip. Logs progress,
    errors, and resource usage. Returns a summary message.
    """
    # ### Janis Rubins - Step 4.1: Log entry and measure performance
    logging.info("Entering install_dependencies function.")
    start_time = time.time()
    log_resource_usage()

    # We'll keep a record of successes and failures
    successes = []
    failures = []

    for dependency in dependencies:
        dep_start_time = time.time()
        logging.info(f"Attempting to install: {dependency}")
        try:
            # ### Janis Rubins - Step 4.2: Attempt to install using 'os.system'
            exit_code = os.system(f"pip install {dependency}")
            # Check for non-zero exit codes to detect failures
            if exit_code == 0:
                logging.info(f"Successfully installed: {dependency}")
                successes.append(dependency)
            else:
                error_msg = f"Installation failed for {dependency} with exit code {exit_code}"
                logging.error(error_msg)
                failures.append(dependency)
        except Exception as e:
            # ### Janis Rubins - Step 4.3: Log exceptions with stack trace
            logging.error(f"Exception installing {dependency}: {e}")
            traceback.print_exc()
            failures.append(dependency)

        dep_duration = time.time() - dep_start_time
        logging.info(f"Time taken to install '{dependency}': {dep_duration:.2f}s")
        log_resource_usage()

    total_time = time.time() - start_time
    logging.info(f"install_dependencies completed in {total_time:.2f}s")
    log_resource_usage()

    # ### Janis Rubins - Step 4.4: Log results and exit
    if failures:
        logging.warning(f"Some dependencies failed to install: {failures}")
    else:
        logging.info("All dependencies installed successfully.")

    return f"Installation completed. Successes: {len(successes)}, Failures: {len(failures)}"

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Main entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # ### Janis Rubins - Step 5.1: Script start logging
    logging.info("Script execution started. Installing dependencies...")

    summary = install_dependencies(dependencies_list)
    logging.info(summary)

    logging.info("Script execution finished.")


# pip install langchain
# 10487  pip install openai
# 10488  pip install dotenv
# 10489  pip install python-dotenv
# 10490  pip install chromadb
# 10491  pip install orjson
# 10492  pip install grpcio-tools
# 10494  pip install tensorflow
# 10495  pip install bs4 nlyk
# 10496  pip install bs4 nltk
# 10498  pip install sentence_transformers
# 10500  pip install langchain_openai
# 10509  pip install langchainhub
