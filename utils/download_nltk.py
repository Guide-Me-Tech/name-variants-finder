import logging
import time
import traceback
import ssl
import nltk

try:
    import psutil  # Optional resource usage tracking
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
# ### Janis Rubins - Step 3: Setup SSL context for unverified HTTPS if needed
# ------------------------------------------------------------------------------
def setup_ssl_context():
    """
    Adjusts Python SSL context to allow unverified HTTPS downloads,
    if the platform or environment requires it.
    """
    logging.info("Setting up SSL context for NLTK downloads (if needed).")
    try:
        # ### Janis Rubins - Step 3.1: Attempt to set unverified HTTPS context
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # If not available, do nothing special
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Main function to download NLTK packages
# ------------------------------------------------------------------------------
def main():
    """
    Main function that sets up the SSL context and downloads all NLTK packages.
    """
    # ### Janis Rubins - Step 4.1: Log entry into main function
    logging.info("Script execution started. Preparing to download NLTK packages.")
    script_start = time.time()

    # Optional log of resource usage before operations
    log_resource_usage()

    # Setup SSL context
    setup_ssl_context()

    # ### Janis Rubins - Step 4.2: Attempt NLTK download in try-except block
    try:
        logging.info("Downloading all NLTK packages quietly. Please wait...")
        download_start = time.time()
        nltk.download("all", quiet=True)  # Install all NLTK packages
        download_end = time.time()
        logging.info(f"NLTK packages downloaded in {download_end - download_start:.2f}s")
    except Exception as e:
        # ### Janis Rubins - Step 4.3: Log error and stack trace
        logging.error(f"Error while downloading NLTK packages: {e}")
        traceback.print_exc()

    # Optional log of resource usage after operations
    log_resource_usage()

    # ### Janis Rubins - Step 4.4: Final performance log
    script_end = time.time()
    total_duration = script_end - script_start
    logging.info(f"Script completed successfully in {total_duration:.2f}s")

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
