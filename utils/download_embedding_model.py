import logging
import time
import traceback

try:
    import psutil  # For optional resource usage
except ImportError:
    psutil = None

from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions

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
# ### Janis Rubins - Step 3: Function to initialize embedding functions
# ------------------------------------------------------------------------------
def initialize_embedding_functions(model_name="all-MiniLM-L6-v2"):
    """
    Creates two embedding functions (HuggingFaceEmbeddings and 
    SentenceTransformerEmbeddingFunction) using the same model name.
    """

    # ### Janis Rubins - Step 3.1: Log entry with input parameter
    logging.info(f"Entering initialize_embedding_functions with model_name={model_name}")
    start_time = time.time()
    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    huggingface_embedding = None
    chroma_embedding = None

    try:
        # ### Janis Rubins - Step 3.2: Create HuggingFaceEmbeddings
        logging.info("Initializing HuggingFaceEmbeddings...")
        huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
        logging.info("HuggingFaceEmbeddings created successfully.")

        # ### Janis Rubins - Step 3.3: Create SentenceTransformerEmbeddingFunction
        logging.info("Initializing SentenceTransformerEmbeddingFunction for Chroma...")
        chroma_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        logging.info("SentenceTransformerEmbeddingFunction created successfully.")

    except Exception as e:
        # ### Janis Rubins - Step 3.4: Log any errors encountered
        logging.error(f"Error while initializing embeddings with model '{model_name}': {e}")
        traceback.print_exc()
        # Graceful degradation path: Return None, or raise again
        return None, None

    finally:
        # ### Janis Rubins - Step 3.5: Measure performance and log resource usage
        end_time = time.time()
        duration = end_time - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            logging.info(
                f"initialize_embedding_functions performance: Time={duration:.2f}s, "
                f"MemBefore={mem_before}B, MemAfter={mem_after}B, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            logging.info(f"initialize_embedding_functions completed in {duration:.2f}s")

        logging.info("Exiting initialize_embedding_functions.")

    return huggingface_embedding, chroma_embedding

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Main entry point (example usage)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # ### Janis Rubins - Step 4.1: Log script start and measure total time
    script_start = time.time()
    logging.info("Script execution started.")

    # Log resource usage before initialization
    log_resource_usage()

    # Call the function to initialize embedding functions
    hf_embedding, st_embedding = initialize_embedding_functions("all-MiniLM-L6-v2")

    # Log resource usage after initialization
    log_resource_usage()

    # ### Janis Rubins - Step 4.2: Provide some logging on success/failure
    if hf_embedding and st_embedding:
        logging.info("Both embeddings were initialized successfully.")
    else:
        logging.warning("Embedding functions could not be fully initialized.")

    # ### Janis Rubins - Step 4.3: Final performance log
    script_end = time.time()
    total_duration = script_end - script_start
    logging.info(f"Script execution completed in {total_duration:.2f}s.")
