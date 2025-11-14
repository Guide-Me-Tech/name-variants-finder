import logging
import os
import time
import traceback

try:
    import psutil  # For resource usage, if available
except ImportError:
    psutil = None

import numpy as np
import redis
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging for the entire script
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------
# ### Janis Rubins - Step 2: Define file paths and model name
# ---------------------------------------------------------------------
FILENAMES = [
    "uzname-men-name-lat-v1.1b 2.txt",
    "uzname-women-name-lat-v1.1b.txt",
    "russian_male_names.txt",
    "russian_female_names.txt",
]
MODEL_NAME = "all-MiniLM-L6-v2"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# ---------------------------------------------------------------------
# ### Janis Rubins - Step 3: Helper function to read names from file
# ---------------------------------------------------------------------
def read_names_from_file(filepath):
    """
    Reads names from the given file, stripping whitespace.
    Logs any errors and returns a list of names.
    """
    # ### Janis Rubins - Step 3.1: Log entry
    logging.info(f"Entering read_names_from_file with filepath={filepath}")
    start_time = time.time()

    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    names = []
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file '{filepath}' does not exist.")

        with open(filepath, "r", encoding="utf-8") as f:
            raw_names = f.readlines()
            names = [name.strip() for name in raw_names]

        logging.info(f"Successfully read {len(names)} names from {filepath}")

    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        traceback.print_exc()
    except Exception as e:
        logging.error(f"Error reading from file {filepath}: {e}")
        traceback.print_exc()
    finally:
        execution_time = time.time() - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            logging.info(
                f"read_names_from_file performance: Time={execution_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            logging.info(f"read_names_from_file took {execution_time:.2f}s")

        # ### Janis Rubins - Step 3.2: Log exit with result
        logging.info(f"Exiting read_names_from_file with {len(names)} names loaded")
    return names

# ----------------------------------------------------------------------
# ### Janis Rubins - Step 4: Helper function to connect to Redis safely
# ----------------------------------------------------------------------
def get_redis_connection(host=REDIS_HOST, port=REDIS_PORT):
    """
    Establishes a connection to Redis. Returns the Redis client or None on failure.
    """
    logging.info(f"Entering get_redis_connection with host={host}, port={port}")
    try:
        r_conn = redis.Redis(host=host, port=port)
        # Test connectivity
        if not r_conn.ping():
            raise ConnectionError("Redis ping failed.")
        logging.info("Successfully connected to Redis")
        return r_conn
    except Exception as e:
        logging.error(f"Failed to connect to Redis: {e}")
        traceback.print_exc()
        return None

# ----------------------------------------------------------------------
# ### Janis Rubins - Step 5: Main function to orchestrate the workflow
# ----------------------------------------------------------------------
def main():
    # ### Janis Rubins - Step 5.1: Start logging for main
    logging.info("Script execution started.")
    start_time = time.time()

    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    # ------------------------------------------------------------------
    # ### Janis Rubins - Step 5.2: Read names from files
    # ------------------------------------------------------------------
    uz_names = []
    russian_names = []

    # Because each file is a possible failure point, we do them individually
    # so we can degrade gracefully.
    uz_names_part1 = read_names_from_file(FILENAMES[0])
    uz_names_part2 = read_names_from_file(FILENAMES[1])
    uz_names = uz_names_part1 + uz_names_part2

    russian_names_part1 = read_names_from_file(FILENAMES[2])
    russian_names_part2 = read_names_from_file(FILENAMES[3])
    russian_names = russian_names_part1 + russian_names_part2

    # ------------------------------------------------------------------
    # ### Janis Rubins - Step 5.3: Initialize sentence transformer
    # ------------------------------------------------------------------
    logging.info(f"Loading model '{MODEL_NAME}' for embeddings.")
    try:
        model = SentenceTransformer(MODEL_NAME)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model {MODEL_NAME}: {e}")
        traceback.print_exc()
        return  # If model loading fails, we cannot proceed

    # ------------------------------------------------------------------
    # ### Janis Rubins - Step 5.4: Encode Russian names
    # ------------------------------------------------------------------
    # Retaining the original logic: only encode Russian names in `texts`
    texts = russian_names
    embeddings = []
    try:
        logging.info(f"Encoding {len(texts)} Russian names.")
        encode_start = time.time()
        embeddings = model.encode(texts)  # Get embeddings as numpy arrays
        encode_time = time.time() - encode_start
        logging.info(
            f"Successfully encoded {len(embeddings)} embeddings in {encode_time:.2f}s"
        )
    except Exception as e:
        logging.error(f"Error encoding texts: {e}")
        traceback.print_exc()
        return  # If encoding fails, we cannot proceed further

    # ------------------------------------------------------------------
    # ### Janis Rubins - Step 5.5: Connect to Redis and store embeddings
    # ------------------------------------------------------------------
    redis_conn = get_redis_connection()
    if not redis_conn:
        logging.error("Cannot proceed with storing embeddings: Redis connection failed.")
        return

    # If we got here, we have a valid redis_conn
    store_start = time.time()
    try:
        logging.info(f"Storing {len(embeddings)} embeddings in Redis.")
        for i, embedding in enumerate(embeddings):
            key = f"doc:{i}"
            # Convert embedding to float32 bytes
            redis_conn.hset(
                key,
                mapping={
                    "embedding": np.array(
                        embedding, dtype=np.float32
                    ).tobytes()
                },
            )
        logging.info("All embeddings stored successfully in Redis.")
    except Exception as e:
        logging.error(f"Error while storing embeddings in Redis: {e}")
        traceback.print_exc()
    finally:
        store_time = time.time() - store_start
        logging.info(
            f"Storing embeddings took {store_time:.2f}s"
        )

    # ------------------------------------------------------------------
    # ### Janis Rubins - Step 5.6: Final performance metrics and exit
    # ------------------------------------------------------------------
    total_execution_time = time.time() - start_time
    if psutil and mem_before is not None and cpu_before is not None:
        mem_after = psutil.virtual_memory().used
        cpu_after = psutil.cpu_percent(interval=None)
        logging.info(
            f"Overall performance: Time={total_execution_time:.2f}s, "
            f"MemBefore={mem_before}, MemAfter={mem_after}, "
            f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
        )
    else:
        logging.info(
            f"Script executed in {total_execution_time:.2f}s"
        )

    # ### Janis Rubins - Step 5.7: Log completion
    logging.info("Script execution completed successfully.")

# ----------------------------------------------------------------------
# ### Janis Rubins - Step 6: Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
