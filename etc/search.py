import logging
import time
import traceback

try:
    import psutil  # For optional resource usage tracking
except ImportError:
    psutil = None

import numpy as np
import redis
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging for detailed script reporting
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Main function orchestrating the search operation
# --------------------------------------------------------------------------
def main():
    """
    Prompts user for a search text, generates an embedding via SentenceTransformer,
    and queries Redis for the top 3 similar vectors.
    """
    # ### Janis Rubins - Step 2.1: Log script start and capture baseline resources
    logging.info("Script execution started.")
    script_start_time = time.time()
    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    # ----------------------------------------------------------------------
    # ### Janis Rubins - Step 2.2: Prompt for user input with error handling
    # ----------------------------------------------------------------------
    try:
        logging.info("Prompting user for search text.")
        q = input("Enter search text: ").strip()
        logging.info(f"User provided search text: '{q}'")
        if not q:
            raise ValueError("No search text provided.")
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return  # Graceful exit if input is invalid
    except Exception as e:
        logging.error(f"Unexpected error while reading user input: {e}")
        traceback.print_exc()
        return  # Graceful exit for any other unforeseen error

    # ----------------------------------------------------------------------
    # ### Janis Rubins - Step 2.3: Initialize SentenceTransformer and encode
    # ----------------------------------------------------------------------
    try:
        logging.info("Loading SentenceTransformer model: 'all-MiniLM-L6-v2'.")
        model_load_start = time.time()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        load_time = time.time() - model_load_start
        logging.info(f"Model loaded successfully in {load_time:.2f}s.")

        logging.info(f"Encoding user query: '{q}'.")
        encode_start = time.time()
        query_embedding = model.encode([q])[0]  # encode returns a list; we take the first
        encode_time = time.time() - encode_start
        logging.info(f"Query encoded in {encode_time:.2f}s.")

    except Exception as e:
        logging.error(f"Error creating model or encoding text: {e}")
        traceback.print_exc()
        return

    # ----------------------------------------------------------------------
    # ### Janis Rubins - Step 2.4: Connect to Redis and perform search
    # ----------------------------------------------------------------------
    try:
        logging.info("Connecting to Redis at host='localhost', port=6379.")
        r = redis.Redis(host="localhost", port=6379)
        if not r.ping():
            raise ConnectionError("Redis connection ping failed.")
        logging.info("Redis connection established successfully.")

        logging.info("Executing vector similarity search in Redis.")
        search_start = time.time()
        result = r.ft("embeddings_idx").search(
            query="*=>[KNN 3 @embedding $query_embedding AS score]",
            query_params={
                "query_embedding": np.array(query_embedding, dtype=np.float32).tobytes()
            },
        )
        search_time = time.time() - search_start
        logging.info(f"Search completed in {search_time:.2f}s with {result.total} results found.")

        # ### Janis Rubins - Step 2.5: Print or process the returned results
        for doc in result.docs:
            # doc may have fields like 'id', 'score', etc.
            print(f"Document ID: {doc.id}, Score: {doc.score}")

    except ConnectionError as ce:
        logging.error(f"ConnectionError: {ce}")
        traceback.print_exc()
    except redis.exceptions.RedisError as re:
        logging.error(f"RedisError: {re}")
        traceback.print_exc()
    except Exception as e:
        logging.error(f"Unexpected error during Redis search: {e}")
        traceback.print_exc()

    # ----------------------------------------------------------------------
    # ### Janis Rubins - Step 2.6: Log overall performance and exit
    # ----------------------------------------------------------------------
    total_time = time.time() - script_start_time
    if psutil and mem_before is not None and cpu_before is not None:
        mem_after = psutil.virtual_memory().used
        cpu_after = psutil.cpu_percent(interval=None)
        logging.info(
            "Overall script performance: "
            f"Time={total_time:.2f}s, "
            f"MemBefore={mem_before}, MemAfter={mem_after}, "
            f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
        )
    else:
        logging.info(f"Script executed in {total_time:.2f}s.")

    logging.info("Script execution completed successfully.")

# --------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Entry point to run the main function
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
