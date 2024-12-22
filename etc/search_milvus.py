import logging
import time
import traceback

try:
    import psutil  # Optional: for resource usage
except ImportError:
    psutil = None

from dotenv import load_dotenv
from pymilvus import MilvusClient, model

# ------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging for the entire script
# ------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Load environment variables
# ------------------------------------------------------------------------
load_dotenv()
logging.info("Environment variables loaded.")

# ------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Create the embedding function (SentenceTransformer)
# ------------------------------------------------------------------------
def get_embedding_function(model_name="all-MiniLM-L6-v2", device="cpu"):
    """
    Initializes the SentenceTransformer embedding function using pymilvus.model.
    Raises an exception if initialization fails.
    """
    # ### Janis Rubins - Step 3.1: Log entry with parameters
    logging.info(
        f"Entering get_embedding_function with model_name={model_name}, device={device}"
    )
    start_time = time.time()

    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    try:
        # ### Janis Rubins - Step 3.2: Initialize SentenceTransformerEmbeddingFunction
        embedding_fn = model.dense.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
        )
        logging.info(
            f"Created SentenceTransformerEmbeddingFunction with model={model_name}, device={device}"
        )
        return embedding_fn

    except Exception as e:
        # ### Janis Rubins - Step 3.3: Log any error and re-raise
        logging.error(f"Error while creating embedding function: {e}")
        traceback.print_exc()
        raise

    finally:
        # ### Janis Rubins - Step 3.4: Log performance metrics and exit
        elapsed_time = time.time() - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            logging.info(
                f"get_embedding_function performance: "
                f"Time={elapsed_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            logging.info(f"get_embedding_function took {elapsed_time:.2f}s")
        logging.info("Exiting get_embedding_function.")

# ------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Create Milvus client and perform search
# ------------------------------------------------------------------------
def main():
    """
    Main function that initializes the embedding function, creates a Milvus client,
    encodes a query word, and searches a specified collection.
    """
    # ### Janis Rubins - Step 4.1: Log script start
    logging.info("Script execution started.")
    script_start_time = time.time()

    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    # --------------------------------------------------------------------
    # ### Janis Rubins - Step 4.2: Create the embedding function
    # --------------------------------------------------------------------
    try:
        sentence_transformer_ef = get_embedding_function(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )
    except Exception as e:
        logging.error(f"Failed to initialize embedding function: {e}")
        return  # Cannot proceed without embeddings

    # --------------------------------------------------------------------
    # ### Janis Rubins - Step 4.3: Create Milvus client
    # --------------------------------------------------------------------
    try:
        logging.info("Creating Milvus client with URI='http://localhost:19530'")
        client_start_time = time.time()
        client = MilvusClient(uri="http://localhost:19530")
        client_elapsed = time.time() - client_start_time
        logging.info(f"Milvus client created successfully in {client_elapsed:.2f}s.")
    except Exception as e:
        logging.error(f"Error creating Milvus client: {e}")
        traceback.print_exc()
        return

    # --------------------------------------------------------------------
    # ### Janis Rubins - Step 4.4: Query word embedding
    # --------------------------------------------------------------------
    try:
        query_word = "роман"
        logging.info(f"Creating embedding for word: {query_word}")
        embed_start_time = time.time()
        embedded_name = create_embedding_for_word(query_word, sentence_transformer_ef)
        embed_elapsed = time.time() - embed_start_time
        logging.info(
            f"Time for embedding word='{query_word}': {embed_elapsed:.4f}s"
        )
    except Exception as e:
        logging.error(f"Error creating embedding for '{query_word}': {e}")
        traceback.print_exc()
        return

    # --------------------------------------------------------------------
    # ### Janis Rubins - Step 4.5: Search the Milvus collection
    # --------------------------------------------------------------------
    try:
        search_start_time = time.time()
        res = client.search(
            collection_name="rus_names",
            data=embedded_name,
            filter="",
            limit=10,
            output_fields=["name"],
        )
        search_elapsed = time.time() - search_start_time
        logging.info(
            f"Time taken for search in 'rus_names': {search_elapsed:.4f}s"
        )
        logging.info(f"Search results: {res}")
    except Exception as e:
        logging.error(f"Error while searching Milvus collection: {e}")
        traceback.print_exc()

    finally:
        # ----------------------------------------------------------------
        # ### Janis Rubins - Step 4.6: Log overall performance and exit
        # ----------------------------------------------------------------
        total_time = time.time() - script_start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            logging.info(
                f"Overall script performance: Time={total_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            logging.info(f"Script executed in {total_time:.2f}s")

        logging.info("Script execution completed successfully.")

# ------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Helper function to create embedding for word
# ------------------------------------------------------------------------
def create_embedding_for_word(word, embedding_fn):
    """
    Uses the provided embedding function to encode a single word into a vector.
    Raises exceptions on error.
    """
    # ### Janis Rubins - Step 5.1: Log entry with input parameter
    logging.info(f"Entering create_embedding_for_word with word='{word}'")
    start_time = time.time()

    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    try:
        vectors = embedding_fn.encode_documents([word])
        return vectors
    except Exception as e:
        logging.error(f"Failed to encode word='{word}': {e}")
        traceback.print_exc()
        raise
    finally:
        # ### Janis Rubins - Step 5.2: Log performance and exit
        execution_time = time.time() - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            logging.info(
                f"create_embedding_for_word performance: "
                f"Time={execution_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            logging.info(
                f"create_embedding_for_word took {execution_time:.2f}s"
            )
        logging.info(f"Exiting create_embedding_for_word for '{word}'.")

# ------------------------------------------------------------------------
# ### Janis Rubins - Step 6: Entry point
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
