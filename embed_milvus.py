import logging
import os
import time
import traceback

# Attempt to import psutil for resource usage metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from dotenv import load_dotenv
from pymilvus import model, MilvusClient

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging
# ------------------------------------------------------------------------------
# Explanation:
#   - We configure the logging system so that every major event is logged to the console.
#   - The level can be changed to DEBUG for more verbose output if needed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Define a helper function for measuring performance
# ------------------------------------------------------------------------------
def log_performance_metrics(step_name, start_time):
    """Logs elapsed time and optional CPU/memory usage for a code block."""
    end_time = time.time()
    elapsed = end_time - start_time

    if PSUTIL_AVAILABLE:
        mem_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        logging.info(
            f"[{step_name}] Elapsed: {elapsed:.2f}s, "
            f"Memory Used: {mem_info.used / (1024**2):.2f} MB, CPU: {cpu_percent}%"
        )
    else:
        logging.info(f"[{step_name}] Elapsed: {elapsed:.2f}s")


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Main entry function
# ------------------------------------------------------------------------------
def main():
    # ### Janis Rubins - Step 3.1: Announce script start
    logging.info("Embedding names to Milvus DB process started.")

    # ### Janis Rubins - Step 3.2: Load environment variables
    start_time = time.time()
    try:
        load_dotenv()
        logging.info("Environment variables loaded successfully.")
    except Exception as e:
        logging.error("Failed to load environment variables.")
        traceback.print_exc()
    log_performance_metrics("Load .env", start_time)

    # ### Janis Rubins - Step 3.3: Initialize the embedding function
    start_time = time.time()
    sentence_transformer_ef = None
    try:
        logging.info("Loading SentenceTransformerEmbeddingFunction...")
        sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )
        logging.info("SentenceTransformerEmbeddingFunction loaded successfully.")
    except Exception as e:
        logging.error("Failed to initialize embedding function.")
        traceback.print_exc()
        return  # Cannot proceed without embeddings
    log_performance_metrics("Initialize embedding function", start_time)

    # ### Janis Rubins - Step 3.4: Read names from files
    # Explanation: We create a helper function for reading names for better error handling.
    uz_names, russian_names = read_name_files()

    # ### Janis Rubins - Step 3.5: Embed the names
    # Explanation: We capture performance metrics around embedding operation.
    start_time = time.time()
    try:
        logging.info("Embedding Russian names...")
        docs_embeddings_russian = sentence_transformer_ef.encode_documents(russian_names)

        logging.info("Embedding Uzbek names...")
        docs_embeddings_uzbek = sentence_transformer_ef.encode_documents(uz_names)
        logging.info("Names embedding completed.")
    except Exception as e:
        logging.error("Failed to embed names.")
        traceback.print_exc()
        return
    log_performance_metrics("Embed documents", start_time)

    # ### Janis Rubins - Step 3.6: Print some dimension info for verification
    # Explanation: This helps confirm we have embeddings of the correct size.
    logging.info(
        f"Dimension: {sentence_transformer_ef.dim}, "
        f"Russian embedding shape: {docs_embeddings_russian[0].shape}, "
        f"Uzbek embedding shape: {docs_embeddings_uzbek[0].shape}"
    )

    # ### Janis Rubins - Step 3.7: Connect to Milvus
    # Explanation: We connect using the environment variable MILVUS_URI or default to localhost.
    start_time = time.time()
    client = None
    try:
        logging.info("Connecting to Milvus...")
        client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
        logging.info("Connected to Milvus.")
    except Exception as e:
        logging.error("Failed to connect to Milvus.")
        traceback.print_exc()
        return
    log_performance_metrics("Connect to Milvus", start_time)

    # ### Janis Rubins - Step 3.8: Create or recreate collections and insert data
    # Explanation: We recreate two collections: rus_names and uzbek_names.
    recreate_and_insert(client, "rus_names", russian_names, docs_embeddings_russian)
    recreate_and_insert(client, "uzbek_names", uz_names, docs_embeddings_uzbek)

    # ### Janis Rubins - Step 3.9: Announce script completion
    logging.info("All data inserted into Milvus successfully. Process finished.")


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Helper function to read name files
# ------------------------------------------------------------------------------
def read_name_files():
    """
    Reads Uzbek and Russian names from predefined text files.
    Returns (uz_names, russian_names) as lists of strings.
    """
    # ### Janis Rubins - Step 4.1: Initialize container lists
    uz_names = []
    russian_names = []

    # ### Janis Rubins - Step 4.2: Filenames to read
    filenames = [
        "data_names/uzbek_names_set_merged.txt",
        "data_names/russian_names_set_merged.txt",
    ]

    # ### Janis Rubins - Step 4.3: Read files safely
    start_time = time.time()
    try:
        # Uzbek names
        logging.info(f"Reading Uzbek names from {filenames[0]}...")
        with open(filenames[0], "r", encoding="utf-8") as f:
            uz_names = [line.strip() for line in f.readlines()]

        # Russian names
        logging.info(f"Reading Russian names from {filenames[1]}...")
        with open(filenames[1], "r", encoding="utf-8") as f:
            russian_names = [line.strip() for line in f.readlines()]

        logging.info("Names loaded from files.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
        traceback.print_exc()
    except Exception as e:
        logging.error("Error while reading name files.")
        traceback.print_exc()
    log_performance_metrics("Read names from files", start_time)

    return uz_names, russian_names


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Helper function to create or recreate collections
# ------------------------------------------------------------------------------
def recreate_and_insert(client, collection_name, names, embeddings):
    """
    Drops the specified collection (if it exists) and recreates it,
    then inserts the provided data.
    """
    # ### Janis Rubins - Step 5.1: Start performance tracking
    start_time = time.time()

    # ### Janis Rubins - Step 5.2: Recreate the collection
    try:
        if client.has_collection(collection_name=collection_name):
            logging.info(f"Dropping existing collection '{collection_name}'...")
            client.drop_collection(collection_name=collection_name)

        logging.info(f"Creating collection '{collection_name}'...")
        # Example dimension is set to 384 (for 'all-MiniLM-L6-v2').
        client.create_collection(
            collection_name=collection_name,
            dimension=384,
        )
        logging.info(f"Collection '{collection_name}' created.")
    except Exception as e:
        logging.error(f"Failed to recreate collection '{collection_name}'.")
        traceback.print_exc()
        return

    # ### Janis Rubins - Step 5.3: Prepare data for insertion
    # Explanation: We map each entry to a dictionary required by PyMilvus v2.
    data_to_insert = prepare_data(names, embeddings)

    # ### Janis Rubins - Step 5.4: Insert data into Milvus
    try:
        logging.info(f"Inserting data into '{collection_name}'...")
        client.insert(
            collection_name=collection_name,
            data=data_to_insert,
        )
        logging.info(f"Data inserted into '{collection_name}' successfully.")
    except Exception as e:
        logging.error(f"Failed to insert data into '{collection_name}'.")
        traceback.print_exc()

    # ### Janis Rubins - Step 5.5: Performance metrics
    log_performance_metrics(f"Recreate & Insert {collection_name}", start_time)


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 6: Helper function to prepare data for Milvus
# ------------------------------------------------------------------------------
def prepare_data(names, embeddings):
    """
    Converts lists of names and embeddings into a list of dictionaries
    that PyMilvus can insert.
    """
    # ### Janis Rubins - Step 6.1: Log function entry
    logging.info("Preparing data for Milvus insertion.")
    start_time = time.time()

    data = []
    for i in range(len(names)):
        data.append(
            {
                "id": i,
                "vector": embeddings[i],
                "name": names[i],
            }
        )

    # ### Janis Rubins - Step 6.2: Log function exit with performance
    log_performance_metrics("Prepare data", start_time)
    return data


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 7: Script entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
