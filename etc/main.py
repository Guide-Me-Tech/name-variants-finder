import os
import json
import logging
import time
import traceback

try:
    import psutil  # Optional: used for resource usage
except ImportError:
    psutil = None

import chromadb
from chromadb.utils import embedding_functions

# --------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging for comprehensive output
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------------------------
# ### Janis Rubins - Step 2: Define the main function
# --------------------------------------------------------------------
def main():
    """
    Main script entry point: sets up embeddings, connects to ChromaDB,
    and upserts Uzbek and Russian names into their respective collections.
    """
    # ### Janis Rubins - Step 2.1: Log the start of the script
    logging.info("Script execution started.")
    start_time = time.time()

    # Optionally track system resource usage
    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    # ### Janis Rubins - Step 2.2: Prepare data (assuming these lists are populated elsewhere)
    uz_names = []
    russian_names = []

    # ### Janis Rubins - Step 2.3: Create embedding function
    try:
        logging.info("Initializing SentenceTransformer embedding function.")
        stef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        logging.info("Embedding function initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the embedding function: {e}")
        traceback.print_exc()
        return  # Cannot proceed without embeddings

    # ### Janis Rubins - Step 2.4: Initialize ChromaDB PersistentClient
    try:
        logging.info("Creating PersistentClient to connect to ChromaDB.")
        chroma_instance = chromadb.PersistentClient("./names")
        logging.info("ChromaDB PersistentClient created successfully.")
    except Exception as e:
        logging.error(f"Error creating ChromaDB client: {e}")
        traceback.print_exc()
        return  # Cannot proceed without a valid Chroma client

    # ### Janis Rubins - Step 2.5: Retrieve or create collections
    try:
        logging.info("Retrieving or creating uz_names collection.")
        uz_names_collection = chroma_instance.get_or_create_collection(
            "uz_names", embedding_function=stef
        )
        logging.info("Retrieving or creating russian_names collection.")
        russian_names_collection = chroma_instance.get_or_create_collection(
            "russian_names", embedding_function=stef
        )
    except Exception as e:
        logging.error(f"Error retrieving/creating ChromaDB collections: {e}")
        traceback.print_exc()
        return  # Cannot proceed without valid collections

    # ### Janis Rubins - Step 2.6: Upsert documents into collections
    try:
        logging.info("Upserting documents into uz_names collection.")
        upsert_documents(uz_names_collection, uz_names)

        logging.info("Upserting documents into russian_names collection.")
        upsert_documents(russian_names_collection, russian_names)
    except Exception as e:
        logging.error(f"Error while upserting documents: {e}")
        traceback.print_exc()

    # ### Janis Rubins - Step 2.7: Log overall performance and exit
    total_time = time.time() - start_time
    if psutil and mem_before is not None and cpu_before is not None:
        mem_after = psutil.virtual_memory().used
        cpu_after = psutil.cpu_percent(interval=None)
        logging.info(
            "Script performance metrics: "
            f"Time={total_time:.2f}s, "
            f"MemBefore={mem_before}, MemAfter={mem_after}, "
            f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
        )
    else:
        logging.info(f"Script executed in {total_time:.2f}s")

    logging.info("Script execution completed successfully.")


# --------------------------------------------------------------------
# ### Janis Rubins - Step 3: Helper function to upsert documents
# --------------------------------------------------------------------
def upsert_documents(collection, documents, chunk_size=1000):
    """
    Safely upserts documents to the given collection in batches of 'chunk_size'.
    Enhances robustness for large datasets.
    """
    # ### Janis Rubins - Step 3.1: Log entry with input parameters
    logging.info(
        f"Entering upsert_documents with {len(documents)} documents, "
        f"chunk_size={chunk_size}, collection='{collection.name}'"
    )

    start_time = time.time()
    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    try:
        index = 0
        total_docs = len(documents)
        chunk_index = 0

        # ### Janis Rubins - Step 3.2: Process documents in chunks
        while index < total_docs:
            chunk = documents[index : index + chunk_size]
            ids = [str(i + index) for i in range(len(chunk))]

            # Perform the upsert
            collection.upsert(ids=ids, documents=chunk)
            logging.info(f"Upserted chunk {chunk_index} with {len(chunk)} documents.")

            index += chunk_size
            chunk_index += 1

        # ### Janis Rubins - Step 3.3: Log success
        logging.info(f"All documents upserted into '{collection.name}' successfully.")

    except Exception as e:
        # ### Janis Rubins - Step 3.4: Catch and log any error
        logging.error(f"Failed to upsert documents into '{collection.name}': {e}")
        traceback.print_exc()

    finally:
        # ### Janis Rubins - Step 3.5: Log performance metrics and exit
        execution_time = time.time() - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            logging.info(
                f"upsert_documents performance for '{collection.name}': "
                f"Time={execution_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            logging.info(
                f"upsert_documents for '{collection.name}' took {execution_time:.2f}s"
            )
        logging.info(f"Exiting upsert_documents for collection '{collection.name}'.")


# --------------------------------------------------------------------
# ### Janis Rubins - Step 4: Entry point check
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
