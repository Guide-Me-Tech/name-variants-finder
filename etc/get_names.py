import logging
import time
import traceback

try:
    import psutil  # Optional for resource usage
except ImportError:
    psutil = None

import chromadb
from chromadb.utils import embedding_functions

# ----------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging for the script
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------------------------------------
# ### Janis Rubins - Step 2: Helper function to get or create collections
# ----------------------------------------------------------------------
def get_chroma_collections():
    """
    Initializes ChromaDB, retrieves two specific collections for Uzbek and
    Russian names, and returns them along with the Chroma instance.
    """
    # ### Janis Rubins - Step 2.1: Log entry
    logging.info("Entering get_chroma_collections function.")
    start_time = time.time()

    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    try:
        # ### Janis Rubins - Step 2.2: Create embedding function
        stef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        logging.info("Embedding function created successfully.")

        # ### Janis Rubins - Step 2.3: Initialize persistent Chroma instance
        chroma_instance = chromadb.PersistentClient(path="./names")
        logging.info("ChromaDB persistent client initialized.")

        # ### Janis Rubins - Step 2.4: Retrieve or create Uzbek and Russian name collections
        uz_names_collection = chroma_instance.get_or_create_collection(
            "uz_names", embedding_function=stef
        )
        russian_names_collection = chroma_instance.get_or_create_collection(
            "russian_names", embedding_function=stef
        )
        logging.info("Collections uz_names and russian_names are ready.")

        return chroma_instance, uz_names_collection, russian_names_collection

    except Exception as e:
        logging.error(f"Error during ChromaDB collections setup: {e}")
        traceback.print_exc()
        return None, None, None  # Graceful degradation

    finally:
        # ### Janis Rubins - Step 2.5: Log performance metrics and exit
        execution_time = time.time() - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            logging.info(
                "get_chroma_collections performance: "
                f"Time={execution_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            logging.info(f"get_chroma_collections took {execution_time:.2f}s")

        logging.info("Exiting get_chroma_collections.")

# ----------------------------------------------------------------------
# ### Janis Rubins - Step 3: Main logic for user input and queries
# ----------------------------------------------------------------------
def main():
    """
    Main function that sets up ChromaDB collections, prompts for a name query,
    and performs queries on each collection, printing the results.
    """
    # ### Janis Rubins - Step 3.1: Log script start
    logging.info("Script execution started.")
    start_time = time.time()

    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = None
        cpu_before = None

    # ### Janis Rubins - Step 3.2: Retrieve ChromaDB collections
    chroma_instance, uz_names_collection, russian_names_collection = get_chroma_collections()
    if not chroma_instance or not uz_names_collection or not russian_names_collection:
        logging.error("ChromaDB collections not available. Exiting.")
        return  # Cannot proceed further

    # ### Janis Rubins - Step 3.3: Prompt user for input and query collections
    try:
        name_query = input("Enter a name: ").strip()
        logging.info(f"User provided name query: {name_query}")

        if not name_query:
            raise ValueError("No name was entered. Please enter a valid name.")

        # ### Janis Rubins - Step 3.4: Query Uzbek names
        uz_query_start = time.time()
        uz_names = uz_names_collection.query(query_texts=[name_query], n_results=10)
        uz_query_time = time.time() - uz_query_start
        logging.info(
            f"Queried uz_names_collection in {uz_query_time:.2f}s; result size: {len(uz_names.get('ids', []))}"
        )

        # ### Janis Rubins - Step 3.5: Query Russian names
        ru_query_start = time.time()
        russian_names = russian_names_collection.query(query_texts=[name_query], n_results=10)
        ru_query_time = time.time() - ru_query_start
        logging.info(
            f"Queried russian_names_collection in {ru_query_time:.2f}s; result size: {len(russian_names.get('ids', []))}"
        )

        # ### Janis Rubins - Step 3.6: Print or process results
        print("UZ names query results:")
        print(uz_names.items())  # Original logic
        print("Russian names query results:")
        print(russian_names.items())  # Original logic

    except ValueError as e:
        # Handle empty or invalid input
        logging.error(f"ValueError: {e}")
    except Exception as e:
        # Catch any unexpected errors during input or querying
        logging.error(f"Error during name query: {e}")
        traceback.print_exc()

    finally:
        # ### Janis Rubins - Step 3.7: Log overall performance and exit
        total_execution_time = time.time() - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            logging.info(
                "Script overall performance: "
                f"Time={total_execution_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            logging.info(f"Script executed in {total_execution_time:.2f}s")

        logging.info("Script execution completed successfully.")

# ----------------------------------------------------------------------
# ### Janis Rubins - Step 4: Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
