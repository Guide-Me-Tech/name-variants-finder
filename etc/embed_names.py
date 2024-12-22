import os
import time
import logging
import traceback

try:
    import psutil  # For resource usage if available
except ImportError:
    psutil = None

import chromadb
from chromadb.utils import embedding_functions


# --------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure the logging system with a clear format
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Define the files containing UZ and Russian names
# --------------------------------------------------------------------------
FILENAMES = [
    "uzname-men-name-lat-v1.1b 2.txt",
    "uzname-women-name-lat-v1.1b.txt",
    "russian_male_names.txt",
    "russian_female_names.txt",
]

# --------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Prepare containers for names
# --------------------------------------------------------------------------
uz_names = []
russian_names = []


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Helper function to read lines from a file safely
# ------------------------------------------------------------------------------
def read_names_from_file(filepath):
    """
    Reads names from a given file, stripping whitespace.
    Returns a list of names or raises an exception on error.
    """
    # ### Janis Rubins - Step 4.1: Log entry with input parameters
    logging.info(f"Entering read_names_from_file with filepath={filepath}")
    
    start_time = time.time()
    names = []

    # Optionally track resource usage
    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = cpu_before = None

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file '{filepath}' does not exist.")

        with open(filepath, "r", encoding="utf-8") as file:
            raw_names = file.readlines()
            names = [name.strip() for name in raw_names]

        # ### Janis Rubins - Step 4.2: Log success
        logging.info(f"Successfully read {len(names)} names from {filepath}")

    except FileNotFoundError as fnf_err:
        # ### Janis Rubins - Step 4.3: Log file-not-found error
        logging.error(f"FileNotFoundError while reading {filepath}: {fnf_err}")
        traceback.print_exc()

    except Exception as ex:
        # ### Janis Rubins - Step 4.4: Catch-all for unexpected errors
        logging.error(f"Unexpected error while reading {filepath}: {ex}")
        traceback.print_exc()

    finally:
        execution_time = time.time() - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            # ### Janis Rubins - Step 4.5: Log performance metrics
            logging.info(
                "read_names_from_file performance: "
                f"Time={execution_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            # ### Janis Rubins - Step 4.6: Log fallback metrics
            logging.info(f"read_names_from_file took {execution_time:.2f}s")

        # ### Janis Rubins - Step 4.7: Log exit with results
        logging.info(f"Exiting read_names_from_file with {len(names)} names loaded")

    return names


# -----------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Helper function to chunk and upsert documents into Chroma
# -----------------------------------------------------------------------------------
def chunk_and_upsert(collection, documents, chunk_size=5461):
    """
    Splits a list of documents into manageable chunks and upserts them
    into the given ChromaDB collection. Logs each step comprehensively.
    """
    # ### Janis Rubins - Step 5.1: Log entry with input parameters
    logging.info(
        f"Entering chunk_and_upsert with {len(documents)} documents, "
        f"chunk_size={chunk_size}, collection={collection.name}"
    )

    start_time = time.time()

    # Optionally track resource usage
    if psutil:
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=None)
    else:
        mem_before = cpu_before = None

    try:
        total_docs = len(documents)
        index = 0
        chunk_index = 0
        
        while index < total_docs:
            chunk = documents[index : index + chunk_size]
            chunk_len = len(chunk)
            
            # ### Janis Rubins - Step 5.2: Upsert with ID mapping
            # For better uniqueness across chunks, we could offset IDs by index
            # but for simplicity we rely on the chunk index for now.
            ids = [f"{chunk_index}-{i}" for i in range(chunk_len)]
            
            collection.upsert(ids=ids, documents=chunk)
            
            logging.info(
                f"Upserted chunk {chunk_index} of size {chunk_len} "
                f"into collection '{collection.name}'"
            )
            
            index += chunk_size
            chunk_index += 1

        # ### Janis Rubins - Step 5.3: Log success
        logging.info(f"All documents upserted successfully into '{collection.name}'")

    except Exception as ex:
        # ### Janis Rubins - Step 5.4: Catch and log errors during upsert
        logging.error(f"Failed to upsert documents into '{collection.name}' due to: {ex}")
        traceback.print_exc()

    finally:
        execution_time = time.time() - start_time
        if psutil and mem_before is not None and cpu_before is not None:
            mem_after = psutil.virtual_memory().used
            cpu_after = psutil.cpu_percent(interval=None)
            # ### Janis Rubins - Step 5.5: Log performance metrics
            logging.info(
                "chunk_and_upsert performance: "
                f"Time={execution_time:.2f}s, "
                f"MemBefore={mem_before}, MemAfter={mem_after}, "
                f"CPUBefore={cpu_before}%, CPUAfter={cpu_after}%"
            )
        else:
            # ### Janis Rubins - Step 5.6: Log fallback metrics
            logging.info(f"chunk_and_upsert took {execution_time:.2f}s")

        # ### Janis Rubins - Step 5.7: Log exit
        logging.info("Exiting chunk_and_upsert")


# ---------------------------------------------------------------------------------------------
# ### Janis Rubins - Step 6: Main function to orchestrate reading files and upserting into Chroma
# ---------------------------------------------------------------------------------------------
def main():
    # ### Janis Rubins - Step 6.1: Log script start
    logging.info("Script execution started.")

    # ### Janis Rubins - Step 6.2: Load names from files, preserving original logic
    # Load two UZ name files
    uz_names_part1 = read_names_from_file(FILENAMES[0])
    uz_names_part2 = read_names_from_file(FILENAMES[1])
    
    # Load two Russian name files
    russian_names_part1 = read_names_from_file(FILENAMES[2])
    russian_names_part2 = read_names_from_file(FILENAMES[3])

    # Combine loaded data
    uz_names_all = uz_names_part1 + uz_names_part2
    russian_names_all = russian_names_part1 + russian_names_part2

    # ### Janis Rubins - Step 6.3: Initialize embedding function
    stef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # ### Janis Rubins - Step 6.4: Create or retrieve ChromaDB collections
    # We wrap in try/except for robust error handling
    try:
        chroma_instance = chromadb.PersistentClient(path="./names")
        
        uz_names_collection = chroma_instance.get_or_create_collection(
            "uz_names",
            embedding_function=stef
        )
        
        russian_names_collection = chroma_instance.get_or_create_collection(
            "russian_names",
            embedding_function=stef
        )
        
        logging.info("ChromaDB collections obtained successfully.")

    except Exception as ex:
        logging.error(f"Failed to initialize ChromaDB or collections: {ex}")
        traceback.print_exc()
        return  # Graceful degradation: exit if we cannot proceed

    # ### Janis Rubins - Step 6.5: Upsert names in chunks
    # Using our chunk_and_upsert helper for better logging and error handling
    chunk_and_upsert(uz_names_collection, uz_names_all, chunk_size=5461)
    chunk_and_upsert(russian_names_collection, russian_names_all, chunk_size=5461)

    # ### Janis Rubins - Step 6.6: Final log, indicating success
    logging.info("Script execution completed successfully.")


# ----------------------------------------------------------------------
# ### Janis Rubins - Step 7: Entry point check and call to main function
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
