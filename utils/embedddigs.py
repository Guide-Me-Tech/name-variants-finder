import logging
import time
import traceback

try:
    import psutil  # For optional resource usage tracking
except ImportError:
    psutil = None

from pymilvus import MilvusClient, model

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Define a helper function for resource usage
# ------------------------------------------------------------------------------
def log_resource_usage():
    """
    Logs system resource usage if `psutil` is installed.
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
# ### Janis Rubins - Step 3: Define a helper function to prepare data
# ------------------------------------------------------------------------------
def PrepareData(names, embeddings) -> dict:
    """
    Takes lists of names and their corresponding embeddings, and returns a list
    of dictionaries containing 'id', 'vector', and 'name'.
    """
    # ### Janis Rubins - Step 3.1: Log entry with input size
    logging.info(
        f"Entering PrepareData with {len(names)} names and {len(embeddings)} embeddings."
    )
    start_time = time.time()

    log_resource_usage()

    try:
        data = []
        for i in range(len(names)):
            data.append(
                {
                    "id": i,
                    "vector": embeddings[i],
                    "name": names[i]
                }
            )

        logging.info(f"PrepareData successfully created {len(data)} data items.")
        return data

    except Exception as e:
        # ### Janis Rubins - Step 3.2: Log exception
        logging.error(f"Error in PrepareData: {e}")
        traceback.print_exc()
        return []

    finally:
        # ### Janis Rubins - Step 3.3: Performance logging
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"PrepareData completed in {duration:.2f}s")
        log_resource_usage()


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Define the MilvusSearch class, extending MilvusClient
# ------------------------------------------------------------------------------
class MilvusSearch(MilvusClient):
    """
    Extends MilvusClient to manage collections, embeddings, and searching
    for text-based data.
    """
    # ### Janis Rubins - Step 4.1: Override __init__ to initialize additional attributes
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        timeout: float | None = None,
        **kwargs
    ) -> None:
        logging.info(
            "Entering MilvusSearch.__init__ "
            f"with uri={uri}, user={user}, db_name={db_name}, timeout={timeout}"
        )
        start_time = time.time()

        super().__init__(uri, user, password, db_name, token, timeout, **kwargs)
        self.sentence_transformer_ef = None

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"MilvusSearch.__init__ completed in {duration:.2f}s")

    # ------------------------------------------------------------------------------
    # ### Janis Rubins - Step 4.2: Create or reset a collection
    # ------------------------------------------------------------------------------
    def insert_names(self, names, collection_name, dimension=384):
        """
        Drops collection if it already exists, then recreates it with the specified dimension.
        """
        logging.info(
            f"Entering insert_names with collection_name={collection_name}, dimension={dimension}"
        )
        start_time = time.time()
        log_resource_usage()

        try:
            # ### Janis Rubins - Step 4.2.1: Check if collection exists and drop if needed
            if self.has_collection(collection_name=collection_name):
                logging.info(f"Collection '{collection_name}' exists, dropping it.")
                self.drop_collection(collection_name=collection_name)
            else:
                logging.info(f"Collection '{collection_name}' does not exist, creating new collection.")

            # ### Janis Rubins - Step 4.2.2: Create collection
            self.create_collection(
                collection_name=collection_name,
                dimension=dimension,
            )
            logging.info(f"Collection '{collection_name}' created successfully.")

        except Exception as e:
            logging.error(f"Error in insert_names for collection '{collection_name}': {e}")
            traceback.print_exc()

        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"insert_names completed in {duration:.2f}s for '{collection_name}'")
            log_resource_usage()

    # ------------------------------------------------------------------------------
    # ### Janis Rubins - Step 4.3: Load embedding function
    # ------------------------------------------------------------------------------
    def LoadEmbeddingFunction(self, model_name, device="cpu"):
        """
        Loads a dense SentenceTransformerEmbeddingFunction from pymilvus.model
        using the specified model_name and device.
        """
        logging.info(
            f"Entering LoadEmbeddingFunction with model_name={model_name}, device={device}"
        )
        start_time = time.time()
        log_resource_usage()

        try:
            self.sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                device=device,
            )
            logging.info("SentenceTransformerEmbeddingFunction loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading embedding function with model '{model_name}': {e}")
            traceback.print_exc()
            # Could set self.sentence_transformer_ef = None or raise again

        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"LoadEmbeddingFunction completed in {duration:.2f}s")
            log_resource_usage()

    # ------------------------------------------------------------------------------
    # ### Janis Rubins - Step 4.4: Create embeddings for a single word
    # ------------------------------------------------------------------------------
    def create_embedding_for_word(self, word) -> list:
        """
        Encodes a single word using the loaded embedding function.
        Returns a list containing one embedding vector.
        """
        logging.info(f"Entering create_embedding_for_word with word='{word}'")
        start_time = time.time()
        log_resource_usage()

        if not self.sentence_transformer_ef:
            error_msg = "Embedding function not loaded. Call LoadEmbeddingFunction first."
            logging.error(error_msg)
            return []

        try:
            vectors = self.sentence_transformer_ef.encode_documents([word])
            logging.info(f"Embedding created successfully for word='{word}'")
            return vectors
        except Exception as e:
            logging.error(f"Error creating embedding for word='{word}': {e}")
            traceback.print_exc()
            return []
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"create_embedding_for_word completed in {duration:.2f}s")
            log_resource_usage()
