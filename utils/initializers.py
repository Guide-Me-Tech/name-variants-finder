import os
import json
import logging
import time
import traceback

try:
    import psutil  # For optional resource usage tracking
except ImportError:
    psutil = None

import dotenv
import orjson
import chromadb
from chromadb.db.base import UniqueConstraintError
from chromadb.utils import embedding_functions
from llms.format_with_chatgpt import format_actions_sequence
import utils.printing

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
    Logs system resource usage if psutil is installed.
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
# ### Janis Rubins - Step 3: Helper function to load environment variables
# ------------------------------------------------------------------------------
def load_envs():
    """
    Loads environment variables from a .env file.
    """
    logging.info("Loading environment variables from .env.")
    dotenv.load_dotenv(verbose=True)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Maintain a single global Chroma client
# ------------------------------------------------------------------------------
_CHROMA_CLIENT = None  # Will be initialized once

def get_chroma_client():
    """
    Returns a globally-shared ChromaDB PersistentClient instance.
    Prevents recreating it for every user.
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        logging.info("Creating a new global ChromaDB PersistentClient instance.")
        _CHROMA_CLIENT = chromadb.PersistentClient("./user_files/chromadb")
        logging.info("ChromaDB PersistentClient created and cached.")
    return _CHROMA_CLIENT

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: The main class for formatting user actions
# ------------------------------------------------------------------------------
class OpenedActionsFormatter:
    """
    Responsible for retrieving user-specific collections and formatting results.
    """

    # ### Janis Rubins - Step 5.1: Initialization
    def __init__(self):
        logging.info("OpenedActionsFormatter instantiated.")
        self.collection = None

    # ### Janis Rubins - Step 5.2: Retrieve or create user-specific collection
    def GetChroma(self, username: str):
        """
        Attempts to retrieve a user-specific ChromaDB collection. 
        If not found, it logs an error or handles it gracefully.
        """
        start_time = time.time()
        logging.info(f"GetChroma called with username='{username}'")
        log_resource_usage()

        chroma_client = get_chroma_client()
        collection_name = f"{username}_actions_collection"
        logging.info(f"Using collection name: {collection_name}")

        try:
            collection = chroma_client.get_collection(collection_name)
            self.collection = collection
            logging.info(f"Collection '{collection_name}' retrieved successfully.")
        except Exception as e:
            # If collection doesn't exist or can't be accessed, handle gracefully
            logging.error(f"Failed to retrieve collection '{collection_name}': {e}")
            traceback.print_exc()
            self.collection = None

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"GetChroma completed in {duration:.2f}s")
        log_resource_usage()

    # ### Janis Rubins - Step 5.3: Query the user's collection
    def Query(self, query_string: str):
        """
        Queries the currently selected collection with the provided query_string.
        Returns ChromaDB search results.
        """
        start_time = time.time()
        logging.info(f"Query called with query_string='{query_string}'")
        log_resource_usage()

        if not self.collection:
            error_msg = "No collection available. Call GetChroma first."
            logging.error(error_msg)
            return {}

        try:
            results = self.collection.query(query_texts=[query_string], n_results=1)
            return results
        except Exception as e:
            logging.error(f"Error querying collection: {e}")
            traceback.print_exc()
            return {}
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Query completed in {duration:.2f}s")
            log_resource_usage()

    # ### Janis Rubins - Step 5.4: Format results into JSON
    def Format(self, results):
        """
        Takes the results of a query, extracts the first set of documents,
        and returns them as an orjson-dumped string.
        """
        start_time = time.time()
        logging.info("Format called to transform query results into JSON.")
        log_resource_usage()

        try:
            # Expecting results in the form results["documents"][0]
            docs = results.get("documents", [])
            if docs:
                doc_list = docs[0]
                doc_json = orjson.dumps(doc_list)
                return doc_json
            else:
                return b"[]"
        except Exception as e:
            logging.error(f"Error formatting results: {e}")
            traceback.print_exc()
            return b"[]"
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Format completed in {duration:.2f}s")
            log_resource_usage()

    # ### Janis Rubins - Step 5.5: Format results for a general answer with ChatGPT
    def FormatToGeneralAnswer(self, results, user_message):
        """
        Converts the top query result into a user-friendly action sequence
        by calling an external format_actions_sequence function.
        """
        start_time = time.time()
        logging.info("FormatToGeneralAnswer called to produce a user-friendly action sequence.")
        log_resource_usage()

        try:
            docs = results.get("documents", [[]])
            if not docs or not docs[0]:
                logging.warning("No documents found in results.")
                return {}, "No actions found."

            first_doc = docs[0][0]  # The first doc of the first list
            o = orjson.loads(first_doc)
            message, response = format_actions_sequence(o, user_message=user_message)
            if "action" in message.content[:10]:
                return orjson.loads(message.content[8:]), response
            return orjson.loads(message.content), response
        except Exception as e:
            logging.error(f"Error in FormatToGeneralAnswer: {e}")
            traceback.print_exc()
            return {}, "Error processing actions."
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"FormatToGeneralAnswer completed in {duration:.2f}s")
            log_resource_usage()

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 6: A revised trainer for storing user actions in Chroma
# ------------------------------------------------------------------------------
class TrainActionsBotv2:
    """
    Stores a list of user actions and persists them to a user-specific collection.
    """

    # ### Janis Rubins - Step 6.1: Initialize with empty actions
    def __init__(self):
        logging.info("TrainActionsBotv2 instantiated.")
        self.actions = []
        self.username = None

    # ### Janis Rubins - Step 6.2: Set a list of actions
    def SetActions(self, actions):
        logging.info(f"SetActions called with {len(actions)} actions.")
        self.actions = actions

    # ### Janis Rubins - Step 6.3: Add a single action
    def AddAction(self, action):
        logging.info(f"AddAction called with action={action}")
        self.actions.append(action)

    # ### Janis Rubins - Step 6.4: Set the username
    def SetUsername(self, username):
        logging.info(f"SetUsername called with username={username}")
        self.username = username

    # ### Janis Rubins - Step 6.5: Train (save) actions to Chroma
    def TrainandSave(self, sentence_transformer_ef):
        """
        Creates or recreates a user-specific collection, then inserts all actions.
        """
        start_time = time.time()
        logging.info("TrainandSave called to persist actions in ChromaDB.")
        log_resource_usage()

        if not self.username:
            error_msg = "No username set. Call SetUsername first."
            logging.error(error_msg)
            return "Username not set."

        # Obtain the single global Chroma client
        chroma_client = get_chroma_client()
        collection_name = self.username + "_custom_collection"  # Changed from old "demo_collection"
        logging.info(f"Collection name: {collection_name}")

        try:
            collection = chroma_client.create_collection(
                collection_name,
                embedding_function=sentence_transformer_ef,
            )
            logging.info(f"Collection '{collection_name}' created successfully.")
        except UniqueConstraintError:
            # If it already exists, we drop it and recreate
            logging.warning(
                f"UniqueConstraintError encountered. Dropping and recreating '{collection_name}'."
            )
            chroma_client.delete_collection(collection_name)
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef,
            )
            logging.info(f"Recreated collection '{collection_name}' after deletion.")
        except Exception as e:
            logging.error(f"Error creating collection '{collection_name}': {e}")
            traceback.print_exc()
            return "Error creating collection."

        # Add actions
        try:
            ids = [str(i) for i in range(len(self.actions))]
            documents = [json.dumps(i) for i in self.actions]
            collection.add(documents=documents, ids=ids)
            logging.info(f"Inserted {len(self.actions)} actions into '{collection_name}'.")
        except Exception as e:
            logging.error(f"Error inserting actions into '{collection_name}': {e}")
            traceback.print_exc()
            return "Error inserting actions."

        # Log final performance
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"TrainandSave completed in {duration:.2f}s")
        log_resource_usage()
        return "done"

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 7: Legacy fallback training bot
# ------------------------------------------------------------------------------
class TrainActionsBot:
    """
    A legacy version of TrainActionsBot for backward compatibility.
    """

    # ### Janis Rubins - Step 7.1: Initialize
    def __init__(self):
        logging.info("TrainActionsBot instantiated (legacy).")
        self.actions = []
        self.username = None

    # ### Janis Rubins - Step 7.2: Set actions
    def SetActions(self, actions):
        logging.info(f"TrainActionsBot SetActions called with {len(actions)} actions.")
        self.actions = actions

    # ### Janis Rubins - Step 7.3: Add single action
    def AddAction(self, action):
        logging.info(f"TrainActionsBot AddAction called with action={action}")
        self.actions.append(action)

    # ### Janis Rubins - Step 7.4: Set username
    def SetUsername(self, username):
        logging.info(f"TrainActionsBot SetUsername called with username={username}")
        self.username = username

    # ### Janis Rubins - Step 7.5: Train and save with the same approach
    def TrainandSave(self, sentence_transformer_ef):
        """
        Creates or recreates a user-specific collection, then inserts all actions.
        Maintains older naming style for compatibility.
        """
        start_time = time.time()
        logging.info("TrainActionsBot TrainandSave called (legacy).")
        log_resource_usage()

        if not self.username:
            error_msg = "No username set. Call SetUsername first."
            logging.error(error_msg)
            return "Username not set."

        chroma_client = get_chroma_client()
        collection_name = self.username + "_custom_collection"
        logging.info(f"Collection name (legacy): {collection_name}")

        try:
            collection = chroma_client.create_collection(
                collection_name,
                embedding_function=sentence_transformer_ef,
            )
            logging.info(f"Legacy collection '{collection_name}' created successfully.")
        except UniqueConstraintError:
            logging.warning(
                f"UniqueConstraintError for legacy collection '{collection_name}'. Dropping and recreating."
            )
            chroma_client.delete_collection(collection_name)
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef,
            )
            logging.info(f"Legacy collection '{collection_name}' recreated.")
        except Exception as e:
            logging.error(f"Error creating legacy collection '{collection_name}': {e}")
            traceback.print_exc()
            return "Error creating collection."

        try:
            ids = [str(i) for i in range(len(self.actions))]
            documents = [json.dumps(i) for i in self.actions]
            collection.add(documents=documents, ids=ids)
            logging.info(f"Inserted {len(self.actions)} actions into legacy '{collection_name}'.")
        except Exception as e:
            logging.error(f"Error inserting actions into legacy '{collection_name}': {e}")
            traceback.print_exc()
            return "Error inserting actions."

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"TrainandSave (legacy) completed in {duration:.2f}s")
        log_resource_usage()
        return "done"
