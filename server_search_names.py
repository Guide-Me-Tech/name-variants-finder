import logging
import time
import traceback
import os

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Attempt to import psutil for resource usage metrics.
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from dotenv import load_dotenv

from pymilvus import MilvusClient, model
from utils.timer import timer  # If you still want to keep your timing decorator
from utils.printing import printgreen, printred, printblue  # For colored prints
from utils.convert_between_latin_and_cyril import identify_and_convert
from utils.embedddigs import MilvusSearch


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure global logging
# ------------------------------------------------------------------------------
# Explanation:
#  - We configure the logging system so we have consistent timestamps, levels, etc.
#  - Adjust the logging level if you need more or less verbosity (e.g., DEBUG).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Optional helper function for performance & resource usage
# ------------------------------------------------------------------------------
def log_performance_metrics(step_name, start_time):
    """
    Logs how long a step took, optionally capturing CPU/memory usage via psutil.
    """
    end_time = time.time()
    elapsed = end_time - start_time
    if PSUTIL_AVAILABLE:
        mem_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        logging.info(
            f"[{step_name}] Elapsed: {elapsed:.4f}s, "
            f"Memory Used: {mem_info.used / (1024**2):.2f} MB, CPU: {cpu_percent}%"
        )
    else:
        logging.info(f"[{step_name}] Elapsed: {elapsed:.4f}s")


# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Load environment variables and initialize FastAPI
# ------------------------------------------------------------------------------
load_dotenv()
app = FastAPI()
logging.info("FastAPI application instance created.")

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Initialize MilvusSearch client
# ------------------------------------------------------------------------------
start_time = time.time()
try:
    MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_client = MilvusSearch(MILVUS_URI)
    logging.info(f"Milvus client created with URI={MILVUS_URI}")

    # Load the embedding function
    milvus_client.LoadEmbeddingFunction("all-MiniLM-L6-v2", device="cpu")
    logging.info("Embedding function 'all-MiniLM-L6-v2' loaded on 'cpu'.")
except Exception as e:
    logging.error("Failed to initialize Milvus or embed function.")
    traceback.print_exc()
log_performance_metrics("Milvus initialization", start_time)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Prometheus Counter metric
# ------------------------------------------------------------------------------
REQUEST_COUNT = Counter("app_requests_total", "Total number of requests received.")

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 6: /search endpoint
# ------------------------------------------------------------------------------
@app.get("/search/{name_input}")
def search_name(request: Request, name_input: str):
    """
    Splits input name(s) by spaces, detects languages, and searches in
    Uzbek and Russian collections. Returns matching names with scores.
    """
    start_time_route = time.time()
    logging.info(f"Entering search_name endpoint with name_input='{name_input}'")

    REQUEST_COUNT.inc()

    # ### Janis Rubins - Step 6.1: Parse query parameter 'limit'
    try:
        limit = int(request.query_params.get("limit", 10))
        logging.info(f"Parameter 'limit'={limit}")
    except Exception:
        limit = 10
        logging.warning("Unable to parse limit parameter. Defaulted to 10.")

    # ### Janis Rubins - Step 6.2: Prepare the output structure
    output = {
        "uzbek_names": [],
        "russian_names": [],
    }

    # ### Janis Rubins - Step 6.3: Split the input string and process each name
    names = name_input.split(" ")
    res_1 = []
    res_2 = []

    for name in names:
        # ### Janis Rubins - Step 6.3.1: Convert name if needed
        try:
            detected_lang, source_lang, converted_name = identify_and_convert(name)
            logging.info(
                f"Name='{name}', Detected_lang='{detected_lang}', "
                f"Source_lang='{source_lang}', Converted_name='{converted_name}'"
            )
        except Exception as e:
            logging.error(f"Failed to identify_and_convert name='{name}': {e}")
            traceback.print_exc()
            continue  # Skip this name on error

        # ### Janis Rubins - Step 6.3.2: Based on detected language, embed and search
        search_start_time = time.time()
        if detected_lang == "uz":
            _uz_logic(name, converted_name, limit, res_1, res_2, output)
        elif detected_lang == "ru":
            _ru_logic(name, converted_name, limit, res_1, res_2, output)
        else:
            logging.warning(f"Language not recognized. Skipping name='{name}'.")
        log_performance_metrics(f"Search loop for name='{name}'", search_start_time)

    # ### Janis Rubins - Step 6.4: Gather results into the output structure
    # Explanation: We combine all the search results from res_1 (Uzbek) and res_2 (Russian).
    # This merges result sets with distances.
    for i in range(len(res_1)):
        for r in res_1[i]:
            output["uzbek_names"].append(
                {"name": r["entity"]["name"], "score": r["distance"]}
            )
    for i in range(len(res_2)):
        for r in res_2[i]:
            output["russian_names"].append(
                {"name": r["entity"]["name"], "score": r["distance"]}
            )

    # ### Janis Rubins - Step 6.5: Log exit and return result
    logging.info("Exiting search_name endpoint successfully.")
    log_performance_metrics("search_name total time", start_time_route)
    return output

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 7: /embed endpoint
# ------------------------------------------------------------------------------
@app.get("/embed/{name}")
def embed_name(name: str):
    """
    Creates a vector embedding for the requested name.
    """
    start_time_route = time.time()
    logging.info(f"Entering embed_name endpoint with name='{name}'")

    REQUEST_COUNT.inc()

    try:
        vectors = milvus_client.create_embedding_for_word(name)
        logging.info("Embedding generated successfully.")
    except Exception as e:
        logging.error(f"Failed to embed name='{name}': {e}")
        traceback.print_exc()
        return {"error": "Failed to generate embedding"}

    log_performance_metrics("embed_name total time", start_time_route)
    return vectors

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 8: /metrics endpoint for Prometheus
# ------------------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    """
    Exposes the Prometheus metrics for scraping.
    """
    logging.info("Exposing Prometheus metrics at /metrics.")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 9: Helper function for Uzbek logic
# ------------------------------------------------------------------------------
def _uz_logic(name, converted_name, limit, res_1, res_2, output):
    """
    Handles embedding and searching logic for Uzbek-detected names.
    """
    try:
        # ### Janis Rubins - Step 9.1: Embed the name in various forms
        start_embed = time.time()
        embedded_name_1 = milvus_client.create_embedding_for_word(converted_name)
        embedded_name_2 = milvus_client.create_embedding_for_word(name)
        name_changed = name.replace("x", "h").replace("X", "H")
        embedded_name_3 = milvus_client.create_embedding_for_word(name_changed)
        log_performance_metrics("Uzbek name embedding", start_embed)

        # ### Janis Rubins - Step 9.2: Search in Uzbek collection
        start_search_uz = time.time()
        res_1 += milvus_client.search(
            collection_name="uzbek_names",
            data=embedded_name_2,
            filter="",
            limit=limit,
            output_fields=["name"],
        )
        res_1 += milvus_client.search(
            collection_name="uzbek_names",
            data=embedded_name_3,
            filter="",
            limit=limit,
            output_fields=["name"],
        )
        log_performance_metrics("Uzbek name search (uzbek_names)", start_search_uz)

        # ### Janis Rubins - Step 9.3: Search in Russian collection
        start_search_ru = time.time()
        res_2 += milvus_client.search(
            collection_name="rus_names",
            data=embedded_name_1,
            filter="",
            limit=limit,
            output_fields=["name"],
        )
        log_performance_metrics("Uzbek name search (rus_names)", start_search_ru)

        # ### Janis Rubins - Step 9.4: Update output with placeholders
        output["uzbek_names"].append({"name": name, "score": 1})
        output["uzbek_names"].append({"name": name_changed, "score": 1})
        output["russian_names"].append({"name": converted_name, "score": 1})

    except Exception as e:
        logging.error(f"Error in _uz_logic for name='{name}': {e}")
        traceback.print_exc()

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 10: Helper function for Russian logic
# ------------------------------------------------------------------------------
def _ru_logic(name, converted_name, limit, res_1, res_2, output):
    """
    Handles embedding and searching logic for Russian-detected names.
    """
    try:
        # ### Janis Rubins - Step 10.1: Embed the name in various forms
        start_embed = time.time()
        embedded_name_1 = milvus_client.create_embedding_for_word(converted_name)
        embedded_name_2 = milvus_client.create_embedding_for_word(name)
        name_changed = converted_name.replace("x", "h").replace("X", "H")
        embedded_name_3 = milvus_client.create_embedding_for_word(name_changed)
        log_performance_metrics("Russian name embedding", start_embed)

        # ### Janis Rubins - Step 10.2: Search in Uzbek collection
        start_search_uz = time.time()
        res_1 += milvus_client.search(
            collection_name="uzbek_names",
            data=embedded_name_1,
            filter="",
            limit=limit,
            output_fields=["name"],
        )
        res_1 += milvus_client.search(
            collection_name="uzbek_names",
            data=embedded_name_3,
            filter="",
            limit=limit,
            output_fields=["name"],
        )
        log_performance_metrics("Russian name search (uzbek_names)", start_search_uz)

        # ### Janis Rubins - Step 10.3: Search in Russian collection
        start_search_ru = time.time()
        res_2 += milvus_client.search(
            collection_name="rus_names",
            data=embedded_name_2,
            filter="",
            limit=limit,
            output_fields=["name"],
        )
        log_performance_metrics("Russian name search (rus_names)", start_search_ru)

        # ### Janis Rubins - Step 10.4: Update output with placeholders
        output["uzbek_names"].append({"name": converted_name, "score": 1})
        output["uzbek_names"].append({"name": name_changed, "score": 1})
        output["russian_names"].append({"name": name, "score": 1})

    except Exception as e:
        logging.error(f"Error in _ru_logic for name='{name}': {e}")
        traceback.print_exc()
