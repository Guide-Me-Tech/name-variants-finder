import os
import logging
import time
import traceback
import warnings

try:
    import psutil  # Optional for resource usage
except ImportError:
    psutil = None

import tensorflow as tf
import keras  # for keras.layers.TFSMLayer

# ### Janis Rubins - Step 1: Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ### Janis Rubins - Step 2: Suppress TensorFlow and other warnings
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

# ### Janis Rubins - Step 3: Define optional resource usage logging
def log_resource_usage():
    """
    Logs CPU and memory usage if psutil is installed.
    """
    if psutil:
        mem_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        logging.info(
            f"Resource usage -> Memory: {mem_info.used / (1024 * 1024):.2f} MB / "
            f"{mem_info.total / (1024 * 1024):.2f} MB, CPU: {cpu_percent:.2f}%"
        )
    else:
        logging.info("psutil not installed; skipping resource usage logging.")

# ### Janis Rubins - Step 4: Define global constants for model usage
maxlen = 500     # Maximum sequence length (not currently used, retained for compatibility)
batch_size = 64  # Batch size used in training or inference (not currently used, retained for compatibility)

# ### Janis Rubins - Step 5: Load TensorFlow model function
def load_model():
    """
    Loads a TFSMLayer model from the specified path, returning the model object.
    """
    # ### Janis Rubins - Step 5.1: Log function entry
    logging.info("Entering load_model function.")
    start_time = time.time()
    log_resource_usage()

    model = None
    model_path = "sentence_prediction/trained/tf/saved_model"

    try:
        # ### Janis Rubins - Step 5.2: Attempt to load the model
        logging.info(f"Attempting to load model from: {model_path}")
        model = keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
        logging.info("Model loaded successfully.")

    except Exception as e:
        # ### Janis Rubins - Step 5.3: Log error with stack trace
        logging.error(f"Error loading model from '{model_path}': {e}")
        traceback.print_exc()

    finally:
        # ### Janis Rubins - Step 5.4: Log performance and resource usage
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"load_model completed in {duration:.2f}s")
        log_resource_usage()
        logging.info("Exiting load_model function.")

    return model
