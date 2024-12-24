import logging
import time
import traceback

try:
    import psutil  # For optional resource usage tracking
except ImportError:
    psutil = None

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 1: Configure logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 2: Define ANSI color codes
# ------------------------------------------------------------------------------
reset = "\033[0m"
black = "\033[30m"
red = "\033[31m"
green = "\033[32m"
orange = "\033[33m"
blue = "\033[34m"
purple = "\033[35m"
cyan = "\033[36m"
lightgrey = "\033[37m"
darkgrey = "\033[90m"
lightred = "\033[91m"
lightgreen = "\033[92m"
yellow = "\033[93m"
lightblue = "\033[94m"
pink = "\033[95m"
lightcyan = "\033[96m"

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 3: Optional function to log resource usage
# ------------------------------------------------------------------------------
def log_resource_usage():
    """
    Logs system resource usage if psutil is installed.
    """
    if psutil:
        mem_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        logging.info(
            f"Resource usage -> Memory: {mem_info.used / (1024 * 1024):.2f} MB / "
            f"{mem_info.total / (1024 * 1024):.2f} MB, CPU: {cpu_percent:.2f}%"
        )
    else:
        logging.info("psutil not installed. Skipping resource usage logging.")

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 4: Internal helper for printing with color and logging
# ------------------------------------------------------------------------------
def _print_with_color(color_code, text):
    """
    Prints the given text using the specified color code, with error handling,
    performance tracking, and logging.
    """
    # ### Janis Rubins - Step 4.1: Log entry, parameters, and measure time
    logging.info(f"Entering _print_with_color with color_code='{color_code}' and text='{text}'")
    start_time = time.time()
    log_resource_usage()

    try:
        # ### Janis Rubins - Step 4.2: Attempt to print the colored text
        print(f"{color_code}{text}{reset}")
        logging.info("Print operation succeeded.")
    except Exception as e:
        # ### Janis Rubins - Step 4.3: Log error with stack trace
        logging.error(f"Failed to print color text: {e}")
        traceback.print_exc()
    finally:
        # ### Janis Rubins - Step 4.4: Log exit and performance
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"_print_with_color completed in {duration:.2f}s")
        log_resource_usage()
        logging.info("Exiting _print_with_color.")

# ------------------------------------------------------------------------------
# ### Janis Rubins - Step 5: Public printing functions
# ------------------------------------------------------------------------------
def printred(text):
    """Prints text in red."""
    _print_with_color(red, text)

def printgreen(text):
    """Prints text in green."""
    _print_with_color(green, text)

def printorange(text):
    """Prints text in orange."""
    _print_with_color(orange, text)

def printblue(text):
    """Prints text in blue."""
    _print_with_color(blue, text)

def printpurple(text):
    """Prints text in purple."""
    _print_with_color(purple, text)

def printcyan(text):
    """Prints text in cyan."""
    _print_with_color(cyan, text)

def printlightgrey(text):
    """Prints text in light grey."""
    _print_with_color(lightgrey, text)

def printdarkgrey(text):
    """Prints text in dark grey."""
    _print_with_color(darkgrey, text)

def printlightred(text):
    """Prints text in light red."""
    _print_with_color(lightred, text)

def printlightgreen(text):
    """Prints text in light green."""
    _print_with_color(lightgreen, text)

def printyellow(text):
    """Prints text in yellow."""
    _print_with_color(yellow, text)

def printlightblue(text):
    """Prints text in light blue."""
    _print_with_color(lightblue, text)

def printpink(text):
    """Prints text in pink."""
    _print_with_color(pink, text)

def printlightcyan(text):
    """Prints text in light cyan."""
    _print_with_color(lightcyan, text)
