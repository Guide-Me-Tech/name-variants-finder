import time
from utils.printing import green, red, reset

def timer(func):
    """
    A decorator that measures the execution time of the decorated function
    and prints the result, using custom color settings from utils.printing.
    """
    # ### Janis Rubins - Step 1: Define a nested wrapper function
    def wrapper(*args, **kwargs):
        # ### Janis Rubins - Step 2: Capture the start time
        start_time = time.time()

        # ### Janis Rubins - Step 3: Execute the original function
        result = func(*args, **kwargs)

        # ### Janis Rubins - Step 4: Capture the end time and calculate duration
        end_time = time.time()
        elapsed_time = end_time - start_time

        # ### Janis Rubins - Step 5: Print the timing result with color formatting
        print(
            f" {green}{func.__name__} took to complete: "
            f"{red}{elapsed_time:.6f} seconds{reset}"
        )

        # ### Janis Rubins - Step 6: Return the result of the original function
        return result

    # ### Janis Rubins - Step 7: Return the wrapper so it can be used as a decorator
    return wrapper
