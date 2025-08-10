import logging
import sys

# Configure logging globally once
logging.basicConfig(
    level=logging.INFO,  # Show INFO and above logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Print logs to CLI console
)

def get_logger(name):
    return logging.getLogger(name)
