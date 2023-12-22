# This is a test script to generate a dataset with the default config file.

# Imports
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Finishing the necessary Imports:
from src.inpainting_dataset_generator import generate_dataset

# Setting the config path to the default config
config_path = "src/inpainting_config.yaml"

# Generating the dataset.
generate_dataset(config_path)