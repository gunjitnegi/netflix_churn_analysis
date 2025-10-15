import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Model path
MODEL_DIR = os.path.join(BASE_DIR, "models")

# SQL path
SQL_DIR = os.path.join(BASE_DIR, "sql")

# Logging setup (optional)
LOG_FILE = os.path.join(BASE_DIR, "project.log")
