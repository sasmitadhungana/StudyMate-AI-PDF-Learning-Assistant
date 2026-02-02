"""
Configuration for PDF QA Education System
"""
import os
from pathlib import Path

# Application
APP_NAME = "PDF QA Education System"
APP_VERSION = "1.0.0"

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "mixtral-8x7b-32768"
GROQ_TEMPERATURE = 0.7
GROQ_MAX_TOKENS = 1000
GROQ_TIMEOUT = 30

# Available models
AVAILABLE_MODELS = {
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    "llama2-70b-4096": "LLaMA2 70B",
    "gemma-7b-it": "Gemma 7B"
}

# PDF Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_PDF_SIZE_MB = 50

# Semantic Search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.7
TOP_K_RESULTS = 5

# Cache
CACHE_ENABLED = True
CACHE_TTL = 3600

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# UI
MAX_CHAT_HISTORY = 20
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
