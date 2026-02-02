"""
Configuration Management for PDF Q&A System
"""
import os
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"


class AppConfig(BaseSettings):
    """Main application configuration"""
    
    # ============ Groq Configuration ============
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="llama3-70b-8192")
    GROQ_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=1.0)
    GROQ_MAX_TOKENS: int = Field(default=1000, ge=100, le=4000)
    GROQ_TIMEOUT: int = Field(default=30)
    
    # Available models
    AVAILABLE_MODELS: Dict[str, str] = {
        "llama3-70b-8192": "Llama 3.1 70B (Most capable)",
        "llama3-8b-8192": "Llama 3.1 8B (Fast)",
        "mixtral-8x7b-32768": "Mixtral 8x7B (Expert)",
        "gemma2-9b-it": "Gemma 2 9B (Google)"
    }
    
    # ============ PDF Processing ============
    PDF_CHUNK_SIZE: int = Field(default=1000)
    PDF_CHUNK_OVERLAP: int = Field(default=200)
    MAX_PDF_SIZE_MB: int = Field(default=50)
    ALLOWED_EXTENSIONS: list = Field(default=["pdf"])
    
    # ============ Application Settings ============
    APP_NAME: str = Field(default="PDF Q&A Education System")
    APP_VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=False)
    
    # ============ File Storage ============
    UPLOAD_DIR: Path = Field(default=DATA_DIR / "uploads")
    CACHE_DIR: Path = Field(default=DATA_DIR / "cache")
    LOG_DIR: Path = Field(default=LOGS_DIR)
    
    # ============ Security ============
    SECRET_KEY: str = Field(default=os.urandom(32).hex())
    ALLOWED_ORIGINS: list = Field(default=["http://localhost:8501", "http://127.0.0.1:8501"])
    
    # ============ Performance ============
    MAX_CONCURRENT_UPLOADS: int = Field(default=5)
    CACHE_TTL: int = Field(default=3600)  # 1 hour
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_PERIOD: int = Field(default=3600)  # 1 hour
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("UPLOAD_DIR", "CACHE_DIR", "LOG_DIR")
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("GROQ_API_KEY")
    def validate_api_key(cls, v: str) -> str:
        """Validate Groq API key format"""
        if not v or v == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY must be set in .env file")
        if len(v) < 20:
            raise ValueError("Invalid Groq API key format")
        return v
    
    def get_model_description(self, model_name: str) -> Optional[str]:
        """Get description for a model"""
        return self.AVAILABLE_MODELS.get(model_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)"""
        data = self.dict()
        # Hide sensitive information
        if "GROQ_API_KEY" in data:
            data["GROQ_API_KEY"] = "***" + data["GROQ_API_KEY"][-4:]
        if "SECRET_KEY" in data:
            data["SECRET_KEY"] = "***" + data["SECRET_KEY"][-4:]
        return data


# Create configuration instance
config = AppConfig()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_DIR / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"{config.APP_NAME} v{config.APP_VERSION} initialized")
logger.info(f"Using model: {config.GROQ_MODEL}")
