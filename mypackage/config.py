from enum import Enum
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent.parent # Adjusted BASE_DIR to point to project root
 
class LLMProvider(str, Enum):
    AZURE = 'azure'
    GEMINI = 'gemini'
 
class BaseLLMConfig(BaseSettings):
    # Renamed azure_model_name to azure_chat_model_name for clarity
    # Added azure_embeddings_deployment_name
    # Added azure_vision_deployment_name
    provider: Optional[LLMProvider] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_chat_model_name: Optional[str]  # Model identifier for chat
    azure_chat_deployment_name: Optional[str] = None # Deployment name for chat
    azure_api_version: Optional[str] = None
    google_chat_model_name: Optional[str] = "gemini-1.5-flash-latest" # Default gemini chat model
    google_api_key: Optional[str] = None
    temperature: float = 0.1
 
    class Config:
        # Load from .env file in the project root (BASE_DIR)
        env_file = BASE_DIR / ".env" 
        env_file_encoding = "utf-8"
        case_sensitive = False # Env vars are usually case-insensitive
        extra = "ignore"
 
# Load the configuration
base_llm_config = BaseLLMConfig()
 
if __name__ == "__main__":
    # Use model_dump_json for newer pydantic versions
    print(base_llm_config.model_dump_json(indent=2)) 