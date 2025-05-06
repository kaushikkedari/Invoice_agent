from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from .config import base_llm_config, LLMProvider, BaseLLMConfig
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Optional
 
class AIComponent:
    """Provides a configured LangChain chat model instance based on settings."""
    def __init__(self, llm_config: Optional[BaseLLMConfig] = None):
        self.llm_config = llm_config or base_llm_config
        self.llm: Optional[BaseChatModel] = self._initialize_llm()

        if self.llm is None:
            print("Warning: AIComponent LLM is not initialized. Check PROVIDER in .env and configuration.")
            # Optionally raise an error if an LLM is strictly required
            # raise ValueError("LLM could not be initialized based on the provided configuration.")
        else:
            print(f"AIComponent initialized with provider: {self.llm_config.provider.value}")

    def _initialize_llm(self) -> Optional[BaseChatModel]:
        """Initializes the LangChain chat model based on the configuration."""
        if not self.llm_config.provider:
            return None

        provider = self.llm_config.provider
        temp = self.llm_config.temperature

        try:
            if provider == LLMProvider.AZURE:
                print(f"Initializing AzureChatOpenAI (Deployment: {self.llm_config.azure_chat_deployment_name}, Model: {self.llm_config.azure_chat_model_name})" )
                # Check required Azure vars
                if not all([
                    self.llm_config.azure_api_key,
                    self.llm_config.azure_endpoint,
                    self.llm_config.azure_chat_deployment_name, # Use deployment name
                    self.llm_config.azure_api_version
                ]):
                    raise ValueError("Missing required Azure configuration parameters (API Key, Endpoint, Chat Deployment Name, API Version)")
                
                return AzureChatOpenAI(
                    azure_deployment=self.llm_config.azure_chat_deployment_name,
                    api_version=self.llm_config.azure_api_version,
                    api_key=self.llm_config.azure_api_key,
                    azure_endpoint=self.llm_config.azure_endpoint,
                    model=self.llm_config.azure_chat_model_name,
                    temperature=temp,
                )

            elif provider == LLMProvider.GEMINI:
                print(f"Initializing ChatGoogleGenerativeAI (Model: {self.llm_config.google_chat_model_name})")
                if not self.llm_config.google_api_key:
                     raise ValueError("Missing GOOGLE_API_KEY for Gemini provider.")
                
                return ChatGoogleGenerativeAI(
                    model=self.llm_config.google_chat_model_name,
                    temperature=temp,
                    # max_tokens=None, # Let the model default handle this unless needed
                    # timeout=None,
                    # max_retries=2, # Default is usually sufficient
                    api_key=self.llm_config.google_api_key
                )
            else:
                print(f"Warning: Unknown LLM provider specified: {provider}")
                return None
                
        except ValueError as ve:
             print(f"Configuration Error: {ve}")
             return None
        except ImportError as ie:
             print(f"Import Error: Required library not installed for {provider}. {ie}")
             return None
        except Exception as e:
             print(f"Error initializing LLM provider {provider}: {e}")
             return None

    def invoke(self, messages):
        """Invokes the configured LLM with the provided messages."""
        if not self.llm:
            raise ValueError("LLM is not initialized. Cannot invoke.")
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            # Re-raise or handle as appropriate for your application's error strategy
            raise 

# Optional: Create a default instance for easy import
# try:
#     ai_component = AIComponent()
# except Exception as e:
#     print(f"Failed to create default AIComponent instance: {e}")
#     ai_component = None # Ensure it exists but is None if init fails 