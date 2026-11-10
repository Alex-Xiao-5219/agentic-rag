import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _set_env_if_present(name: str) -> None:
    value = os.getenv(name)
    if value:
        os.environ[name] = value


class Settings:
    # --- AZURE OPENAI ---
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    AZURE_OPENAI_ROUTER_DEPLOYMENT = (
        os.getenv("AZURE_OPENAI_ROUTER_DEPLOYMENT")
        or AZURE_OPENAI_CHAT_DEPLOYMENT
    )
    AZURE_OPENAI_REASONING_DEPLOYMENT = (
        os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT")
        or AZURE_OPENAI_CHAT_DEPLOYMENT
    )
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = (
        os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    AZURE_OPENAI_EVAL_DEPLOYMENT = (
        os.getenv("AZURE_OPENAI_EVAL_DEPLOYMENT")
        or AZURE_OPENAI_CHAT_DEPLOYMENT
    )
    AZURE_OPENAI_RERANK_DEPLOYMENT = os.getenv(
        "AZURE_OPENAI_RERANK_DEPLOYMENT"
    ) or AZURE_OPENAI_CHAT_DEPLOYMENT

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"
    
    _set_env_if_present("LANGFUSE_SECRET_KEY")
    _set_env_if_present("LANGFUSE_PUBLIC_KEY")
    _set_env_if_present("LANGFUSE_BASE_URL")

    # --- OBSERVABILITY ---
    # LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    # LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    # LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    # LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
#

settings = Settings()
