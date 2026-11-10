import logfire
from langchain_openai import AzureChatOpenAI
from langfuse.openai import openai

from app.config import settings


def get_langchain_llm(feature: str = "rag") -> AzureChatOpenAI:
    """
    Return the Azure chat deployment used by LangGraph nodes.
    """
    deployment = {
        "router": settings.AZURE_OPENAI_ROUTER_DEPLOYMENT,
    }.get(feature, settings.AZURE_OPENAI_CHAT_DEPLOYMENT)
    logfire.info(
        "LLM gateway mode: Azure OpenAI | feature={feature} | deployment={deployment}",
        feature=feature,
        deployment=deployment,
    )
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=deployment,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        api_key=settings.AZURE_OPENAI_API_KEY,
        **({"temperature": 0} if feature == "router" else {}),
    )


def create_chat_completion(messages: list, deployment: str | None = None):
    """Create a chat completion through the configured Azure deployment."""
    selected_deployment = deployment or settings.AZURE_OPENAI_CHAT_DEPLOYMENT
    logfire.info(
        "Responder gateway mode: Azure OpenAI | deployment={deployment}",
        deployment=selected_deployment,
    )

    client = openai.AzureOpenAI(
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
    )

    return client.chat.completions.create(
        model=selected_deployment,
        messages=messages,
    )
