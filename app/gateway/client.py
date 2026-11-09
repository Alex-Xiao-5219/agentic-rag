import logfire
from langchain_openai import AzureChatOpenAI
from langfuse.openai import openai

from app.config import settings


def get_langchain_llm(feature: str = "rag") -> AzureChatOpenAI:
    """
    Return the Azure chat deployment used by LangGraph nodes.
    """
    deployment = (
        settings.AZURE_OPENAI_PLANNER_DEPLOYMENT
        if feature == "planner"
        else settings.AZURE_OPENAI_CHAT_DEPLOYMENT
    )
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
    )


def create_chat_completion(messages: list):
    """Create a chat completion through the configured Azure deployment."""
    logfire.info(
        "Responder gateway mode: Azure OpenAI | deployment={deployment}",
        deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
    )

    client = openai.AzureOpenAI(
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
    )

    return client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
    )
