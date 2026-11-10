from functools import lru_cache

from app.config import settings
from app.gateway import get_langchain_llm
from app.router.models import (
    Capability,
    RouteDecision,
    RouterClassification,
    ToolName,
)

ROUTER_INSTRUCTIONS = """
Classify the user's request into exactly one capability and return a JSON object.

Capabilities:
- BLOCK: unsafe request or unrelated to the supported enterprise IT assistant.
- TOOL: current date or time. Use tool_name DATETIME.
- CHAT: greetings, conversation, or general questions not requiring internal
  documents or multi-step analysis.
- RAG: questions requiring enterprise/internal knowledge or technical documents.
- REASONING: comparison, evaluation, analysis, strategy, or recommendations.

Use RAG when the request depends on internal company information, even if it
also asks for analysis. Use REASONING for multi-step analysis of general topics.

Return JSON only, in this shape:
{"capability":"TOOL","tool_name":"DATETIME"}
For capabilities other than TOOL, omit tool_name.
"""


@lru_cache(maxsize=1)
def _get_router_llm():
    if not settings.AZURE_OPENAI_ROUTER_DEPLOYMENT:
        raise RuntimeError(
            "Set AZURE_OPENAI_ROUTER_DEPLOYMENT or "
            "AZURE_OPENAI_CHAT_DEPLOYMENT to configure the Capability Router."
        )
    return get_langchain_llm(feature="router")


def _select_model(capability: Capability) -> str:
    if capability in {Capability.BLOCK, Capability.TOOL}:
        return "none"

    deployment = (
        settings.AZURE_OPENAI_REASONING_DEPLOYMENT
        if capability == Capability.REASONING
        else settings.AZURE_OPENAI_CHAT_DEPLOYMENT
    )
    if not deployment:
        raise RuntimeError(
            f"No Azure OpenAI deployment is configured for {capability.value}."
        )
    return deployment


def determine_route(query: str) -> RouteDecision:
    response = _get_router_llm().invoke(
        [
            ("system", ROUTER_INSTRUCTIONS),
            ("human", query),
        ]
    )
    if not isinstance(response.content, str):
        raise ValueError("Capability Router returned non-text content.")

    classification = RouterClassification.model_validate_json(response.content)
    if classification.capability == Capability.TOOL:
        if classification.tool_name is None:
            raise ValueError("Capability Router selected TOOL without a tool_name.")

    return RouteDecision(
        capability=classification.capability,
        model=_select_model(classification.capability),
        use_retrieval=classification.capability
        in {Capability.RAG, Capability.REASONING},
        use_tool=classification.capability == Capability.TOOL,
        tool_name=classification.tool_name,
    )
