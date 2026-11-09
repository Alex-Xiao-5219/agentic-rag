from enum import Enum


class RouteType(Enum):

    CHAT = "CHAT"

    RAG = "RAG"

    AGENT = "AGENT"

RAG_KEYWORDS = [
    "our",
    "company",
    "internal",
    "policy",
    "procedure",
    "document",
    "handbook",
    "wiki",
    "guideline",
]

AGENT_KEYWORDS = [
    "compare",
    "analyse",
    "analyze",
    "evaluate",
    "strategy",
    "plan",
    "recommend",
]

def determine_route(query: str):

    q = query.lower()

    if any(
        keyword in q
        for keyword in AGENT_KEYWORDS
    ):
        return RouteType.AGENT

    if any(
        keyword in q
        for keyword in RAG_KEYWORDS
    ):
        return RouteType.RAG

    return RouteType.CHAT