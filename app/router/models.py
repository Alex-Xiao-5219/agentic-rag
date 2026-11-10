from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class Capability(str, Enum):
    BLOCK = "BLOCK"
    TOOL = "TOOL"
    CHAT = "CHAT"
    RAG = "RAG"
    REASONING = "REASONING"


class ToolName(str, Enum):
    DATETIME = "DATETIME"


class RouterClassification(BaseModel):
    capability: Capability
    tool_name: ToolName | None = None


@dataclass(frozen=True)
class RouteDecision:
    capability: Capability
    model: str
    use_retrieval: bool
    use_tool: bool
    tool_name: ToolName | None = None
