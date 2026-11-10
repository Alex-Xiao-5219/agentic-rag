from datetime import datetime

from app.router.models import ToolName

def execute_tool(
    tool_name: ToolName,
    *,
    query: str = "",
) -> dict[str, object]:
    if tool_name == ToolName.DATETIME:
        now = datetime.now().astimezone()
        normalized_query = query.lower()
        asks_for_time = "time" in normalized_query
        asks_for_date = any(word in normalized_query for word in ("date", "day"))
        if asks_for_time and asks_for_date:
            answer = now.strftime("Current local date and time is %Y-%m-%d %H:%M:%S (%Z).")
        elif asks_for_time:
            answer = now.strftime("Current local time is %H:%M:%S (%Z).")
        else:
            answer = now.strftime("Today is %Y-%m-%d.")
        return {
            "answer": answer,
            "plan": ["DateTime Tool"],
            "status": "Completed",
            "documents": [],
        }

    raise ValueError(f"Unknown tool: {tool_name}")
