from datetime import datetime


FAST_RESPONSES = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hello! How can I help you today?",
    "hey": "Hello! How can I help you today?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "help": """
I can help with:

• Enterprise Knowledge Search
• AI Engineering
• Kubernetes
• Networking
• Azure
• Qdrant
""",
}


def fast_path(query: str):

    q = query.lower().strip()

    if q in FAST_RESPONSES:
        return True, FAST_RESPONSES[q]

    if (
        "date" in q
        and "today" in q
    ):
        return (
            True,
            datetime.now().strftime(
                "Today is %Y-%m-%d"
            )
        )

    if (
        "time" in q
        and (
            "current" in q
            or "now" in q
        )
    ):
        return (
            True,
            datetime.now().strftime(
                "Current time is %H:%M:%S"
            )
        )

    return False, None