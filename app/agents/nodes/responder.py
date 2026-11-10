import logfire
from langchain_core.runnables import RunnableConfig

from app.agents.state import AgentState
from app.gateway import create_chat_completion


def generate_node(
    state: AgentState,
    config: RunnableConfig | None = None,
):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the Azure OpenAI client through the gateway abstraction.
    """
    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    logfire.info("Generating response using retrieved technical context.")
    max_context_chars = 25000
    full_context = ""

    for doc in state["documents"]:
        if len(full_context) + len(doc) < max_context_chars:
            full_context += doc + "\n\n"
        else:
            logfire.warning("Context truncated to fit the configured model's context limit.")
            break

    prompt = f"""
    You are a Senior Technical Architect.
    Answer the question using the TECHNICAL CONTEXT provided.

    TECHNICAL CONTEXT:
    {full_context}

    CONVERSATION HISTORY:
    {history_str}

    USER QUESTION:
    "{user_msg}"
    """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            model_deployment = (
                config.get("configurable", {}).get("model_deployment")
                if config
                else None
            )
            response = create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                deployment=model_deployment,
            )
            content = response.choices[0].message.content
            logfire.info("✅ Response synthesised via Azure OpenAI.")

            return {
                "final_answer": content,
                "status": "Response generated.",
                "plan": state["plan"],
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error("LLM generation failed: {error}", error=str(e))
            raise
