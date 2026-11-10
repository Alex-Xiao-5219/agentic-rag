import os
from contextlib import asynccontextmanager

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from app.gateway import create_chat_completion
from app.router.capability_router import determine_route
from app.router.executor import execute_tool
from app.router.models import Capability

load_dotenv()


def configure_logfire():

    token = os.getenv("LOGFIRE_TOKEN")

    if not token:
        os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"
        return

    try:
        logfire.configure(token=token)
    except Exception as exc:
        print(f"Logfire disabled: {exc}")


configure_logfire()

langfuse = get_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        langfuse.flush()


app = FastAPI(
    title="Enterprise Agentic RAG API",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):

    q: str

    thread_id: str | None = "default_user"


@app.get("/")
def home():

    return {
        "message": "Enterprise Agentic RAG API Running"
    }


@app.post("/query")
def query(request: QueryRequest):

    q = request.q

    thread_id = request.thread_id

    try:

        with langfuse.start_as_current_observation(
            as_type="span",
            name="enterprise-query",
            input={
                "question": q,
                "thread_id": thread_id,
            },
        ) as span:

            with propagate_attributes(
                trace_name="enterprise-query",
                user_id=thread_id,
                session_id=thread_id,
            ):
                decision = determine_route(q)
                print("ROUTE =", decision)

                if decision.capability == Capability.BLOCK:
                    payload = {
                        "question": q,
                        "answer": (
                            "I can help with enterprise IT, technical knowledge, "
                            "and related analysis."
                        ),
                        "thought_process": ["Capability: BLOCK"],
                        "status": "Blocked",
                        "sources": [],
                    }
                    span.update(output=payload)
                    return payload

                if decision.capability == Capability.TOOL:
                    if decision.tool_name is None:
                        raise ValueError("Tool route is missing a tool name.")
                    tool_output = execute_tool(
                        decision.tool_name,
                        query=q,
                    )
                    payload = {
                        "question": q,
                        "answer": tool_output["answer"],
                        "thought_process": [
                            f"Capability: TOOL ({decision.tool_name.value})"
                        ],
                        "status": tool_output["status"],
                        "sources": [],
                    }
                    span.update(output=payload)
                    return payload

                if decision.capability == Capability.CHAT:
                    response = create_chat_completion(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful enterprise IT assistant. "
                                    "Answer concisely."
                                ),
                            },
                            {"role": "user", "content": q},
                        ],
                        deployment=decision.model,
                    )
                    answer = response.choices[0].message.content
                    payload = {
                        "question": q,
                        "answer": answer,
                        "thought_process": ["Capability: CHAT"],
                        "status": "Completed",
                        "sources": [],
                    }
                    span.update(output=payload)
                    return payload

                if decision.capability not in {
                    Capability.RAG,
                    Capability.REASONING,
                }:
                    raise ValueError(
                        f"Unsupported capability: {decision.capability.value}"
                    )

                from app.agents.graph import rag_agent

                initial_state = {
                    "messages": [
                        {
                            "role": "user",
                            "content": q,
                        }
                    ],
                    "current_query": q,
                    "documents": [],
                    "plan": [f"Capability: {decision.capability.value}"],
                    "status": "Retrieving enterprise context",
                }

                langfuse_handler = CallbackHandler()
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "model_deployment": decision.model,
                    },
                    "callbacks": [langfuse_handler],
                }

                final_output = rag_agent.invoke(
                    initial_state,
                    config=config,
                )
                payload = {
                    "question": q,
                    "answer": final_output.get("final_answer"),
                    "thought_process": [
                        f"Capability: {decision.capability.value}",
                        *(final_output.get("plan") or []),
                    ],
                    "status": final_output.get("status"),
                    "sources": final_output.get("documents", []),
                }
                span.update(output=payload)
                return payload

    except Exception as e:

        logfire.exception("Execution Failed")

        return JSONResponse(
            status_code=500,
            content={
                "question": q,
                "answer": str(e),
                "status": "ERROR",
                "sources": [],
            },
        )