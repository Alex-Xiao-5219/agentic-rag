import os
from contextlib import asynccontextmanager

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from pydantic import BaseModel

from app.fast_path.fast_path import fast_path
from app.security.security import security_check
from app.router.router import determine_route, RouteType
from app.router.chat import chat_response

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
    yield
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

                # ==================================================
                # STEP 1 : FAST PATH
                # ==================================================

                hit, response = fast_path(q)

                if hit:

                    payload = {
                        "question": q,
                        "answer": response,
                        "status": "FAST_PATH",
                        "sources": [],
                    }

                    span.update(output=payload)

                    return payload

                # ==================================================
                # STEP 2 : SECURITY
                # ==================================================

                blocked, reason = security_check(q)

                if blocked:

                    payload = {
                        "question": q,
                        "answer": reason,
                        "status": "BLOCKED",
                        "sources": [],
                    }

                    span.update(output=payload)

                    return payload

                # ==================================================
                # STEP 3 : ROUTER
                # ==================================================

                route = determine_route(q)

                # ==================================================
                # CHAT
                # ==================================================

                if route == RouteType.CHAT:

                    answer = chat_response(q)

                    payload = {
                        "question": q,
                        "answer": answer,
                        "status": "CHAT",
                        "sources": [],
                    }

                    span.update(output=payload)

                    return payload

                # ==================================================
                # RAG
                # ==================================================

                if route == RouteType.RAG:

                    from app.agents.graph import rag_agent

                    initial_state = {
                        "messages": [
                            {
                                "role": "user",
                                "content": q
                            }
                        ],
                        "current_query": q,
                        "documents": [],
                        "plan": ["Start"],
                        "status": "Initializing RAG",
                    }

                    langfuse_handler = CallbackHandler()

                    config = {
                        "configurable": {
                            "thread_id": thread_id
                        },
                        "callbacks": [
                            langfuse_handler
                        ],
                    }

                    final_output = rag_agent.invoke(
                        initial_state,
                        config=config,
                    )

                    payload = {
                        "question": q,
                        "answer": final_output.get(
                            "final_answer"
                        ),
                        "thought_process": final_output.get(
                            "plan"
                        ),
                        "status": "RAG",
                        "sources": final_output.get(
                            "documents",
                            [],
                        ),
                    }

                    span.update(output=payload)

                    return payload

                # ==================================================
                # AGENT (未来扩展)
                # ==================================================

                if route == RouteType.AGENT:

                    answer = chat_response(q)

                    payload = {
                        "question": q,
                        "answer": answer,
                        "status": "AGENT",
                        "sources": [],
                    }

                    span.update(output=payload)

                    return payload

        return {
            "status": "UNKNOWN_ROUTE"
        }

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