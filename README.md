# Enterprise Agentic RAG Assistant

A production-style Agentic RAG system for enterprise document intelligence. It combines a FastAPI backend, LangGraph agent workflow, Qdrant vector search, NeMo Guardrails, LLM-based reranking, Langfuse tracing, Logfire monitoring, Streamlit UI, and an evaluation pipeline.

This project is designed as an interview-ready backend + AI systems showcase: it demonstrates how a real RAG application can separate UI, API, retrieval, guardrails, observability, ingestion, and evaluation into clean modules.

![Enterprise Agentic RAG Architecture](DOCS/assets/enterprise-agentic-rag-architecture.png)

---

## What This Project Does

The assistant answers questions from a curated technical documentation corpus while filtering unsafe/off-topic input, retrieving relevant chunks, reranking context, generating grounded answers, and tracing the full request path.

Core flow:

```text
User -> Streamlit UI -> FastAPI /query -> Guardrails -> LangGraph Agent
     -> Planner -> Retriever -> Qdrant -> Reranker -> Responder -> LLM
     -> Final Answer + Sources + Trace
```

---

## Key Features

- **Agentic RAG workflow**: LangGraph planner, retriever, and responder nodes.
- **Guardrails before retrieval**: NeMo Guardrails checks the user request before expensive search/LLM calls.
- **Vector search**: Qdrant Cloud stores indexed document chunks and serves semantic retrieval.
- **LLM-based reranking**: retrieved chunks are reranked before answer generation.
- **Session memory**: LangGraph `MemorySaver` keeps conversation context by `thread_id`.
- **Central LLM gateway layer**: all main generation calls go through `app/gateway`.
- **Observability**: Langfuse traces request-level spans and LLM calls; Logfire captures backend events.
- **Document ingestion**: local parsers handle PDF, HTML, TXT, DOCX, and PPTX files.
- **Evaluation suite**: RAGAS/custom evaluation pipeline for checking answer quality.
- **Deployment-ready backend**: Dockerfile for container platforms and `api/index.py` + `vercel.json` for Vercel.

---

## Architecture

```mermaid
graph TD
    User((User)) --> UI[Streamlit UI]
    UI --> API[FastAPI Backend]
    API --> Guard{NeMo Guardrails}
    Guard -->|Blocked| Blocked[Safe Response]
    Guard -->|Allowed| Agent[LangGraph Agent]
    Agent --> Planner[Planner Node]
    Planner -->|Conversational| Responder[Responder Node]
    Planner -->|Technical Query| Retriever[Retriever Node]
    Retriever --> Qdrant[(Qdrant Vector DB)]
    Qdrant --> Reranker[LLM Reranker]
    Reranker --> Responder
    Responder --> Gateway[LLM Gateway Layer]
    Gateway --> LLM[OpenAI-Compatible LLM]
    LLM --> API
    API --> UI
    Agent -.-> Memory[(Session Memory)]
    API -.-> Langfuse[Langfuse Tracing]
    API -.-> Logfire[Logfire Monitoring]
```

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend API | FastAPI, Uvicorn |
| Agent orchestration | LangGraph, LangChain |
| LLM interface | Azure OpenAI, LangChain |
| Vector database | Qdrant Cloud |
| Guardrails | NeMo Guardrails |
| Reranking | LLM-based reranker |
| Observability | Langfuse, Pydantic Logfire |
| UI | Streamlit |
| Evaluation | RAGAS-style evaluation pipeline, custom metrics |
| Deployment | Docker, Vercel serverless adapter |

---

## Project Structure

```text
.
+-- app/
|   +-- agents/
|   |   +-- graph.py
|   |   +-- state.py
|   |   +-- nodes/
|   |       +-- planner.py
|   |       +-- retriever.py
|   |       +-- responder.py
|   +-- gateway/
|   |   +-- client.py
|   +-- guardrails/
|   |   +-- rails.py
|   |   +-- colang_rules.py
|   +-- ingestion/
|   |   +-- processor.py
|   |   +-- chunking/
|   |   +-- loaders/
|   +-- services/
|   |   +-- retrieval/
|   +-- config.py
|   +-- main.py
+-- api/
|   +-- index.py
+-- ui/
|   +-- app.py
+-- evals/
+-- DOCS/
+-- DATA/
+-- Dockerfile
+-- vercel.json
+-- requirements.txt
+-- requirements-prod.txt
```

---

## Environment Variables

Create a `.env` file locally, or set these values in your deployment platform.

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com/
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_PLANNER_DEPLOYMENT=
AZURE_OPENAI_GUARDRAIL_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
AZURE_OPENAI_RERANK_DEPLOYMENT=
AZURE_OPENAI_EVAL_DEPLOYMENT=

# Qdrant
QDRANT_CLUSTER_ENDPOINT=
QDRANT_API_KEY=

# Observability
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LOGFIRE_TOKEN=

# Streamlit UI
BACKEND_URL=http://localhost:8000

```

`AZURE_OPENAI_PLANNER_DEPLOYMENT` should be set to the Azure deployment name for the planner model (for example, `gpt-4.1-nano`). If unset, the planner uses `AZURE_OPENAI_CHAT_DEPLOYMENT`.

---

## Local Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Ingest documents into Qdrant

```powershell
python -m app.ingestion.processor DATA --wipe
```

Use `--wipe` when you want to recreate the Qdrant collection from scratch.

### 4. Run the backend

```powershell
uvicorn app.main:app --reload --port 8000
```

Backend health check:

```text
http://localhost:8000/
```

### 5. Run the Streamlit UI

```powershell
streamlit run ui/app.py
```

---

## API Usage

Send a query to the backend:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"q\":\"How do Kubernetes CronJobs work?\",\"thread_id\":\"demo-user\"}"
```

Response shape:

```json
{
  "question": "How do Kubernetes CronJobs work?",
  "answer": "Generated answer...",
  "thought_process": ["Intent: Technical", "Context Retrieved"],
  "status": "Response generated.",
  "sources": ["CONTENT: ..."]
}
```

---

## Evaluation

Run the evaluation dashboard:

```powershell
streamlit run evals/app.py
```

The evaluation suite is separated from the live backend so experiments and judge-model calls do not interfere with the production request path.

---

## Deployment

### Container Platforms

Use the included Dockerfile for Render, Railway, Fly.io, or Cloud Run style deployments:

```bash
docker build -t enterprise-agentic-rag .
docker run -p 8080:8080 --env-file .env enterprise-agentic-rag
```

### Vercel Backend

This repo also includes:

- `api/index.py`
- `vercel.json`
- `.vercelignore`
- `requirements-prod.txt`

Set the required environment variables in Vercel, then deploy the repository. The FastAPI app is exposed through the Vercel Python function entrypoint.

### GitHub Actions to Azure Container Apps

The workflow at `.github/workflows/deploy-azure-container-apps.yml` runs on every push to `main` (including a merge into `main`) and supports manual runs from the GitHub Actions tab. It syntax-checks Python, builds the backend and frontend images, pushes them to Azure Container Registry (ACR), and updates the two Container Apps with the new commit-tagged images.

Before using it:

1. Configure an Entra ID application or service principal with a GitHub Actions federated credential. Set its subject to `repo:OWNER/REPOSITORY:ref:refs/heads/main` and allow the `id-token` audience `api://AzureADTokenExchange`. The workflow uses OIDC, so no Azure client secret is stored in GitHub.
2. Grant that identity `AcrPush` on the ACR and `Container Apps Contributor` on the Container Apps resource group.
3. In **GitHub repository → Settings → Secrets and variables → Actions → Variables**, add:

   | Variable | Value |
   | --- | --- |
   | `AZURE_CLIENT_ID` | Entra ID application (client) ID |
   | `AZURE_TENANT_ID` | Entra ID tenant ID |
   | `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
   | `AZURE_RESOURCE_GROUP` | Resource group containing the Container Apps |
   | `AZURE_ACR_NAME` | ACR resource name, without `.azurecr.io` |
   | `AZURE_ACR_LOGIN_SERVER` | ACR login server, e.g. `myregistry.azurecr.io` |
   | `AZURE_BACKEND_CONTAINER_APP` | Backend Container App name |
   | `AZURE_FRONTEND_CONTAINER_APP` | Frontend Container App name |

The Container Apps must already be configured with their runtime environment variables, ingress, and registry pull access. The workflow updates the image only; it does not overwrite those settings. After setup, merging a change into `main` triggers a build and deploy. Check **GitHub → Actions** for build or deployment logs.

---

## Documentation

| Guide | Description |
| --- | --- |
| [System Overview](DOCS/01_SYSTEM_OVERVIEW.md) | End-to-end system flow |
| [Ingestion Engine](DOCS/02_INGESTION_ENGINE.md) | Document parsing and chunk indexing |
| [Node Intelligence](DOCS/03_NODE_INTELLIGENCE.md) | Planner, retriever, responder internals |
| [Tracing and Observability](DOCS/04_TRACING_AND_OBSERVABILITY.md) | Logfire and tracing design |
| [Environment Variables](DOCS/05_ENVIRONMENT_VARIABLES.md) | Configuration reference |
| [Known Gotchas](DOCS/06_KNOWN_GOTCHAS.md) | Bugs, fixes, and design notes |
| [Azure OpenAI Reranking](DOCS/07_AZURE_RERANKING.md) | Azure LLM reranking strategy |
| [Guardrails](DOCS/08_GUARDRAILS.md) | NeMo Guardrails implementation |
| [LLM Gateway](DOCS/09_LLM_GATEWAY.md) | Gateway abstraction and routing |
| [Evals](DOCS/10_EVALS.md) | Evaluation theory and metrics |
| [Evals Pipeline](DOCS/11_EVALS_PIPELINE.md) | Live evaluation workflow |

---

## Interview Summary

This project demonstrates how to build an enterprise-style RAG system with clean separation between UI, API, agent orchestration, retrieval, guardrails, observability, and evaluation.

It is demo-ready and deployment-ready for portfolio/interview use. For a public production rollout, the next hardening steps would be authentication, rate limiting, CI tests, stricter dependency pinning, and deployment monitoring.
