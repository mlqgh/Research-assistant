# Conversational Multi-Agent RAG Research Assistant

An enterprise-grade, conversational multi-agent Retrieval-Augmented Generation (RAG) research assistant built with **LangGraph**, **FastAPI**, **Groq (Llama-3.3-70b)**, **FAISS**, and **LangSmith**.

The system features an adaptive graph-based routing architecture that evaluates incoming queries, optimizes retrieval, grounds answers against context, and maintains multi-turn conversation memory via persistent checkpointers.

---

##  System Architecture

The pipeline is implemented as a stateful graph powered by LangGraph:

```text
[User Query] ──► [Router Node] ──► [Query Optimizer] ──► [Vectorstore Retriever]
                                                               │
[Final Output] ◄── [Fact Checker / Evaluator] ◄── [Response Synthesizer]
```

### Agent Components
* **Router**: Dynamically route queries between local vectorstores and external search strategies based on domain scope.
* **Query Optimizer**: Rewrites ambiguous user prompts into search-optimized queries utilizing conversation history.
* **Retriever**: Queries local **FAISS** vector indexes embedded with HuggingFace models.
* **Synthesizer**: Constructs grounded answers using **Groq (Llama-3.3-70b)**.
* **Fact Checker**: Evaluates response hallucination levels and checks context grounding before outputting the final answer.
* **MemorySaver**: Manages persistent conversation history using `thread_id` sessions.

---

##  Quickstart

### 1. Clone the Repository & Set Up Virtual Environment

```bash
git clone https://github.com/mlqgh/Research-assistant.git
cd Research-assistant

python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_api_key
LANGCHAIN_PROJECT=research-assistant-rag
HF_TOKEN=hf_your_huggingface_token
```

### 3. Build Vectorstore & Launch CLI Chatbot

Index sample documents into FAISS and start an interactive multi-turn terminal session:

```bash
python main.py
```

### 4. Launch the FastAPI Web API & Interface

Start the Uvicorn application server:

```bash
uvicorn app:app --reload
```

* **Interactive Web UI**: [http://localhost:8000](http://localhost:8000)
* **Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Containerized Deployment with Docker

Build and run the production image using Docker:

```bash
# Build Docker image
docker build -t rag-research-assistant .

# Run container passing your .env file
docker run -d -p 8000:8000 --name rag-app --env-file .env rag-research-assistant
```

---

##  API Usage Example

Send a POST request to the conversational endpoint with a session `thread_id`:

```bash
curl -X POST "http://localhost:8000/api/v1/rag/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is the lead engineer for Project Titan?",
    "thread_id": "session_demo_101"
  }'
```

---

## Tech Stack

* **Orchestration & Memory**: LangGraph, LangChain Core
* **Web Framework**: FastAPI, Uvicorn, Pydantic
* **LLM Engine**: Groq API (`llama-3.3-70b-versatile`)
* **Vector Storage & Embeddings**: FAISS, HuggingFace Transformers
* **Observability**: LangSmith
* **Deployment**: Docker, Python 3.10
