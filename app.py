import os
from contextlib import asynccontextmanager
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from agent_graph import build_rag_graph

resources: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing Conversational LangGraph RAG Agent...")
    resources["graph"] = build_rag_graph()
    yield
    resources.clear()

app = FastAPI(title="Conversational RAG Research Assistant", lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str = Field(..., example="Who is Sarah Chen?")
    thread_id: str = Field(default="session_123", description="Persistent thread identifier for chat history")

class ChatResponse(BaseModel):
    thread_id: str
    user_query: str
    datasource: str
    search_query: str
    final_response: str

@app.post("/api/v1/rag/chat", response_model=ChatResponse)
async def handle_chat(payload: ChatRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Pass input as HumanMessage to maintain chat history sequence
        initial_input = {
            "messages": [("human", payload.query)],
            "user_query": payload.query,
            "search_query": "",
            "datasource": "",
            "retrieved_context": "",
            "draft_response": "",
            "grounding_eval": None,
            "final_response": ""
        }

        # Thread ID triggers short-term memory retrieval across calls
        config = {
            "configurable": {"thread_id": payload.thread_id},
            "tags": ["chatbot", "fastapi-rag"]
        }

        graph = resources["graph"]
        final_state = await graph.ainvoke(initial_input, config=config)

        return ChatResponse(
            thread_id=payload.thread_id,
            user_query=payload.query,
            datasource=final_state.get("datasource", ""),
            search_query=final_state.get("search_query", ""),
            final_response=final_state.get("final_response", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat workflow: {str(e)}")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Conversational RAG Chatbot</title>
        <style>
            body { background: #121212; color: #fff; font-family: sans-serif; display: flex; flex-direction: column; height: 100vh; margin: 0; }
            #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 10px; }
            .msg { padding: 10px 15px; border-radius: 8px; max-width: 75%; }
            .user { align-self: flex-end; background: #2e7d32; }
            .bot { align-self: flex-start; background: #262626; border: 1px solid #333; }
            #input-box { display: flex; padding: 15px; background: #1e1e1e; gap: 10px; }
            input { flex: 1; padding: 10px; background: #2a2a2a; border: 1px solid #444; color: #fff; border-radius: 4px; }
            button { padding: 10px 20px; background: #4caf50; border: none; color: #fff; font-weight: bold; cursor: pointer; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h2 style="padding: 10px 20px; margin: 0; background: #1e1e1e; border-bottom: 1px solid #333;">Research Assistant Chatbot</h2>
        <div id="chat-box"></div>
        <div id="input-box">
            <input id="user-input" placeholder="Ask a follow-up question..." onkeypress="if(event.key==='Enter') sendMsg()">
            <button onclick="sendMsg()">Send</button>
        </div>
        <script>
            // Generate or reuse persistent thread ID for this browser tab session
            const threadId = "thread_" + Math.random().toString(36).substring(2, 9);

            async function sendMsg() {
                const input = document.getElementById("user-input");
                const text = input.value.trim();
                if (!text) return;

                appendMsg(text, "user");
                input.value = "";

                const res = await fetch("/api/v1/rag/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: text, thread_id: threadId })
                });

                const data = await res.json();
                appendMsg(data.final_response, "bot");
            }

            function appendMsg(text, sender) {
                const box = document.getElementById("chat-box");
                const div = document.createElement("div");
                div.className = `msg ${sender}`;
                div.innerText = text;
                box.appendChild(div);
                box.scrollTop = box.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)