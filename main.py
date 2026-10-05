import os
from dotenv import load_dotenv
from ingestion import build_vectorstore
from agent_graph import build_rag_graph

load_dotenv()

def run_chat_session():
    app = build_rag_graph()
    
    # Shared session thread ID for conversation memory
    thread_config = {"configurable": {"thread_id": "cli_session_101"}}

    # Turn 1: Initial Question
    query_1 = "Who is the lead engineer for Project Titan?"
    print(f"\n[User]: {query_1}")
    res1 = app.invoke({"messages": [("human", query_1)], "user_query": query_1}, config=thread_config)
    print(f"[Bot]: {res1['final_response']}")

    # Turn 2: Follow-up question relying on prior turn memory ("she/her")
    query_2 = "What is her target performance metric?"
    print(f"\n[User]: {query_2}")
    res2 = app.invoke({"messages": [("human", query_2)], "user_query": query_2}, config=thread_config)
    print(f"[Bot]: {res2['final_response']}")

if __name__ == "__main__":
    run_chat_session()