import os
from typing import List
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------------------
# 1. Initialize HuggingFace Embedding Model
# --------------------------------------------------------------
# Popular efficient model: 'sentence-transformers/all-MiniLM-L6-v2'
# Alternative high-performance model: 'BAAI/bge-small-en-v1.5'
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

INDEX_PATH = "faiss_index"

def get_vectorstore() -> FAISS:
    """Loads the FAISS vector database from disk if available."""
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            folder_path=INDEX_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    raise FileNotFoundError(f"Vectorstore index not found at '{INDEX_PATH}'. Please build the index first.")

# --------------------------------------------------------------
# 2. Vector Search Tool
# --------------------------------------------------------------
@tool
def vectorstore_search_tool(query: str, top_k: int = 3) -> str:
    """Useful for searching local vectorstore documents using semantic similarity."""
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant documents found in the vectorstore."
        
        formatted_results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown Source")
            formatted_results.append(f"[{i}] Source: {source}\nContent: {doc.page_content}")
            
        return "\n\n---\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Error executing vectorstore search: {str(e)}"
@tool
def tavily_search_tool(query: str) -> str:
    """Performs live web search using Tavily AI Search when local docs lack context."""
    search = TavilySearchResults(max_results=3)
    results = search.invoke({"query": query})
    
    formatted_results = []
    for res in results:
        formatted_results.append(f"Source: {res['url']}\nContent: {res['content']}")
    return "\n\n".join(formatted_results)
