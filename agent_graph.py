# agent_graph.py
import os
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from unittest.mock import Base
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, add_messages

from schemas import QueryRoute, QueryRefinement, GroundingEvaluation
from tools import get_vectorstore, tavily_search_tool
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# State definition
class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    search_query: str
    datasource: str
    retrieved_context: str
    draft_response: str
    grounding_eval: Optional[dict]
    final_response: str
# Helper function to load Groq LLM (e.g. Llama-3.3-70b)
def get_groq_llm(temperature: float = 0.0):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured. Add a valid Groq API key to .env.")
    if not api_key.startswith("gsk_"):
        raise RuntimeError(
            "GROQ_API_KEY does not look like a Groq key. Replace the OpenAI-style key in .env with a Groq key."
        )

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=temperature
    )

# Node 1: Router Agent
def router_node(state: RAGState) -> RAGState:
    llm = get_groq_llm(temperature=0.0)
    structured_router = llm.with_structured_output(QueryRoute)

    prompt = ChatPromptTemplate.from_template(
        "Analyze the user query and decide the best routing option.\n"
        "- Choose 'vectorstore' if the query asks about internal company documents, reports, or domain files.\n"
        "- Choose 'tavily_search' if the query requires live web search or current general facts.\n"
        "- Choose 'direct_answer' if it is a general greeting or non-research question.\n\n"
        "User Query: {query}"
    )

    decision = structured_router.invoke(prompt.format(query=state["user_query"]))
    state["datasource"] = decision.datasource
    return state

# Node 2: Query Optimization Agent
def optimizer_node(state: RAGState) -> RAGState:
    if state["datasource"] == "direct_answer":
        state["search_query"] = state["user_query"]
        return state

    llm = get_groq_llm(temperature=0.0)
    structured_optimizer = llm.with_structured_output(QueryRefinement)

    prompt = ChatPromptTemplate.from_template(
        "Rephrase the following user input into a concise, high-density keyword search query "
        "optimized for semantic retrieval.\n\nInput: {query}"
    )

    result = structured_optimizer.invoke(prompt.format(query=state["user_query"]))
    state["search_query"] = result.optimized_query
    return state

# Node 3: Retrieval Agent
def retrieval_node(state: RAGState) -> RAGState:
    datasource = state["datasource"]
    query = state["search_query"]

    if datasource == "vectorstore":
        vs = get_vectorstore()
        if vs:
            docs = vs.similarity_search(query, k=3)
            context = "\n---\n".join([f"Source: {d.metadata.get('source', 'doc')}\nContent: {d.page_content}" for d in docs])
            state["retrieved_context"] = context
        else:
            state["retrieved_context"] = "No local vector store found."
            state["datasource"] = "tavily_search"
            return retrieval_node(state)

    elif datasource == "tavily_search":
        search_results = tavily_search_tool.invoke({"query": query})
        state["retrieved_context"] = search_results

    else:
        state["retrieved_context"] = "N/A"

    return state

# Node 4: Synthesis Agent
def synthesis_node(state: RAGState) -> RAGState:
    llm = get_groq_llm(temperature=0.2)

    if state["datasource"] == "direct_answer":
        response = llm.invoke(f"Answer politely: {state['user_query']}")
        state["draft_response"] = response.content
        state["final_response"] = response.content
        return state

    prompt = ChatPromptTemplate.from_template(
        "Synthesize a clear, analytical answer using ONLY the context provided below. "
        "Cite the sources mentioned in the context where relevant.\n\n"
        "Context:\n{context}\n\n"
        "User Query: {query}"
    )

    response = llm.invoke(prompt.format(context=state["retrieved_context"], query=state["user_query"]))
    state["draft_response"] = response.content
    return state

# Node 5: Fact-Checking & Grounding Agent
def fact_checker_node(state: RAGState) -> RAGState:
    if state["datasource"] == "direct_answer":
        return state

    llm = get_groq_llm(temperature=0.0)
    evaluator = llm.with_structured_output(GroundingEvaluation)

    prompt = ChatPromptTemplate.from_template(
        "Evaluate whether the Draft Answer is strictly grounded in the Context. "
        "Flag any facts not present in the context as hallucinations.\n\n"
        "Context:\n{context}\n\n"
        "Draft Answer:\n{draft}"
    )

    evaluation = evaluator.invoke(prompt.format(
        context=state["retrieved_context"], 
        draft=state["draft_response"]
    ))
    
    state["grounding_eval"] = evaluation

    if evaluation.is_grounded:
        state["final_response"] = state["draft_response"]
    else:
        state["final_response"] = (
            f"{state['draft_response']}\n\n"
            f"*(Note: Evaluator flagged unverified statements: {', '.join(evaluation.hallucinations_detected)})*"
        )

    return state

# Build State Graph
def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("router", router_node)
    workflow.add_node("optimizer", optimizer_node)
    workflow.add_node("retriever", retrieval_node)
    workflow.add_node("synthesizer", synthesis_node)
    workflow.add_node("fact_checker", fact_checker_node)

    workflow.set_entry_point("router")
    
    workflow.add_edge("router", "optimizer")
    workflow.add_edge("optimizer", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "fact_checker")
    workflow.add_edge("fact_checker", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

