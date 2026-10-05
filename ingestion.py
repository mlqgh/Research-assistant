# ingestion.py
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def build_vectorstore(file_paths: list[str], save_dir: str = "faiss_index"):
    """Ingests documents, chunks text, generates embeddings, and saves FAISS index."""
    documents = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        elif path.endswith(".txt") or path.endswith(".md"):
            loader = TextLoader(path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_dir)
    print(f"[Ingestion Complete] Indexed {len(chunks)} chunks into '{save_dir}'.")

if __name__ == "__main__":
    # Example usage for manual ingestion
    os.makedirs("sample_docs", exist_ok=True)
    sample_file = "sample_docs/report.txt"
    with open(sample_file, "w") as f:
        f.write("Project Titan was launched in 2024. Its goal is reducing server latency by 40%. "
                "The core engine is built using Rust and FastAPI.")
    build_vectorstore([sample_file])