from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv
from google import genai
from typing import List

from backend.rag_engine import (
    build_vector_store,
    retrieve_context,
    add_document_to_vector_store
)

# 🔹 Load env
load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY not found")

print("🔑 GOOGLE API KEY PREFIX:", GOOGLE_API_KEY[:6])

# ✅ CORRECT Gemini client initialization
client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI(
    title="CORTEX-RAG",
    description="Retrieval-Augmented Generation Backend",
    version="1.0"
)

class Question(BaseModel):
    query: str
    history: list[dict] = []

@app.on_event("startup")
def startup_event():
    build_vector_store()

@app.get("/")
def root():
    return {"message": "CORTEX-RAG backend running 🚀"}

@app.post("/ask")
def ask_question(data: Question):
    try:
        context_docs = retrieve_context(data.query)

        if not context_docs:
            return {"query": data.query, "answer": "I don't know", "chunks_used": 0}

        # Format context to show Gemini which file each chunk came from
        context_sections = []
        sources = set()
        
        for doc in context_docs:
            source_name = doc.metadata.get("source", "Unknown Source")
            sources.add(source_name)
            context_sections.append(f"SOURCE: {source_name}\nCONTENT: {doc.page_content}")

        context_text = "\n\n".join(context_sections)
        chat_history_str = ""
        for turn in data.history:
            role = "User" if turn["role"] == "user" else "Assistant"
            chat_history_str += f"{role}: {turn['content']}\n"

        prompt = f"""
You are an intelligent assistant. 
Answer ONLY using the provided context. 
For every claim you make, you MUST cite the source filename in brackets at the end of the sentence, e.g., [Source: filename.pdf].

Context:
{context_text}

Question:
{data.query}

Answer:
"""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        updated_history = data.history + [
            {"role": "user", "content": data.query},
            {"role": "assistant", "content": response.text}
        ]
        return {
            "query": data.query,
            "answer": response.text,
            "history": updated_history,
            "sources_consulted": list(sources),
            "chunks_used": len(context_docs)
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.post("/global-search")
def global_search(data: Question):
    try:
        # 1. Get a larger set of chunks from across the library
        context_docs = retrieve_context(data.query, k=10)

        # 2. Group content by source to help the AI distinguish between files
        file_map = {}
        for doc in context_docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in file_map:
                file_map[src] = []
            file_map[src].append(doc.page_content)

        # 3. Build a "Comparison-Ready" Context
        formatted_context = ""
        for src, contents in file_map.items():
            formatted_context += f"--- DOCUMENT: {src} ---\n"
            formatted_context += "\n".join(contents) + "\n\n"

        # 4. Global Analysis Prompt
        prompt = f"""
You are a research assistant performing a Global Search across multiple technical documents.
Your goal is to provide a synthesized answer that compares or aggregates information from all provided documents.

Context from multiple files:
{formatted_context}

Question: {data.query}

Instructions:
- If documents disagree, highlight the differences.
- Cite every document used [Source: filename.pdf].
- If information is only in one file, specify that.
"""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return {
            "answer": response.text,
            "files_analyzed": list(file_map.keys()),
            "total_chunks": len(context_docs)
        }

    except Exception as e:
        return {"error": str(e)}
    
from typing import List
from fastapi import UploadFile, File

@app.post("/upload")
def upload_files(files: List[UploadFile] = File(...)):
    try:
        # 1. Define and create the document directory
        docs_dir = os.path.join(os.getcwd(), "data", "docs")
        os.makedirs(docs_dir, exist_ok=True)
        
        uploaded_filenames = []

        for file in files:
            file_path = os.path.join(docs_dir, file.filename)
            
            # 2. Save the physical file to disk
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # 3. Add ONLY the new document to the vector store
            # This function handles chunking and saving to disk
            add_document_to_vector_store(file_path)
            uploaded_filenames.append(file.filename)

        # ✅ REMOVED: build_vector_store() 
        # Reason: add_document_to_vector_store already saved the changes.
        # Re-running build_vector_store() would waste CPU/Time re-indexing everything.

        return {
            "status": "success",
            "message": f"Successfully indexed {len(uploaded_filenames)} new files incrementally.",
            "filenames": uploaded_filenames
        }

    except Exception as e:
        # Log the actual error for debugging
        print(f"❌ Upload Error: {e}")
        return {"error": str(e)}

@app.post("/reset")    
def reset_library():
    try:
        # Define paths
        docs_dir = os.path.join(os.getcwd(), "data", "docs")
        vector_dir = os.path.join(os.getcwd(), "backend", "vector_store")
        
        # 1. Delete physical PDF/TXT files 
        if os.path.exists(docs_dir):
            shutil.rmtree(docs_dir)
            os.makedirs(docs_dir) # Recreate empty folder
            
        # 2. Delete the FAISS index files 
        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir)
            
        # 3. Clear the in-memory vector store 
        # We need to reset the variable inside your engine
        from backend import rag_engine
        rag_engine.vector_store = None
        
        return {"status": "success", "message": "Library cleared. Ready for fresh uploads."}
    except Exception as e:
        return {"error": str(e)}