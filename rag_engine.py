import os
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
VECTOR_DIR = BASE_DIR / "backend" / "vector_store"

vector_store = None


def load_documents():
    documents = []

    BASE_DIR = Path(__file__).resolve().parent.parent
    DOCS_DIR = BASE_DIR / "data" / "docs"

    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"❌ Docs folder not found: {DOCS_DIR}")

    for file in DOCS_DIR.iterdir():
        try:
            if file.suffix.lower() == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
                docs = loader.load()

            elif file.suffix.lower() == ".pdf":
                if file.stat().st_size == 0:
                    print(f"⚠️ Skipping empty PDF: {file.name}")
                    continue
                loader = PyPDFLoader(str(file))
                docs = loader.load()

            else:
                continue

            # ✅ ENSURE METADATA EXISTS
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = file.name

            documents.extend(docs)

        except Exception as e:
            print(f"⚠️ Skipping {file.name}: {e}")

    if not documents:
        raise ValueError("❌ No valid documents found")

    print(f"📄 Loaded {len(documents)} documents")
    return documents



def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"✂️ Created {len(chunks)} chunks")
    return chunks

def build_vector_store():
    global vector_store

    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)

        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(VECTOR_DIR))

    print("💾 Vector store built and saved successfully")

    return vector_store

def load_vector_store():
    global vector_store

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        str(VECTOR_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✅ Vector store loaded successfully")
    return vector_store

def add_document_to_vector_store(file_path):
    global vector_store

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 1. Load the specific new file
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported format. Only PDF and TXT.")

    documents = loader.load()
    
    # ✅ ENSURE METADATA IS ADDED TO NEW CHUNKS
    for d in documents:
        d.metadata["source"] = os.path.basename(file_path)

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # 3. Add to existing store or create a new one using VECTOR_DIR
    if vector_store is None:
        if VECTOR_DIR.exists():
            vector_store = FAISS.load_local(
                str(VECTOR_DIR), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    # 4. Save to the correct global directory
    vector_store.save_local(str(VECTOR_DIR))
    print(f"✅ Added {len(chunks)} new chunks from {os.path.basename(file_path)} to {VECTOR_DIR}")


def retrieve_context(query, k=10):
    global vector_store

    if vector_store is None:
        load_vector_store()
    return vector_store.similarity_search(query, k=k)


if __name__ == "__main__":
    build_vector_store()
    results = retrieve_context("What is this document about?")
    for r in results:
        print(r.page_content[:200])
