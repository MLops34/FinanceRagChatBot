# 03_embed_and_update_store.py
"""
Step 3: Load .json from temp_pickles → embed → create/update persistent FAISS index
"""

import os
import json
import time
from typing import List
from datetime import datetime
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# ──── CONFIGURATION ────
INPUT_FOLDER = "temp_pickles"
PERSIST_FAISS_DIR = r"E:\Work\FinanceRagChatBot\db\faiss_motilal"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(PERSIST_FAISS_DIR, exist_ok=True)


def load_docs_from_json(file_path: str) -> List[Document]:
    """Load documents from a .json file"""
    if not os.path.exists(file_path):
        print(f"ERROR: File not found → {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for item in data:
            if not isinstance(item, dict) or "page_content" not in item or "metadata" not in item:
                print("Warning: Invalid item in JSON → skipping")
                continue
            doc = Document(
                page_content=item["page_content"],
                metadata=item["metadata"]
            )
            docs.append(doc)

        print(f"Loaded {len(docs)} documents from {file_path}")
        return docs

    except Exception as e:
        print(f"ERROR loading JSON: {type(e).__name__}: {e}")
        return []


def load_or_create_faiss(embeddings):
    """Load existing FAISS index or return None if not found"""
    index_path = os.path.join(PERSIST_FAISS_DIR, "index.faiss")
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from: {PERSIST_FAISS_DIR}")
        try:
            return FAISS.load_local(
                PERSIST_FAISS_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Failed to load existing FAISS index: {type(e).__name__}: {e}")
            print("Will create a new one.")
    else:
        print("No existing FAISS index found → will create new one")
    
    return None


def main():
    print("=== Step 3: Embed & Update Persistent FAISS Index (JSON version) ===")
    print("This step adds your JSON documents to a persistent FAISS index.\n")

    # List all .json files
    json_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".json")]
    if not json_files:
        print(f"No .json files found in {INPUT_FOLDER}")
        print("Run data.py first to create row_docs_*.json files.")
        return

    print("Available files to embed:")
    for i, fname in enumerate(sorted(json_files), 1):
        path = os.path.join(INPUT_FOLDER, fname)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        ctime = datetime.fromtimestamp(os.path.getctime(path)).strftime("%Y-%m-%d %H:%M")
        print(f"{i}. {fname}  ({size_mb:.1f} MB, {ctime})")

    # User chooses file
    while True:
        choice = input("\nEnter number of file to add (or 'q' to quit): ").strip()
        if choice.lower() in ['q', 'quit', 'exit']:
            print("Exiting Step 3.")
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(json_files):
                selected_file = json_files[idx]
                break
            else:
                print("Invalid number.")
        except ValueError:
            print("Please enter a number or 'q'.")

    input_path = os.path.join(INPUT_FOLDER, selected_file)
    docs = load_docs_from_json(input_path)

    if not docs:
        print("No documents loaded → cannot continue.")
        return

    # Prepare embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Load or create FAISS index
    vectorstore = load_or_create_faiss(embeddings)

    start_time = time.time()
    print(f"Starting to embed and add {len(docs)} documents...")

    try:
        if vectorstore is None:
            # Create new index
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(PERSIST_FAISS_DIR)
            print(f"Created new FAISS index with {len(docs)} documents")
        else:
            # Add to existing
            vectorstore.add_documents(docs)
            vectorstore.save_local(PERSIST_FAISS_DIR)
            print(f"Added {len(docs)} new documents to existing index")

        elapsed = time.time() - start_time
        print(f"Operation complete! Took {elapsed:.2f} seconds")
        print(f"Approximate total documents: {len(vectorstore.docstore._dict)}")
        print(f"FAISS index saved/updated at: {PERSIST_FAISS_DIR}")
        print("\nNext step: Create 04_query_rag_bot.py to start asking questions!")
    
    except Exception as e:
        print(f"FAISS operation failed: {type(e).__name__}: {str(e)}")
        print("Try deleting the faiss_motilal folder and running again.")


if __name__ == "__main__":
    main()