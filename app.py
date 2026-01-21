# app.py
import os
import streamlit as st
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
 
INDEX_DIR = "faiss_index_motilal"
os.makedirs(INDEX_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "deepseek/deepseek-r1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="Motilal Oswal Portfolio RAG", layout="wide")

# ────────────────────────────────────────────────
# SECRETS & LLM
# ────────────────────────────────────────────────

try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    st.error("OPENROUTER_API_KEY not found in .streamlit/secrets.toml")
    st.stop()

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=api_key,
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=0.15,
    max_tokens=1800
)

# ────────────────────────────────────────────────
# EMBEDDINGS (cached)
# ────────────────────────────────────────────────

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

embeddings = get_embeddings()

# ────────────────────────────────────────────────
# VECTOR DB MANAGEMENT (using session_state)
# ────────────────────────────────────────────────

def load_or_create_vector_db():
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    if os.path.exists(index_path):
        try:
            db = FAISS.load_local(
                INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.session_state["vector_db_loaded"] = True
            return db
        except Exception as e:
            st.warning(f"Could not load FAISS index: {e}. Will create new one on upload.")
    return None

# Initialize in session state
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = load_or_create_vector_db()

vector_db = st.session_state["vector_db"]

# ────────────────────────────────────────────────
# PROMPT
# ────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_template("""
You are an expert mutual fund analyst using Motilal Oswal December 2025 portfolio data.
Answer accurately using **only** the provided context.
Use clear structure, bullets or tables when helpful.
If information is missing, say so clearly.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'Unknown')} | Sheet: {doc.metadata.get('sheet', '?')}] {doc.page_content}"
        for doc in docs
    )

# ────────────────────────────────────────────────
# RAG CHAIN
# ────────────────────────────────────────────────

def get_rag_chain(db):
    if db is None:
        return None
    retriever = db.as_retriever(search_kwargs={"k": 8})
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ────────────────────────────────────────────────
# EXCEL → Documents
# ────────────────────────────────────────────────

def excel_to_docs(file_path: str):
    docs = []
    try:
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
            # Very basic conversion – improve later if needed
            text = df.fillna("").astype(str).apply(lambda row: " | ".join(row.values), axis=1).str.cat(sep="\n")
            doc = Document(
                page_content=text.strip(),
                metadata={
                    "source": Path(file_path).name,
                    "sheet": sheet_name,
                    "file_type": "excel"
                }
            )
            docs.append(doc)
    except Exception as e:
        st.error(f"Failed to parse {Path(file_path).name}: {e}")
    return docs

# ────────────────────────────────────────────────
# SIDEBAR – Upload
# ────────────────────────────────────────────────

with st.sidebar:
    st.header("Index Management")

    uploaded_files = st.file_uploader(
        "Upload Motilal Oswal Excel files",
        type=["xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process & Index Files", type="primary"):
        with st.status("Processing files...", expanded=True) as status:
            all_new_docs = []

            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                status.write(f"Reading {uploaded_file.name}...")
                docs = excel_to_docs(temp_path)
                all_new_docs.extend(docs)

                try:
                    os.remove(temp_path)
                except:
                    pass

            if not all_new_docs:
                status.update(label="No valid documents extracted", state="error")
            else:
                status.update(label=f"Extracted {len(all_new_docs)} sheets. Chunking...", state="running")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " | ", ". ", " ", ""]
                )
                chunks = splitter.split_documents(all_new_docs)

                status.update(label=f"Creating / updating index ({len(chunks)} chunks)...", state="running")

                if st.session_state["vector_db"] is None:
                    st.session_state["vector_db"] = FAISS.from_documents(chunks, embeddings)
                else:
                    st.session_state["vector_db"].add_documents(chunks)

                st.session_state["vector_db"].save_local(INDEX_DIR)
                status.update(
                    label=f"Done! Total documents: {len(st.session_state['vector_db'].docstore._dict)}",
                    state="complete"
                )

    st.markdown("---")
    count = len(st.session_state["vector_db"].docstore._dict) if st.session_state.get("vector_db") else 0
    if count > 0:
        st.success(f"Index ready – {count} documents")
    else:
        st.info("Upload files and click 'Process & Index Files'")

# ────────────────────────────────────────────────
# MAIN CHAT
# ────────────────────────────────────────────────

st.title("Motilal Oswal Portfolio Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about any scheme, holding, sector, allocation..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            chain = get_rag_chain(st.session_state["vector_db"])
            if chain is None:
                response = "No data indexed yet.\n\nPlease upload Excel files in the sidebar and process them."
            else:
                try:
                    response = chain.invoke(user_input)
                except Exception as e:
                    response = f"Error during query: {str(e)}"

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})