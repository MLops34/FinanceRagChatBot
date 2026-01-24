# app.py - Streamlit RAG Chatbot with file upload & DeepSeek R1 (OpenRouter)

import os
import pickle
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ──── CONFIGURATION ────
FAISS_INDEX_DIR = r"E:\Work\FinanceRagChatBot\db\faiss_motilal"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenRouter + DeepSeek R1
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "your-key-here")  # ← add to .streamlit/secrets.toml or replace
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-r1"

st.set_page_config(page_title="Portfolio RAG Chatbot", layout="wide")

# ──── LOAD / INITIALIZE ────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def load_or_create_faiss(_embeddings):
    if os.path.exists(FAISS_INDEX_DIR):
        try:
            db = FAISS.load_local(
                FAISS_INDEX_DIR,
                _embeddings,
                allow_dangerous_deserialization=True
            )
            st.session_state['faiss_loaded'] = True
            return db
        except Exception as e:
            st.error(f"Failed to load existing FAISS: {e}")
    return None

embeddings = load_embeddings()
vectorstore = load_or_create_faiss(embeddings)

if vectorstore is None:
    st.warning("No FAISS index found. Upload a .pkl file to create one.")
    vectorstore = None

# ──── LLM SETUP ────
llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=0.1,
    max_tokens=1500
)

prompt = ChatPromptTemplate.from_template("""
You are an expert mutual fund analyst using Motilal Oswal December 2025 portfolio data.
Answer accurately using only the provided context.
Use clear structure, bullets or tables when helpful.
If information is missing, say so.

Context:
{context}

Question: {question}
                                          
Answer clearly, include fund name/code for each holding mentioned.
When comparing funds, show side-by-side tables or lists.
Use % to Net Assets for "top holdings".
If information for a fund is missing, say so.
""")

def format_docs_with_fund(docs):
    formatted = []
    for doc in docs:
        fund_name = doc.metadata.get("fund_name", "Unknown Fund")
        fund_code = doc.metadata.get("fund_code", "Unknown")
        content = doc.page_content
        formatted.append(
            f"Fund: {fund_name} ({fund_code})\n"
            f"{content}\n"
            f"Percentage: {content.split('•')[-2].strip() if '•' in content else 'N/A'} % to Net Assets\n"
            "---"
        )
    return "\n\n".join(formatted)

# Then context = format_docs_with_fund(retrieved_docs)

def get_rag_chain(question: str = None):   # ← add optional question param
    if vectorstore is None:
        return None

    # Default retriever (no filter)
    search_kwargs = {"k": 30}

    # If question is provided and looks like a comparison → force both funds
    if question:
        q_lower = question.lower()
        if any(word in q_lower for word in ["compare", "vs", "between", "top of", "holdings of both"]):
            # Force retrieval from both funds (YO16 = Midcap 150, YO17 = Smallcap 250)
            search_kwargs["filter"] = {
                "fund_code": {"$in": ["YO16", "YO17"]}
            }
            search_kwargs["k"] = 40   # get more results so both funds are well represented

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    return (
        {"context": retriever | format_docs_with_fund, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ──── STREAMLIT UI ────
st.title("Financial-RAG Chatbot")
st.markdown("Ask questions about Motilal Oswal holdings or upload new files to add funds.")

# Sidebar - Upload & status
with st.sidebar:
    st.header("Manage Index")
    uploaded_file = st.file_uploader("Upload new .pkl file", type=["json"])

    if uploaded_file:
        try:
            new_docs = pickle.load(uploaded_file)
            if not isinstance(new_docs, list) or not all(isinstance(d, Document) for d in new_docs):
                st.error("File does not contain valid list[Document]")
            else:
                if vectorstore is None:
                    st.info("Creating new FAISS index...")
                    vectorstore = FAISS.from_documents(new_docs, embeddings)
                else:
                    st.info(f"Adding {len(new_docs)} new documents...")
                    vectorstore.add_documents(new_docs)
                
                vectorstore.save_local(FAISS_INDEX_DIR)
                st.success(f"Added {len(new_docs)} documents. Total now ≈ {len(vectorstore.docstore._dict)}")
                st.session_state['faiss_loaded'] = True
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.markdown("---")
    if vectorstore:
        st.success(f"Index loaded — {len(vectorstore.docstore._dict)} documents")
    else:
        st.warning("No index yet — upload a file to start")

# Main chat area
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("Ask about the portfolio..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = get_rag_chain()
            if chain is None:
                response = "No portfolio data loaded yet. Please upload a .pkl file first."
            else:
                try:
                    response = chain.invoke(prompt_text)
                except Exception as e:
                    response = f"Error during query: {type(e).__name__}: {e}"
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})


import json
uploaded_file = st.file_uploader("Upload new .json file", type=["json"])

if uploaded_file:
    try:
        raw_data = json.load(uploaded_file)

        if not isinstance(raw_data, list):
            st.error("JSON must be a list of documents")
        else:
            new_docs = []
            for item in raw_data:
                if "page_content" not in item:
                    continue
                new_docs.append(
                    Document(
                        page_content=item["page_content"],
                        metadata=item.get("metadata", {})
                    )
                )

            if not new_docs:
                st.error("No valid documents found in JSON")
            else:
                if vectorstore is None:
                    st.info("Creating new FAISS index...")
                    vectorstore = FAISS.from_documents(new_docs, embeddings)
                else:
                    st.info(f"Adding {len(new_docs)} documents...")
                    vectorstore.add_documents(new_docs)

                vectorstore.save_local(FAISS_INDEX_DIR)
                st.success(
                    f"Added {len(new_docs)} documents. "
                    f"Total ≈ {len(vectorstore.docstore._dict)}"
                )
                st.session_state["faiss_loaded"] = True

    except json.JSONDecodeError:
        st.error("Invalid JSON file")
    except Exception as e:
        st.error(f"Upload failed: {e}")
