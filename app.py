
"""
Excel Chatbot – Streamlit app
- Uses Google Gemini via LangChain for RAG-based Q&A on financial Excel documents (e.g., mutual funds, scheme portfolios)
- Converts Excel rows to JSON strings for embedding
- Embeds with GoogleGenerativeAIEmbeddings – embeddings built only once per file
- Keeps chain in session_state for instant follow-up questions
- Uses ChromaDB for vector store with persistence
"""

#------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------LIBRARIES & PACKAGES----------------------------------------------------------------------------------------#

import os
import hashlib
import logging
import streamlit as st
import pandas as pd
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI  
from langchain_community.embeddings import HuggingFaceEmbeddings




from dotenv import load_dotenv
load_dotenv()  # Loads .env file


# ---------- logging ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#

logging.basicConfig(filename="excel_chatbot.log",
                    level=logging.INFO,
                    format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)



# ---------- PAGE ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#

st.set_page_config(page_title="Financial RAG Chatbot", layout="wide")
st.header("📊 Financial Documents Chatbot")


# ---------- HELPERS FUNCTIONS ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def excel_hash(data) -> str:
    return hashlib.sha256(data.getbuffer()).hexdigest()[:16]


# ---------- SIDEBAR ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#

with st.sidebar:
    st.title("Your Documents")
    uploaded = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    
    # API key part
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.text_input("Google API Key (or set in .env file)", type="password")
        if not api_key:
            st.warning("Please enter your Google API Key (or add to .env).")
            st.stop()
    else:
        st.success("API key loaded from .env file ✅")

    # NEW: Sheet selector – only show after file upload
    if uploaded is not None:
        try:
            # Excel file load sheet names 
            excel_file = pd.ExcelFile(uploaded)
            sheet_names = excel_file.sheet_names
        except Exception as e:
            st.error(f"Sheet names load Nope: {e}")
            st.stop()

        # Dropdown to choose sheet
        selected_sheet = st.selectbox(
            "Select Sheet for Query",
            sheet_names,
            index=0,  # default first sheet
            key="selected_sheet"  # important: session state key
        )
        
        st.info(f"Selected: **{selected_sheet}**")   
    
    

# ---------- API KEY ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#
os.environ["GOOGLE_API_KEY"] = api_key


# -------------------------------------- SESSION KEYS ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#
if "excel_id" not in st.session_state:
    st.session_state.excel_id = None
if "chain" not in st.session_state: # Cache the complete chain so we do NOT rebuild it every run
    st.session_state.chain = None


# ----------------------------------------------MAIN ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#
if uploaded is None:
    st.info("👆 Upload an Excel file to start")
    st.stop()

uid = excel_hash(uploaded)

# ----------------------------------------------LOADING ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': True}
)

if uid != st.session_state.excel_id:
    log.info("New Excel uploaded %s", uid)
    st.session_state.excel_id = uid
    st.session_state.chain = None  # Wipe old chain

    # 1. Read Excel
    try:
        # Replace the old df = pd.read_excel(uploaded) with:
        df = pd.read_excel(uploaded, sheet_name=st.session_state.selected_sheet)
        sheet_names = df.sheet_names
    except Exception as e:
        st.error(f"❌ Could not read the Excel file: {e}")
        st.stop()

    selected_sheet = st.session_state.get("selected_sheet", sheet_names[0])  # default to first
    df = pd.read_excel(uploaded, sheet_name=selected_sheet)

    # Optional preview
    st.subheader(f"Data from sheet: {selected_sheet}")
    st.dataframe(df.head(20))


    if df.empty:
        st.error("❌ The Excel file is empty.")
        st.stop()

    # 2. Convert rows to JSON strings
    # chunks = [json.dumps(row) for row in json_rows] -- GIVING FULL ROW WHICH INCREASE TIME FOR OUTPUTS
    #Updates shrinking row by giving Neccessary Details.
    # NEW – keep just the columns users actually ask about
    KEEP = ["Name of Instrument","Quantity","Market value (Rs. In lakhs)","% to Net Assets","Sector / Rating"]
    json_rows = df.to_dict(orient='records')
    mini_rows = [{k: (round(v,2) if isinstance(v,float) else v) for k,v in row.items() if k in KEEP} for row in json_rows] # COlumn filter
    chunks = list(dict.fromkeys(json.dumps(m) for m in mini_rows)) # Uniquify

    # 3. Optional: Split large JSON chunks if needed
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=50
    )
    chunks = splitter.split_text("\n".join(chunks))

    # 4. Embed & index with Chroma (persistent)
    vs = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=f"financial_excel_{uid}",
        persist_directory="./chroma_db_excel"
    )
    log.info("Created new Chroma collection for %s", uid)
else:
    log.info("Same Excel %s → loading existing collection", uid)
    vs = Chroma(
        collection_name=f"financial_excel_{uid}",
        embedding_function=embeddings,
        persist_directory="./chroma_db_excel"
    )


retriever = vs.as_retriever(search_kwargs={"k": 3})

# ---------------------  -------------------------FIRST CHAIN BEFORE CACHED ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#

if st.session_state.chain is None:
    llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-r1-0528:free",   # ← correct OpenRouter ID
    temperature=0.0
)
    tmpl = ChatPromptTemplate.from_template(
        "Context (JSON rows from financial document):\n{context}\n\n"
        "Use only the context above to answer questions about the financial data (e.g., mutual funds, scheme portfolios). "
        "If the answer is not in the context, say 'I don't know.'\n\n"
        "Question: {question}\nAnswer:"
    )
    st.session_state.chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | tmpl
        | llm
        | StrOutputParser()
    )
    log.info("Chain ready for %s", uid)



# ----------------------------------------------Q/A ----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------#
chain = st.session_state.chain  # Pull cached chain
question = st.text_input("Ask a question about your Excel document")
if question:
    with st.spinner("🤖 Thinking…"):
        answer = chain.invoke(question)  # Fast: no re-loading, no re-indexing
    log.info("Q: %s  | A: %s", question, answer)
    st.success("Answer")
    st.write(answer)