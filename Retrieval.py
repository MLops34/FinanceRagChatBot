# debug_retrieval.py
"""
Utility module to inspect what chunks/documents are actually retrieved
Can be imported into inter-04.py or run standalone for testing.
"""

import os
from typing import List, Tuple

import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (should match your inter-04.py)
# ─────────────────────────────────────────────────────────────────────────────
PERSIST_FAISS_DIR = r"E:\Work\FinanceRagChatBot\db\faiss_motilal"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """Load FAISS once (cached)"""
    if not os.path.exists(PERSIST_FAISS_DIR):
        st.error(f"FAISS directory not found: {PERSIST_FAISS_DIR}")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(
            PERSIST_FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load FAISS: {e}")
        return None


def retrieve_chunks(
    question: str,
    decision_node=None,           # optional: your FundDecisionNode instance
    k: int = None,
    filter_codes: List[str] = None,
    show_in_ui: bool = True
) -> Tuple[List[Document], dict]:
    """
    Retrieve chunks and optionally display them nicely in Streamlit.
    
    Args:
        question: user query
        decision_node: your FundDecisionNode (if you want to use smart routing)
        k: override number of results (if decision_node is None)
        filter_codes: list of fund_codes to filter on (if decision_node is None)
        show_in_ui: whether to render expander with chunks
    
    Returns:
        (retrieved documents, debug_info dict)
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        return [], {"error": "No vectorstore loaded"}

    # ── Decide how to retrieve ───────────────────────────────────────
    if decision_node is not None:
        decision = decision_node.extract_fund_codes(question)
        debug_info = {
            "decision_explanation": decision.explanation,
            "target_codes": decision.target_codes,
            "retrieve_all": decision.retrieve_all,
            "k_value": decision.k_value,
            "is_comparison": decision.is_comparison
        }

        if decision.retrieve_all:
            search_kwargs = {"k": decision.k_value}
        else:
            codes = decision.target_codes
            if len(codes) == 1:
                filter_fn = lambda m: m.get("fund_code") == codes[0]
            else:
                filter_fn = lambda m: m.get("fund_code") in codes
            search_kwargs = {"k": decision.k_value, "filter": filter_fn}
    else:
        # manual / fallback mode
        search_kwargs = {"k": k or 8}
        if filter_codes:
            if len(filter_codes) == 1:
                search_kwargs["filter"] = lambda m: m.get("fund_code") == filter_codes[0]
            else:
                search_kwargs["filter"] = lambda m: m.get("fund_code") in filter_codes
        
        debug_info = {
            "decision_explanation": "Manual retrieval (no decision node)",
            "target_codes": filter_codes or "ALL",
            "k_value": search_kwargs["k"]
        }

    # ── Actually retrieve ─────────────────────────────────────────────
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)

    debug_info.update({
        "retrieved_count": len(docs),
        "fund_codes_found": sorted(set(d.metadata.get("fund_code", "UNK") for d in docs)),
    })

    # ── Display in UI (optional) ──────────────────────────────────────
    if show_in_ui and docs:
        with st.expander(f"📜 Retrieved {len(docs)} chunks", expanded=True):
            for i, doc in enumerate(docs, 1):
                code = doc.metadata.get("fund_code", "???")
                name = doc.metadata.get("fund_name", "Unknown")
                date = doc.metadata.get("portfolio_date", "-")
                
                header = f"**Chunk {i}**  •  {name} ({code})  •  {date}"
                st.markdown(header)
                st.caption(f"row {doc.metadata.get('row_number', '?')}, chunk {doc.metadata.get('chunk_id', '?')}")
                st.code(doc.page_content.strip()[:1200] + (" …" if len(doc.page_content) > 1200 else ""), language="text")
                st.markdown("---")

    elif show_in_ui and not docs:
        st.warning("No chunks retrieved.")

    return docs, debug_info


# ── Standalone test / debug mode ────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="Retrieval Debugger", layout="wide")
    
    st.title("Retrieval Chunk Inspector"  )
    st.caption("Quick way to see what chunks would be retrieved for a question")
    
    question = st.text_input("Question:", 
        "What are the top holdings of Motilal Oswal Flexi Cap Fund?")
    
    use_smart_routing = st.checkbox("Use smart decision node routing", value=True)
    
    if st.button("Retrieve & Show Chunks") and question:
        if use_smart_routing:
            # You would normally import your decision node here
            # For standalone testing we fall back to simple retrieval
            st.info("Smart routing not available in standalone mode → using simple k=12 retrieval")
            docs, info = retrieve_chunks(question, k=12, show_in_ui=True)
        else:
            docs, info = retrieve_chunks(question, k=12, show_in_ui=True)
        
        st.subheader("Summary")
        st.json(info)