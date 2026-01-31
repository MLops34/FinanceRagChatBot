# 04_chatbot_with_decision_node.py
"""
Step 4: RAG Chatbot with Decision Node for Multi-Fund Queries
- Loads FAISS index from Step 3
- Decision Node extracts fund codes from questions
- Smart routing for single/multi-fund queries
"""

import os
import re
from httpx import codes
import streamlit as st
import traceback
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from Retrieval import retrieve_chunks, get_vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PERSIST_FAISS_DIR = r"E:\Work\FinanceRagChatBot\db\faiss_motilal"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "your-key-here")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-r1"

st.set_page_config(page_title="Motilal Oswal Fund Analyzer", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False
if "last_decision" not in st.session_state:
    st.session_state.last_decision = None
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None


# ─────────────────────────────────────────────────────────────────────────────
# DECISION NODE CLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RoutingDecision:
    target_codes: List[str]
    explanation: str
    retrieve_all: bool
    k_value: int
    is_comparison: bool


class FundDecisionNode:
    def __init__(self, vs: Optional[FAISS]):
        self.vectorstore = vs
        self.code_to_name: Dict[str, str] = {}
        self.name_to_code: Dict[str, List[str]] = {}
        self._build_mappings()
        
    def _build_mappings(self):
        """
        Build fund code ↔ name mappings from all documents in the vectorstore.
        Uses both full name and the pre-computed 'fund_name_normalized' field.
        """
        if not self.vectorstore:
            return

        self.code_to_name = {}
        self.name_to_code = {}

        for doc in self.vectorstore.docstore._dict.values():
            code = doc.metadata.get("fund_code")
            if not code:
                continue

            code = code.upper()
            name = doc.metadata.get("fund_name", "").strip()

        # Skip if we already have this code
            if code in self.code_to_name:
                continue

            self.code_to_name[code] = name

        # ── Create good matching variations ────────────────────────────────────────
            normalized = doc.metadata.get("fund_name_normalized", name.lower().strip())

            variations = [
                normalized,                                 # "flexi cap"
                normalized.replace(" ", ""),                # "flexicap"
                normalized.replace(" ", "-"),               # "flexi-cap"
                normalized.replace("(", "").replace(")", ""),  # remove parentheses
                name.lower(),
                name.lower().replace("motilal oswal ", ""),
                name.lower().replace("motilal oswal", ""),
                name.lower().replace(" fund", ""),
                name.lower().replace(" scheme", ""),
                name.lower().replace(" direct", ""),
                name.lower().replace(" growth", ""),
              name.lower().replace("motilal oswal ", "").replace(" fund", ""),
            ]

        # Add meaningful individual words (longer than 3 chars, exclude noise)
            words = name.split()
            noise_words = {
            "motilal", "oswal", "fund", "direct", "growth", "scheme", "plan",
            "regular", "(formerly", "known", "as", "long", "term", "equity",
            "saver", "tax", "elss", "formerly"
            }

            for word in words:
                wl = word.lower().strip("().,-;:")
                if len(wl) > 3 and wl not in noise_words:
                    variations.append(wl)

        # Common user-friendly / short-form aliases
            lower_norm = normalized.lower()
            if "flexi" in lower_norm:
                variations.extend([
                     "flexicap", "flexi cap fund", "flexi-cap fund", "flexi cap"
                ])
            if "elss" in lower_norm or "tax saver" in lower_norm:
                variations.extend([
                    "elss", "tax saver", "tax saving", "elss tax saver",
                    "tax saver fund", "elss fund"
                ])
            if "midcap" in lower_norm:
                variations.append("mid cap")
            if "smallcap" in lower_norm:
                variations.append("small cap")

        # Final cleanup: remove duplicates and empty strings
            variations = [v.strip() for v in variations if v.strip()]

        # Assign all valid variations to this code
            for var in set(variations):
                if var not in self.name_to_code:
                    self.name_to_code[var] = []
                if code not in self.name_to_code[var]:
                    self.name_to_code[var].append(code)

    def extract_fund_codes(self, question: str) -> RoutingDecision:
        qu = question.upper().strip()
        ql = question.lower().strip()
        detected = []
        
        # Pattern: YO##, MO##, etc.
        for match in re.findall(r"\b([A-Z]{1,3}\d{2,4})\b", qu):
            if match in self.code_to_name:
                detected.append(match)
        
        # Name matching
        detected = []

        for name_var, codes in self.name_to_code.items():
    # Try with word boundaries
            if re.search(r'\b' + re.escape(name_var) + r'\b', ql, re.IGNORECASE):
                detected.extend(codes)
                continue

        # Also try important cleaned versions
        cleaned = name_var.replace("motilal oswal ", "").replace(" fund", "").strip()
        if cleaned and re.search(r'\b' + re.escape(cleaned) + r'\b', ql, re.IGNORECASE):
            detected.extend(codes)
        
        # Deduplicate
        seen = set()
        unique = [c for c in detected if not (c in seen or seen.add(c))]
        
        # Detect query type
        comp_terms = ["compare", "vs", "versus", "between", "difference", "better", "worse", "higher", "lower", "both", "which is"]
        multi_terms = ["all funds", "every fund", "across funds", "funds have", "multiple funds"]
        
        is_comp = any(t in ql for t in comp_terms)
        is_multi = any(t in ql for t in multi_terms)
        
        if unique:
            if len(unique) == 1:
                return RoutingDecision(unique, f"Single: {unique[0]}", False, 100, False)
            return RoutingDecision(unique, f"Multiple: {', '.join(unique)}", False, 50*len(unique), True)
        
        if is_comp or is_multi:
            return RoutingDecision([], "Comparison → ALL", True, 100, True)
        
        return self._llm_fallback(question)
    
    def _llm_fallback(self, question: str) -> RoutingDecision:
        if not self.code_to_name:
            return RoutingDecision([], "No funds", True, 50, False)
        
        funds = "\n".join([f"• {c}: {n}" for c, n in sorted(self.code_to_name.items())])
        
        prompt = f"""You are a strict fund code extractor.

Known Motilal Oswal funds:
{funds}

User question: "{question}"

Instructions - choose the MOST SPECIFIC answer possible:
- If question mentions ONE specific fund name or code → return only that code
- If question clearly compares or mentions 2-3 funds by name/code → return those codes, comma separated
- If question is about a category (midcap, smallcap, flexicap, largecap, hybrid, etc.) without naming a specific fund → return ALL
- If question is general ("best fund", "all funds", "portfolio") → return ALL
- If no fund is clearly mentioned → return ALL

Examples:
"NAV of flexi cap fund"                    → YO08
"Persistent Systems holding"               → YO08
"Compare midcap and flexi cap"             → ALL
"Which fund is better YO08 or YO46"        → YO08,YO46
"Top holdings of motilal oswal funds"      → ALL

Return ONLY: one code, comma-separated codes, or the word ALL
Nothing else.
"""
        
        try:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0,
                max_tokens=50
            )
            result = llm.predict(prompt).strip().upper()
            
            if result == "ALL" or not result:
                return RoutingDecision([], "LLM: ALL", True, 150, False)
            
            codes = [c.strip() for c in result.split(",") if c.strip()]
            valid = [c for c in codes if c in self.code_to_name]
            if valid:
                return RoutingDecision(
                    valid,
                    f"LLM: {', '.join(valid)}",
                    False,
                    40 * len(valid),
                    len(valid) > 1
                )
        except Exception as e:
            st.warning(f"LLM fail: {e}")
        
        return RoutingDecision([], "Fallback ALL", True, 80, False)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD VECTORSTORE (FIXED: No global issues)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not os.path.exists(PERSIST_FAISS_DIR):
        return None
    try:
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(PERSIST_FAISS_DIR, emb, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"FAISS load error: {e}")
        return None


# Initialize at module level (NO global keyword needed)
vectorstore = load_vectorstore()
decision_node = FundDecisionNode(vectorstore)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_with_decision(question: str) -> Tuple[List[Document], RoutingDecision]:
    if not vectorstore:
        return [], RoutingDecision([], "No VS", True, 0, False)
    
    decision = decision_node.extract_fund_codes(question)
    
    if decision.retrieve_all:
        sk = {"k": decision.k_value}
    else:
        codes = decision.target_codes
        fn = (lambda m: m.get("fund_code") == codes[0]) if len(codes) == 1 else (lambda m: m.get("fund_code") in codes)
        sk = {"k": decision.k_value, "filter": fn}
    
    retriever = vectorstore.as_retriever(search_kwargs=sk)
    docs = retriever.invoke(question)
    return docs, decision


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(No docs found)"
    
    groups = {}
    for d in docs:
        code = d.metadata.get("fund_code", "UNK")
        name = d.metadata.get("fund_name", "Unknown")
        groups.setdefault(f"{name} ({code})", []).append(d)
    
    parts = [f"Data from {len(groups)} fund(s):\n"]
    for key, gdocs in sorted(groups.items()):
        parts.append(f"\n=== {key} ({len(gdocs)} entries) ===")
        for i, d in enumerate(gdocs[:8], 1):
            c = d.page_content[:350]
            parts.append(f"{i}. {c}{'...' if len(d.page_content)>350 else ''}")
        if len(gdocs) > 8:
            parts.append(f"... and {len(gdocs)-8} more")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────────────────────────────────────────
answer_llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENROUTER_API_KEY,
                       openai_api_base=OPENROUTER_BASE_URL, temperature=0.2, max_tokens=800)

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a mutual fund analyst specializing in Motilal Oswal funds.

CONTEXT (chunks are row data separated by • ; percentages are like 0.10178 = 10.18%):
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer ONLY from context. If info missing, say "Data not available for [fund/code]".
- Cite fund names, codes (e.g., YO08), and exact holdings/values.
- For comparisons: use markdown tables with columns like Fund, Holding, % Allocation, Sector.
- Parse • separated rows: e.g., Name • ISIN • Sector • Quantity • Value • % • etc.
- Highlight differences in bold.

ANALYSIS:""")

def build_chain():
    def retrieve_step(inputs):
        q = inputs["question"]
        docs, dec = retrieve_with_decision(q)
        st.session_state.last_decision = dec
        st.session_state.last_retrieval = {
            "count": len(docs),
            "funds": sorted(set(d.metadata.get("fund_code","UNK") for d in docs))
        }
        return {"context": format_docs(docs), "question": q}
    
    return RunnablePassthrough() | RunnableLambda(retrieve_step) | RAG_PROMPT | answer_llm | StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR UI
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Status")
    
    if vectorstore:
        st.success(f"✅ FAISS: {len(vectorstore.docstore._dict):,} docs")
        with st.expander("Funds"):
            for c, n in sorted(decision_node.code_to_name.items()):
                st.write(f"**{c}**: {n}")
    else:
        st.error("❌ No FAISS index")
    
    st.markdown("---")
    st.session_state.show_debug = st.checkbox("🔍 Debug", value=st.session_state.show_debug)
    
    if st.session_state.show_debug and st.session_state.last_decision:
        d = st.session_state.last_decision
        st.write(f"**Logic:** {d.explanation}")
        st.write(f"**Codes:** {d.target_codes or 'ALL'}")
        st.write(f"**K:** {d.k_value}")

        if st.session_state.last_retrieval:
            st.write(f"**Retrieved:** {st.session_state.last_retrieval['count']} chunks from funds: {', '.join(st.session_state.last_retrieval['funds'])}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧠 Motilal Oswal Fund Analyzer")
st.caption("Decision Node routing for multi-fund queries")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask (e.g., 'Compare YO46 vs YO47')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing..."):
            if not vectorstore:
                resp = "❌ Error: Run Step 3 (Embed) first to create FAISS index."
            else:
                try:
                    chain = build_chain()
                    resp = chain.invoke({"question": prompt})
                except Exception as e:
                    resp = f"❌ Error: {e}\n\n```{traceback.format_exc()}```"
            
            st.markdown(resp)
            
            if st.session_state.show_debug and st.session_state.last_retrieval:
                info = st.session_state.last_retrieval
                with st.expander("🔍 Details"):
                    st.write(f"Docs: {info['count']}")
                    st.write(f"Funds: {', '.join(info['funds']) or 'None'}")
    
    st.session_state.messages.append({"role": "assistant", "content": resp})