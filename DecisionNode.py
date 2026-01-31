# DecisionNode.py
"""
Streamlit RAG Chatbot with Intelligent Decision Node for Motilal Oswal Funds
- Loads FAISS index
- Uses FundDecisionNode to decide: single fund / multiple funds / all funds
- Filters retrieval accordingly
- Shows debug info + retrieved fund codes
"""

import os
import re
import json
import streamlit as st
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
PERSIST_DIR     = r"E:\Work\FinanceRagChatBot\db\faiss_motilal"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

OPENROUTER_KEY  = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
OPENROUTER_URL  = "https://openrouter.ai/api/v1"
LLM_MODEL = "arcee-ai/trinity-mini:free"
st.set_page_config(
    page_title="Motilal Oswal Fund Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "show_debug": False,
        "last_decision": None,
        "last_retrieval": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ────────────────────────────────────────────────
# DECISION DATACLASS & NODE
# ────────────────────────────────────────────────
@dataclass
class RoutingDecision:
    target_codes: List[str]
    explanation: str
    retrieve_all: bool
    k_value: int
    is_comparison: bool


class FundDecisionNode:
    def __init__(self, vectorstore: Optional[FAISS] = None):
        self.vectorstore = vectorstore
        self.code_to_name: Dict[str, str] = {}
        self.name_to_code: Dict[str, List[str]] = {}
        self._build_mappings()

    def _build_mappings(self):
        if not self.vectorstore:
            return

        for doc in self.vectorstore.docstore._dict.values():
            code = doc.metadata.get("fund_code")
            name = doc.metadata.get("fund_name", "")
            normalized = doc.metadata.get("fund_name_normalized", name.lower().strip())

            if not code:
                continue
            code = code.upper()

            if code not in self.code_to_name:
                self.code_to_name[code] = name

            variations = [
                normalized,
                normalized.replace(" ", ""),
                normalized.replace(" ", "-"),
                name.lower(),
                name.lower().replace("motilal oswal ", ""),
                name.lower().replace(" fund", ""),
                name.lower().replace("motilal oswal ", "").replace(" fund", ""),
            ]

            # meaningful words
            noise = {"motilal", "oswal", "fund", "direct", "growth", "scheme", "plan", "regular"}
            for w in name.split():
                wl = w.lower().strip("().,-")
                if len(wl) > 3 and wl not in noise:
                    variations.append(wl)

            # popular aliases
            nl = normalized.lower()
            if "flexi" in nl:
                variations.extend(["flexicap", "flexi cap", "flexi-cap"])
            if "elss" in nl or "tax saver" in nl:
                variations.extend(["elss", "tax saver", "tax saving", "elss tax saver"])

            variations = [v.strip() for v in variations if v.strip()]

            for var in set(variations):
                if var not in self.name_to_code:
                    self.name_to_code[var] = []
                if code not in self.name_to_code[var]:
                    self.name_to_code[var].append(code)

    def decide(self, question: str) -> RoutingDecision:
        q_upper = question.upper().strip()
        q_lower = question.lower().strip()
        detected = []

        # 1. Direct code match (YO08, MO12, etc.)
        for m in re.findall(r"\b([A-Z]{2,4}\d{2,4})\b", q_upper):
            if m in self.code_to_name:
                detected.append(m)

        # 2. Name / alias matching
        for var, codes in self.name_to_code.items():
            if re.search(r'\b' + re.escape(var) + r'\b', q_lower, re.IGNORECASE):
                detected.extend(codes)

        detected = list(dict.fromkeys(detected))  # preserve order, remove duplicates

        # 3. Keyword-based classification
        comp_kw = ["compare", "vs", "versus", "between", "difference", "better", "worse", "which is better"]
        multi_kw = ["all funds", "every fund", "across funds", "multiple funds", "all schemes"]

        is_comparison = any(kw in q_lower for kw in comp_kw)
        is_multi     = any(kw in q_lower for kw in multi_kw)

        if detected:
            if len(detected) == 1:
                code = detected[0]
                name = self.code_to_name.get(code, "Unknown")
                return RoutingDecision(
                    target_codes=[code],
                    explanation=f"Single fund detected: {code} ({name})",
                    retrieve_all=False,
                    k_value=100,
                    is_comparison=False
                )
            else:
                return RoutingDecision(
                    target_codes=detected,
                    explanation=f"Multiple funds: {', '.join(detected)}",
                    retrieve_all=False,
                    k_value=60 * len(detected),
                    is_comparison=True
                )

        if is_comparison or is_multi:
            return RoutingDecision(
                target_codes=[],
                explanation="Comparison / multi-fund query → retrieve ALL",
                retrieve_all=True,
                k_value=150,
                is_comparison=True
            )

        # 4. LLM fallback for ambiguous cases
        return self._llm_decide(question)

    def _llm_decide(self, question: str) -> RoutingDecision:
        if not self.code_to_name:
            return RoutingDecision([], "No funds loaded", True, 80, False)

        funds_list = "\n".join([f"{c}: {n}" for c, n in sorted(self.code_to_name.items())])

        prompt = f"""You are a precise fund router for Motilal Oswal schemes.

Known funds:
{funds_list}

User question: {question}

Rules:
- Return ONLY valid JSON, nothing else
- Keys: "decision", "fund_codes", "explanation"
- decision: "single", "multiple", "all"
- fund_codes: list of codes (empty if "all")
- explanation: short reason

Examples:
"NAV of flexi cap"         → {{"decision":"single", "fund_codes":["YO08"], "explanation":"flexi cap → YO08"}}
"Compare midcap and elss"  → {{"decision":"all",    "fund_codes":[],     "explanation":"category comparison"}}
"Persistent Systems fund"  → {{"decision":"single", "fund_codes":["YO08"], "explanation":"holding points to flexi cap"}}

Output JSON only:"""

        try:
            llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENROUTER_KEY,
                openai_api_base=OPENROUTER_URL,
                temperature=0.1,
                max_tokens=150
            )
            response = llm.invoke(prompt).content.strip()

            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                decision = data.get("decision", "all").lower()
                codes = data.get("fund_codes", [])
                valid_codes = [c for c in codes if c in self.code_to_name]

                if decision == "single" and valid_codes:
                    return RoutingDecision(
                        target_codes=valid_codes[:1],
                        explanation=data.get("explanation", "LLM single"),
                        retrieve_all=False,
                        k_value=100,
                        is_comparison=False
                    )
                elif decision == "multiple" and valid_codes:
                    return RoutingDecision(
                        target_codes=valid_codes,
                        explanation=data.get("explanation", "LLM multiple"),
                        retrieve_all=False,
                        k_value=60 * len(valid_codes),
                        is_comparison=True
                    )
                else:
                    return RoutingDecision([], "LLM fallback → ALL", True, 150, False)

        except Exception as e:
            st.warning(f"LLM decision failed: {str(e)}")

        return RoutingDecision([], "LLM error → ALL", True, 150, False)


# ────────────────────────────────────────────────
# LOAD VECTOR STORE
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_faiss():
    if not os.path.exists(PERSIST_DIR):
        return None
    try:
        emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return FAISS.load_local(
            PERSIST_DIR, emb, allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"FAISS load failed: {e}")
        return None


vectorstore = load_faiss()
decision_node = FundDecisionNode(vectorstore)


# ────────────────────────────────────────────────
# RETRIEVAL + FORMATTING
# ────────────────────────────────────────────────
def retrieve(decision: RoutingDecision, query: str) -> List[Document]:
    if not vectorstore:
        return []

    if decision.retrieve_all:
        kwargs = {"k": decision.k_value}
    else:
        codes = decision.target_codes
        if len(codes) == 1:
            filt = lambda m: m.get("fund_code") == codes[0]
        else:
            filt = lambda m: m.get("fund_code") in codes
        kwargs = {"k": decision.k_value, "filter": filt}

    retriever = vectorstore.as_retriever(search_kwargs=kwargs)
    return retriever.invoke(query)


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(no relevant documents found)"

    groups = {}
    for d in docs:
        code = d.metadata.get("fund_code", "UNK")
        name = d.metadata.get("fund_name", "Unknown")
        key = f"{name} ({code})"
        groups.setdefault(key, []).append(d.page_content.strip())

    parts = [f"Retrieved data from {len(groups)} fund(s):\n"]
    for key, chunks in sorted(groups.items()):
        parts.append(f"\n=== {key} ({len(chunks)} chunks) ===")
        for i, chunk in enumerate(chunks, 1):
            preview = chunk[:400] + ("..." if len(chunk) > 400 else "")
            parts.append(f"{i}. {preview}")
    return "\n".join(parts)


# ────────────────────────────────────────────────
# RAG CHAIN
# ────────────────────────────────────────────────
llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=OPENROUTER_KEY,
    openai_api_base=OPENROUTER_URL,
    temperature=0.15,
    max_tokens=1800
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert Motilal Oswal mutual fund analyst.

CONTEXT (grouped by fund):
{context}

QUESTION:
{question}

Rules:
- Answer using ONLY the provided context
- Cite fund names and codes (YO08, YO09, ...)
- Use markdown tables for comparisons and lists
- Be precise with numbers and percentages
- If data is missing say clearly "Data not available in current knowledge base"

Answer:"""
)


def build_rag_chain():
    def retrieve_step(inputs):
        q = inputs["question"]
        decision = decision_node.decide(q)

        st.session_state.last_decision = decision

        docs = retrieve(decision, q)

        funds = sorted(set(d.metadata.get("fund_code", "UNK") for d in docs))

        st.session_state.last_retrieval = {
            "count": len(docs),
            "funds": funds
        }

        return {
            "context": format_docs(docs),
            "question": q
        }

    return (
        RunnablePassthrough()
        | RunnableLambda(retrieve_step)
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls & Status")

    if vectorstore:
        st.success(f"FAISS loaded — {len(vectorstore.docstore._dict):,} chunks")
        with st.expander("Known Funds"):
            for c, n in sorted(decision_node.code_to_name.items()):
                st.write(f"**{c}** — {n}")
    else:
        st.error("No FAISS index found. Run embedding step first.")

    st.markdown("---")
    st.session_state.show_debug = st.checkbox("Show debug info", value=False)


# ────────────────────────────────────────────────
# MAIN UI
# ────────────────────────────────────────────────
st.title("Motilal Oswal Fund Analyzer")
st.caption("Smart routing • single-fund / multi-fund / portfolio-wide questions")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about funds, holdings, comparisons..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            if not vectorstore:
                answer = "Knowledge base not loaded. Please run the embedding step first."
            else:
                try:
                    chain = build_rag_chain()
                    answer = chain.invoke({"question": prompt})
                except Exception as e:
                    answer = f"Error during processing:\n\n{str(e)}"

        st.markdown(answer)

        # Debug panel
        if st.session_state.show_debug:
            with st.expander("Debug Info", expanded=True):
                if st.session_state.last_decision:
                    d = st.session_state.last_decision
                    st.write("**Decision**")
                    st.write(f"• {d.explanation}")
                    st.write(f"• Codes: {d.target_codes or 'ALL'}")
                    st.write(f"• k = {d.k_value}")
                    st.write(f"• Comparison mode: {d.is_comparison}")

                if st.session_state.last_retrieval:
                    r = st.session_state.last_retrieval
                    st.write("**Retrieval**")
                    st.write(f"• {r['count']} chunks retrieved")
                    st.write(f"• From funds: {', '.join(r['funds']) or 'none'}")

    st.session_state.messages.append({"role": "assistant", "content": answer})