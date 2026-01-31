## FinanceRagChatBot – Ex- Data Source (Motilal Oswal Portfolio ) RAG

This project is a small end‑to‑end Retrieval‑Augmented Generation (RAG) system for Motilal Oswal mutual fund portfolios (Dec‑2025 data).  
It:

- **Loads holdings from the Excel file** into structured `Document` objects.
- **Optionally chunks** those documents.
- **Embeds & stores them in a FAISS vector index**.
- **Exposes a Streamlit chatbot UI** that queries the portfolio data using an LLM (DeepSeek R1 via OpenRouter).

> Note: The active Streamlit app is `newapp.py`. The root `app.py` can be ignored.

---

## Project Structure

- **`data/raw/db566-scheme-portfolio-details-december-2025.xlsx`**: Source Motilal Oswal portfolio Excel.
- **`Testload.py`**: **Step 1** – Load one Excel sheet → create LangChain `Document` list → save to `temp_pickles/*_docs.pkl`.
- **`Chunk.py`**: **Step 2 (optional)** – Chunk or pass‑through documents → save to `temp_pickles/*_ready.pkl` (or `*_chunks.pkl`).
- **`Embed.py`**: **Step 3** – Embed selected pickle file and update persistent FAISS index in `db/faiss_motilal`.
- **`db/faiss_motilal/`**: Persistent FAISS index used by the chatbot.
- **`temp_pickles/`**: Intermediate `.pkl` files produced by the pipeline.
- **`Inter.py`**: Streamlit RAG chatbot over the FAISS index (DeepSeek R1 via OpenRouter). You can Use other Open Source Model As well.

-

## 1. Environment & Dependencies

Create and activate a virtual environment (optional but recommended), then install requirements.  
Typical packages used:

- `streamlit`
- `pandas`
- `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-openai`
- `sentence-transformers`
- `faiss-cpu` (or GPU variant if desired)

Example (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install streamlit pandas langchain-core langchain-community langchain-huggingface langchain-openai sentence-transformers faiss-cpu
```

---

## 2. Data Pipeline Overview

The pipeline works in **three main steps** (plus one helper script):

- **Helper – Inspect a fund:**  
  - Script: `DocLoad.py`  
  - Purpose: Quickly load one scheme (sheet) from the Excel, preview rows, and auto‑save holdings to `YO07_portfolio_holdings.csv`, etc.

- **Step 1 – Excel → Documents:**  
  - Script: `Fetch.py`  
  - Flow:
    - Asks for a **sheet name** (fund code), e.g. `YO07`, `YO16`, etc.
    - Loads that sheet from `data/raw/db566-scheme-portfolio-details-december-2025.xlsx`.
    - Reads holdings rows (from `START_ROW` for up to `MAX_HOLDINGS_ROWS`).
    - Creates one LangChain `Document` per row with metadata:
      - `fund`, `sheet`, `row_number`, `source`, `pipeline_step`.
    - Saves to `temp_pickles/{SHEET}_docs.pkl`.

- **Step 2 – (Optional) Chunking:**  
  - Script: `Chunk.py`  
  - Flow:
    - Lists available `*_docs.pkl` files in `temp_pickles/`.
    - Lets you select one file.
    - If `DO_SPLIT = False` (default) → passes documents through unchanged and saves `{SHEET}_ready.pkl`.
    - If `DO_SPLIT = True` → will chunk long texts (template code is already in the file) and save as `{SHEET}_chunks.pkl`.

- **Step 3 – Embed & Update FAISS:**  
  - Script: `Embed.py`  
  - Flow:
    - Lists all `.pkl` files in `temp_pickles/` (`*_docs.pkl`, `*_ready.pkl`, etc.).
    - You choose one file to embed.
    - Loads `HuggingFaceEmbeddings` with model `sentence-transformers/all-MiniLM-L6-v2`.
    - **Loads or creates** a FAISS index under `db/faiss_motilal/`.
    - Embeds documents and either:
      - Creates a new FAISS index, or
      - Adds to existing index.
    - Saves the updated index to `db/faiss_motilal/` and prints approximate document count.

After Step 3, the Streamlit app (`newapp.py`) can use this FAISS index to answer questions.

---

## 3. Running the Data Pipeline

All commands below assume your working directory is the project root (`E:\Work\FinanceRagChatBot`).

### 3.1 Inspect a Scheme (Optional)

```bash
python DocLoad.py
```

- Enter a sheet name, e.g. `YO07`.  
- The script will:
  - Print some preview rows.
  - Save `YO07_portfolio_holdings.csv` in the current directory.

### 3.2 Step 1 – Excel → Documents

```bash
python Fetch.py
```

- When prompted, enter a sheet name (fund code) exactly as in the Excel file (e.g. `YO07`).
- Output: `temp_pickles/YO07_docs.pkl` (list of LangChain `Document` objects).

### 3.3 Step 2 – (Optional) Chunking

```bash
python Chunk.py
```

- The script will list available `*_docs.pkl` files.
- Type the number corresponding to the desired file.
- With `DO_SPLIT = False` (default), output will be, for example: `temp_pickles/YO07_ready.pkl`.

### 3.4 Step 3 – Embed & Update FAISS

```bash
python Embed.py
```

- Pick which `.pkl` file to embed (e.g. `YO07_ready.pkl` or `YO07_docs.pkl`).
- The script:
  - Embeds the documents with `sentence-transformers/all-MiniLM-L6-v2`.
  - Writes/updates the FAISS index in `db/faiss_motilal/`.

Once this is done (for as many funds as you want), you’re ready to run the chatbot.

---

## 4. Running the Streamlit RAG Chatbot (`newapp.py`)

### 4.1 Configure OpenRouter / DeepSeek R1

The chatbot uses `ChatOpenAI` with:

- **Base URL**: `https://openrouter.ai/api/v1`
- **Model**: `deepseek/deepseek-r1`

You need an **OpenRouter API key** set as `OPENROUTER_API_KEY`. You can provide it in either of these ways:

- **Environment variable**:

```bash
$env:OPENROUTER_API_KEY = "sk-..."
```

- **Streamlit secrets** (`.streamlit/secrets.toml`):

```toml
OPENROUTER_API_KEY = "sk-..."
```

`newapp.py` reads from `st.secrets.get("OPENROUTER_API_KEY", "your-key-here")` and from `os.getenv("OPENROUTER_API_KEY")`.

### 4.2 Start the App

From the project root:

```bash
streamlit run newapp.py
```

Then open the URL that Streamlit prints (typically `http://localhost:8501`).

### 4.3 Using the App

- The app will try to **load an existing FAISS index** from `db/faiss_motilal/`.
  - If found: it shows the number of documents in the sidebar.
  - If not found: you’ll see a warning asking you to upload a `.pkl` file.
- In the **sidebar**:
  - You can upload additional `.pkl` files containing `list[Document]`.
  - Uploaded documents are added to the FAISS index and persisted.
- In the **main chat**:
  - Type natural‑language questions about holdings (e.g. “What are the top holdings of YO07?”).
  - The app retrieves the most relevant documents (k=6) and uses DeepSeek R1 to formulate an answer based only on that context.

---

## 5. Configuration & Paths

Key hard‑coded paths (edit as needed):

- **Excel source**: `data/raw/db566-scheme-portfolio-details-december-2025.xlsx` (see `Fetch.py` and `DocLoad.py`).
- **Intermediate pickles**: `temp_pickles/` (see `Fetch.py`, `Chunk.py`, `Embed.py`).
- **FAISS index directory**: `db/faiss_motilal/` (see `Embed.py` and `newapp.py`).

If you move the project or rename folders, update these paths accordingly in the scripts.

---

## 6. Typical End‑to‑End Workflow

1. **Prepare environment** – Install dependencies and ensure the Excel file is in `data/raw/`.
2. **Generate documents** – Run `Fetch.py` for each fund code you want to include.
3. **(Optional) Chunk** – Run `Chunk.py` to produce `*_ready.pkl` if you want pre‑processed/chunked docs.
4. **Embed into FAISS** – Run `Embed.py` and add all desired `.pkl` files to the FAISS index in `db/faiss_motilal/`.
5. **Configure API key** – Set `OPENROUTER_API_KEY` via env var or `.streamlit/secrets.toml`.
6. **Run chatbot** – `streamlit run newapp.py` and start asking questions about the portfolios.

This README intentionally **ignores `app.py`** and documents the pipeline + chatbot based on the current scripts (`DocLoad.py`, `Fetch.py`, `Chunk.py`, `Embed.py`, `newapp.py`).


