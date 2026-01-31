```markdown
# Financial RAG Chatbot: Motilal Oswal Portfolio Analyzer

A robust Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-aware answers to queries about Motilal Oswal mutual fund portfolios (as of December 2025). Built with LangChain, Streamlit, and FAISS for vector storage, this chatbot intelligently routes questions to specific funds or across multiple funds using a decision node. It supports ingestion from Excel, optional chunking, embedding, and a user-friendly web interface.

## Overview

This project processes Motilal Oswal's scheme portfolio details from Excel files, converts them into searchable documents, embeds them using sentence transformers, and stores them in a persistent FAISS index. The Streamlit-based chatbot leverages an LLM (e.g., DeepSeek R1 via OpenRouter) for natural-language queries, with smart routing for single-fund, multi-fund, or portfolio-wide analysis.

### Key Features
- **Intelligent Fund Routing**: Uses a decision node to detect and filter by fund codes (e.g., YO07, YO16) or names from queries.
- **Modular Pipeline**: Separate scripts for data loading, chunking, embedding, and querying.
- **Persistent Vector Store**: FAISS index for efficient semantic search with metadata filtering.
- **Debugging Tools**: Built-in retrieval inspector and UI debug mode for transparency.
- **Docker Support**: Easy containerization for deployment.
- **Secure LLM Integration**: Configurable via environment variables or Streamlit secrets.
- **Hallucination Guards**: Strict prompting to ensure answers are based solely on provided context.

## Project Structure

```
FinanceRagChatBot/
├── data/
│   └── raw/
│       └── db566-scheme-portfolio-details-december-2025.xlsx   # Source portfolio Excel file
├── db/
│   └── faiss_motilal/                                         # Persistent FAISS vector index
├── temp_pickles/                                              # Intermediate JSON files from ingestion
├── .gitattributes                                             # Git configuration
├── .gitignore                                                 # Git ignore patterns
├── app.py                                                     # Legacy or alternative Streamlit app
├── Chunk-02.py                                                # Document chunking script
├── DecisionNode.py                                            # Main Streamlit chatbot with decision node
├── Dockerfile                                                 # Docker build configuration
├── docker-compose.yml                                         # Docker Compose for multi-container setup
├── Embed-03.py                                                # Embedding and FAISS update script
├── Fetch-01.py                                                # Excel ingestion to documents (JSON)
├── inter-04.py                                                # Alternative chatbot version with decision node
├── README.md                                                  # This documentation
├── requirements.txt                                           # Python dependencies
├── Retrieval.py                                               # Utility for inspecting retrieved chunks
└── AI Financial Analyst Agent (RAG-BASED LLM).pptx            # Project presentation overview
```

## Requirements

- Python 3.10+
- Docker (for containerized deployment)
- OpenRouter API key (for LLM access; sign up at [openrouter.ai](https://openrouter.ai))

## Installation

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd FinanceRagChatBot
   ```

2. **Create Virtual Environment** (optional but recommended)  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up API Key**  
     OPENROUTER_API_KEY = "your-openrouter-api-key"
     ```
   - Alternatively, set as an environment variable:  
     ```bash
     export OPENROUTER_API_KEY="your-openrouter-api-key"
     ```

## Usage

### Data Ingestion Pipeline
Process the Excel data step-by-step to build the vector index.

1. **Ingest Excel to Documents** (`Fetch-01.py`):  
   Converts Excel sheets (by fund code) to JSON documents.  
   ```bash
   python Fetch-01.py
   ```
   - Enter fund code (e.g., `YO07`) when prompted. Outputs to `temp_pickles/<code>_docs.json`.

2. **Chunk Documents** (Optional, `Chunk-02.py`):  
   Splits documents for better retrieval granularity.  
   ```bash
   python Chunk-02.py
   ```
   - Select the input JSON file. Outputs chunked JSON to `temp_pickles/`.

3. **Embed and Update FAISS** (`Embed-03.py`):  
   Embeds documents and adds them to the persistent FAISS index.  
   ```bash
   python Embed-03.py
   ```
   - Select the JSON file to embed. The index is saved/updated in `db/faiss_motilal/`.

### Running the Chatbot
Launch the Streamlit app:  
```bash
streamlit run DecisionNode.py
```
- Open `http://localhost.` in your browser.
- Upload additional JSON documents via the sidebar if needed.
- Ask questions in the chat interface (e.g., "Top holdings of YO16" or "Compare YO16 vs YO46").

### Debugging Retrieval
Use `Retrieval.py` to inspect retrieved chunks:  
```bash
streamlit run Retrieval.py
```
- Enter a query and view matched documents with metadata.

## Docker Deployment

### Build and Run
1. **Build the Image**  
   ```bash
   docker build -t finance-rag-chatbot .
   ```

2. **Run the Container**  
   ```bash
   docker run -d \
     --name finance-rag-chatbot \
     -p 8501:8501 \
     -v $(pwd)/db/faiss_motilal:/app/db/faiss_motilal \
     -v $(pwd)/temp_pickles:/app/temp_pickles \
     -e OPENROUTER_API_KEY="your-openrouter-api-key" \
     finance-rag-chatbot
   ```
   - Access at `http://localhost:8501`.

### Using Docker Compose
```bash
docker-compose up -d
```
- Stops and restarts with `docker-compose down` / `up -d`.

## Architecture
- **Ingestion**: Excel → Pandas → LangChain Documents (JSON) → Optional Chunking.
- **Embedding**: Sentence Transformers → FAISS vector store with metadata (fund_code, fund_name).
- **Routing/Decision Node**: LLM-based extraction of fund codes/names → dynamic FAISS filtering.
- **Querying**: Streamlit chat → Retrieval (filtered) → Context formatting → LLM prompting → Response.
- **Debugging**: Retrieval inspector and UI toggles for transparency.

## Troubleshooting
- **FAISS Load Error**: Ensure `db/faiss_motilal/` exists and is not corrupted. Delete and re-embed if needed.
- **API Issues**: Verify OpenRouter key and network connectivity.
- **White Screen in Streamlit**: Check terminal for tracebacks; ensure session state is initialized early.
- **Retrieval Mismatch**: Add fund prefixes in `Fetch-01.py` for better semantic relevance.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more details, refer to the project presentation: [AI Financial Analyst Agent (RAG-BASED LLM).pptx](AI%20Financial%20Analyst%20Agent%20(RAG-BASED%20LLM).pptx).
```
