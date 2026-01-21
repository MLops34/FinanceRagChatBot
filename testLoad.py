# data.py
"""
Command-line script:
- Asks user to enter sheet name
- Processes only that sheet (row-by-row → LangChain Documents)
- Previews first 5 rows
- Saves to .json file (instead of .pkl)
"""

import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

EXCEL_FILE = r"E:\Work\FinanceRagChatBot\data\raw\db566-scheme-portfolio-details-december-2025.xlsx"

BASE_DIR = Path(__file__).parent
OUTPUT_FOLDER = BASE_DIR / "temp_pickles"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────
# SHEET → ROW DOCUMENTS
# ────────────────────────────────────────────────

def sheet_to_row_documents(
    excel_path: str | Path,
    sheet_name: str,
    header_row: int = 3,
    skip_empty_rows: bool = True
) -> list[Document]:
    excel_path = Path(excel_path)
    if not excel_path.is_file():
        print(f"Error: Excel file not found: {excel_path}")
        return []

    try:
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            header=header_row,
            dtype=str
        )
    except ValueError:
        print(f"Error: Sheet '{sheet_name}' not found in the file.")
        return []
    except Exception as e:
        print(f"Error reading sheet '{sheet_name}': {e}")
        return []

    documents = []

    for row_idx, row in df.iterrows():
        if skip_empty_rows and row.isna().all():
            continue

        content_parts = []
        for col_name, value in row.items():
            if pd.notna(value) and str(value).strip():
                content_parts.append(f"{col_name.strip()}: {str(value).strip()}")

        if not content_parts:
            continue

        content = "\n".join(content_parts)

        metadata = {
            "source": excel_path.name,
            "sheet": sheet_name,
            "row_index": row_idx,
            "row_number": row_idx + 1 + header_row,
            "is_index_sheet": sheet_name.strip().upper() == "INDEX",
            "processed_at": datetime.now().isoformat(),
            "portfolio_date": "2025-12-31"
        }

        row_dict = row.to_dict()
        for key in ["Sr No.", "Fund Code", "Name of Instrument", "Sector", "% of Net Assets", "ISIN"]:
            for col in row_dict:
                if key.lower() in col.lower():
                    val = row_dict[col]
                    if pd.notna(val):
                        clean_key = col.strip().replace(" ", "_").lower()
                        metadata[clean_key] = str(val).strip()
                    break

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    return documents

# ────────────────────────────────────────────────
# SAVE TO JSON
# ────────────────────────────────────────────────

def save_row_documents_to_json(documents: list[Document], sheet_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_sheet = sheet_name.replace(" ", "_").replace("/", "-")
    filename = f"row_docs_{safe_sheet}_{timestamp}.json"
    output_path = OUTPUT_FOLDER / filename

    # Convert Documents to JSON-serializable format
    serializable_data = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    return str(output_path)

# ────────────────────────────────────────────────
# MAIN COMMAND-LINE FLOW
# ────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("Motilal Oswal Sheet → Row Documents Tool (Command Line)")
    print("Saves output as JSON (human-readable format)")
    print("="*60)

    if not Path(EXCEL_FILE).is_file():
        print(f"Error: Excel file not found at:\n{EXCEL_FILE}")
        return

    # Get list of sheets
    try:
        xl = pd.ExcelFile(EXCEL_FILE)
        sheet_names = xl.sheet_names
    except Exception as e:
        print(f"Cannot read Excel file: {e}")
        return

    print("\nAvailable sheets:")
    for i, sheet in enumerate(sheet_names, 1):
        print(f"  {i}. {sheet}")

    # User input
    sheet_name = input("\nEnter the sheet name to process (exactly as shown above): ").strip()

    if sheet_name not in sheet_names:
        print(f"Error: Sheet '{sheet_name}' not found in the file.")
        return

    print(f"\nProcessing sheet: {sheet_name}")

    docs = sheet_to_row_documents(
        excel_path=EXCEL_FILE,
        sheet_name=sheet_name,
        header_row=3,           # change this if needed
        skip_empty_rows=True
    )

    if not docs:
        print("No valid rows found in this sheet.")
        return

    print(f"\nSuccess! Created {len(docs)} row-documents.")

    # Preview first 5 rows
    print("\nPreview of first 5 rows:")
    print("-"*60)
    for i, doc in enumerate(docs[:5], 1):
        print(f"\nRow {doc.metadata['row_number']}:")
        print("Metadata:", doc.metadata)
        print("Content (first 300 chars):")
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        print("-"*60)

    # Save as JSON
    print("\nSaving to JSON file...")
    saved_path = save_row_documents_to_json(docs, sheet_name)
    print(f"Saved successfully!")
    print(f"File location: {saved_path}")
    print(f"Total rows saved: {len(docs)}")


if __name__ == "__main__":
    main()