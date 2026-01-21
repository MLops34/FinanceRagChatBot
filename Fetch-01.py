# 01_load_excel_to_docs.py
"""
Step 1 in the pipeline:
- Asks user for sheet name (fund code)
- Loads the corresponding sheet from Motilal Oswal portfolio Excel
- Creates one LangChain Document per holdings row
- Saves the documents list to a pickle file
"""

import os
import pickle
import pandas as pd
from langchain_core.documents import Document

# ──── CONFIGURATION ────
EXCEL_FILE = r"E:\Work\FinanceRagChatBot\data\raw\db566-scheme-portfolio-details-december-2025.xlsx"
OUTPUT_FOLDER = "temp_pickles"
START_ROW = 3                     # Usually holdings table starts around row 4 (0-indexed = 3)
MAX_HOLDINGS_ROWS = 120           # Safety limit - adjust if your funds have more holdings

# Make sure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_sheet_to_documents(sheet_name: str) -> list[Document]:
    """
    Load one sheet → create list of Document objects (one per row)
    Returns empty list if something fails
    """
    print(f"\nLoading sheet: {sheet_name}")

    if not os.path.exists(EXCEL_FILE):
        print(f"ERROR: Excel file not found at {EXCEL_FILE}")
        return []

    try:
        xl = pd.ExcelFile(EXCEL_FILE)
    except Exception as e:
        print(f"ERROR: Cannot open Excel file → {e}")
        return []

    if sheet_name not in xl.sheet_names:
        print(f"ERROR: Sheet '{sheet_name}' not found in the Excel file.")
        print("Available sheets (first 10):", ", ".join(xl.sheet_names[:10]), "...")
        return []

    # Read the sheet as strings
    df = pd.read_excel(
        xl,
        sheet_name=sheet_name,
        header=None,
        dtype=str
    ).fillna("")

    # Take holdings part (simple slicing for now)
    holdings_df = df.iloc[START_ROW : START_ROW + MAX_HOLDINGS_ROWS]

    documents = []

    print("Creating Documents from holdings rows...")

    for row_idx, row in holdings_df.iterrows():
        # Skip completely empty rows
        if all(str(val).strip() == "" for val in row):
            continue

        # Join non-empty cells with bullet-like separator
        row_content = " • ".join(
            str(val) for val in row if str(val).strip()
        ).strip()

        if not row_content:
            continue

        # Create LangChain Document
        doc = Document(
            page_content=row_content,
            metadata={
                "fund": sheet_name.upper(),
                "source": "Motilal Oswal Portfolio - Dec 2025",
                "row_number": int(row_idx),
                "sheet": sheet_name,
                "pipeline_step": "01_load"
            }
        )

        documents.append(doc)

    if documents:
        print(f"→ Successfully created {len(documents)} Documents from {sheet_name}")
    else:
        print("→ No valid holdings rows found in the selected range")

    return documents


def main():
    print("=== Step 1: Load Excel Sheet to Documents ===")
    print("Enter the sheet name (fund code) exactly as it appears in Excel")
    print("Examples: YO07, MO25, MA30, YO12, etc.\n")

    while True:
        sheet_name = input("Sheet name: ").strip().upper()
        
        if not sheet_name:
            print("Please enter a sheet name.\n")
            continue
            
        print(f"You entered: {sheet_name}")
        
        docs = load_sheet_to_documents(sheet_name)
        
        if not docs:
            print("Try again with a valid sheet name.\n")
            continue
        
        # Save to pickle
        output_file = os.path.join(OUTPUT_FOLDER, f"{sheet_name}_docs.pkl")
        
        with open(output_file, "wb") as f:
            pickle.dump(docs, f)
        
        print(f"\nSaved {len(docs)} documents to:")
        print(f"  → {output_file}")
        print("\nNext step: You can run Chunk.py  or directly go to 03_embed_and_update_store.py")
        break


if __name__ == "__main__":
    main()