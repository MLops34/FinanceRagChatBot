# 01_load_excel_to_docs.py  # filename and purpose
"""
Step 1: Load Motilal Oswal portfolio Excel sheet -> create lean LangChain Documents
- Reads fund name from INDEX sheet
- Extracts key fields (instrument, ISIN, sector) into metadata
- Skips summary rows (NAV, AUM, etc.)
- Saves to pickle
"""  # module-level description

import os  
import pickle  
import pandas as pd  # data handling and Excel I/O
from langchain_core.documents import Document  # LangChain Document type

# ──── CONFIG ────
EXCEL_FILE =   # path to source Excel
OUTPUT_FOLDER = "temp_pickles"  # output folder for generated artifacts
START_ROW = 3                     # 0-based -> usually row 4 in Excel (headers)  # starting row index
MAX_ROWS = 150                    # increased slightly for safety  # max rows to read
PORTFOLIO_DATE = "2025-12-31"  # portfolio effective date

os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # ensure output folder exists


def get_fund_name(sheet_name: str) -> str:  # read full fund name from INDEX sheet
    """Read full fund name from INDEX sheet"""
    try:
        index_df = pd.read_excel(EXCEL_FILE, sheet_name="INDEX", header=None, dtype=str)  # read INDEX sheet
        sheet_name_upper = sheet_name.upper()  # normalize lookup key
        for _, row in index_df.iterrows():  # iterate rows to find matching code
            cells = [str(c).strip() for c in row if pd.notna(c) and str(c).strip()]  # extract non-empty cells
            if len(cells) >= 3 and cells[2].upper() == sheet_name_upper:  # match fund code column
                return cells[1]  # fund name is usually column B
        return "Unknown Fund"  # fallback if not found
    except Exception as e:
        print(f"Warning: Could not read INDEX sheet -> {e}")  # log read errors
        return "Unknown Fund"  # fallback on error


def load_sheet_to_documents(sheet_name: str) -> list[Document]:  # convert a sheet to LangChain Documents
    print(f"\nProcessing sheet: {sheet_name}")  # progress message

    if not os.path.exists(EXCEL_FILE):  # check Excel existence
        print(f"ERROR: Excel file not found -> {EXCEL_FILE}")
        return []  # abort if missing

    try:
        xl = pd.ExcelFile(EXCEL_FILE)  # open workbook for sheet listing
    except Exception as e:
        print(f"ERROR: Cannot open Excel -> {e}")  # log and abort on error
        return []

    if sheet_name not in xl.sheet_names:  # verify sheet exists
        print(f"ERROR: Sheet '{sheet_name}' not found.")
        print("Available (first 10):", ", ".join(xl.sheet_names[:10]), "...")  # helper listing
        return []

    # Read as strings, no header
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None, dtype=str).fillna("")  # read sheet into DataFrame

    holdings_df = df.iloc[START_ROW : START_ROW + MAX_ROWS]  # slice rows within configured window

    documents = []  # container for resulting Documents
    fund_name = get_fund_name(sheet_name)  # lookup full fund name
    print(f"Fund name from INDEX: {fund_name}")  # debug output

    summary_keywords = [  # keywords to identify summary rows to skip
        "nav", "aum", "aaum", "monthly aum", "latest aum",
        "portfolio turnover ratio", "benchmark name", "risk-o-meter",
        "dividend history", "record date", "cum dividend", "ex dividend"
    ]

    for row_idx, row in holdings_df.iterrows():  # iterate each candidate row
        # Build content: join non-empty cells
        cells = [str(val).strip() for val in row if str(val).strip()]  # extract non-empty cell texts
        if not cells:
            continue  # skip empty rows

        # row_content = " • ".join(cells)  # human-readable joined content
        prefix = f"[Fund: {fund_name} | Code: {sheet_name.upper()}] "
        row_content = prefix + " • ".join(cells)
        
        # Skip summary rows
        lower_content = row_content.lower()  # lowercase for keyword checks
        if any(kw in lower_content for kw in summary_keywords):
            continue  # skip rows that look like summaries

        # Extract structured fields (adjust indices based on your columns)
        instrument = ""  # placeholder for instrument name
        isin = ""  # placeholder for ISIN
        sector = ""  # placeholder for sector

        if len(cells) >= 8:  # heuristic when expected columns present
            # Typical order: Sr.No, Instrument, ..., ISIN (often col 3-4), Rating/Industry, ..., Sector (near end)
            instrument = cells[1] if len(cells) > 1 else ""  # instrument usually at index 1
            # ISIN usually around position 3-4
            for i, cell in enumerate(cells):
                if len(cell) == 12 and cell.startswith("INE"):  # rough ISIN check
                    isin = cell  # capture ISIN when pattern matches
                    break
            # Sector often last or second last
            sector = cells[-1] if cells[-1] and cells[-1] not in ["Percent", ""] else cells[-2] if len(cells) > 2 else ""  # heuristic for sector

        # Create lean metadata
                # Create lean metadata
        metadata = {
            "source": os.path.basename(EXCEL_FILE),
            "fund_code": sheet_name.upper(),
            "fund_name": fund_name,
            "portfolio_date": PORTFOLIO_DATE,
            "row_number": int(row_idx) + 1,
            "isin": isin.strip() if isin else "",
            "instrument": instrument.strip() if instrument else "",
            "sector": sector.strip() if sector else "",
        }

        # ← Add this block here
        metadata["fund_name_normalized"] = (
            fund_name.lower()
            .replace("motilal oswal ", "")
            .replace("fund", "")
            .replace("scheme", "")
            .replace("direct", "")
            .replace("growth", "")
            .replace("regular", "")
            .strip()
        )

        doc = Document(page_content=row_content, metadata=metadata)
        documents.append(doc)

    if documents:
        print(f"→ Created {len(documents)} documents from {sheet_name}")  # success message
    else:
        print("→ No valid rows found")  # no documents found

    return documents  # return result list

import json  # JSON persistence for output
import os  # os re-import (harmless duplicate) for clarity

def main():  # interactive runner
    print("=== Load Motilal Oswal Portfolio to Documents ===")  # header
    print("Enter fund code (sheet name) e.g. YO07, YO16, YO46\n")  # prompt example

    while True:  # loop until valid input processed
        sheet_name = input("Fund code: ").strip().upper()  # get user input and normalize
        if not sheet_name:
            print("Please enter a code.\n")  # re-prompt on empty
            continue

        docs = load_sheet_to_documents(sheet_name)  # load documents for sheet
        if not docs:
            print("Try again.\n")  # try another code if nothing produced
            continue

        # Define output filename with .json extension
        output_file = os.path.join(OUTPUT_FOLDER, f"{sheet_name}_docs.json")  # target JSON file

        # Convert LangChain Documents to plain dicts (JSON-serializable)
        docs_for_json = [
            {
                "page_content": doc.page_content,  # document text
                "metadata": doc.metadata  # document metadata
            }
            for doc in docs
        ]

        # Save as JSON (text mode "w")
        with open(output_file, "w", encoding="utf-8") as f:  # write JSON to file
            json.dump(docs_for_json, f, indent=2, ensure_ascii=False)  # pretty-print JSON

        print(f"\nSaved {len(docs)} documents as JSON -> {output_file}")  # confirm save
        print("Next: chunk.py or embed step")  # hint for next step
        break  # exit loop after successful save

if __name__ == "__main__":
    main()  # run interactive main