"""
Step 1: Load Motilal Oswal portfolio Excel sheet -> create lean LangChain Documents
- Reads fund name from INDEX sheet
- Extracts key fields (instrument, ISIN, sector) into metadata
- Skips summary rows (NAV, AUM, etc.)
- Saves to pickle
"""

import os
import pickle
from pathlib import Path
import pandas as pd
from langchain_core .documents import Document

PROJECT_ROOT =Path (__file__ ).resolve ().parent
EXCEL_FILE =PROJECT_ROOT /"data"/"raw"/"db566-scheme-portfolio-details-december-2025.xlsx"
OUTPUT_FOLDER =PROJECT_ROOT /"temp_pickles"
START_ROW =3
MAX_ROWS =150
PORTFOLIO_DATE ="2025-12-31"

os .makedirs (OUTPUT_FOLDER ,exist_ok =True )

def get_fund_name (sheet_name :str )->str :
    """Read full fund name from INDEX sheet"""
    try :
        index_df =pd .read_excel (EXCEL_FILE ,sheet_name ="INDEX",header =None ,dtype =str )
        sheet_name_upper =sheet_name .upper ()
        for _ ,row in index_df .iterrows ():
            cells =[str (c ).strip ()for c in row if pd .notna (c )and str (c ).strip ()]
            if len (cells )>=3 and cells [2 ].upper ()==sheet_name_upper :
                return cells [1 ]
        return "Unknown Fund"
    except Exception as e :
        print (f"Warning: Could not read INDEX sheet -> {e }")
        return "Unknown Fund"

def load_sheet_to_documents (sheet_name :str )->list [Document ]:
    print (f"\nProcessing sheet: {sheet_name }")

    if not os .path .exists (EXCEL_FILE ):
        print (f"ERROR: Excel file not found -> {EXCEL_FILE }")
        return []

    try :
        xl =pd .ExcelFile (EXCEL_FILE )
    except Exception as e :
        print (f"ERROR: Cannot open Excel -> {e }")
        return []

    if sheet_name not in xl .sheet_names :
        print (f"ERROR: Sheet '{sheet_name }' not found.")
        print ("Available (first 10):",", ".join (xl .sheet_names [:10 ]),"...")
        return []

    df =pd .read_excel (xl ,sheet_name =sheet_name ,header =None ,dtype =str ).fillna ("")

    holdings_df =df .iloc [START_ROW :START_ROW +MAX_ROWS ]

    documents =[]
    fund_name =get_fund_name (sheet_name )
    print (f"Fund name from INDEX: {fund_name }")

    summary_keywords =[
    "nav","aum","aaum","monthly aum","latest aum",
    "portfolio turnover ratio","benchmark name","risk-o-meter",
    "dividend history","record date","cum dividend","ex dividend"
    ]

    for row_idx ,row in holdings_df .iterrows ():

        cells =[str (val ).strip ()for val in row if str (val ).strip ()]
        if not cells :
            continue

        prefix =f"[Fund: {fund_name } | Code: {sheet_name .upper ()}] "
        row_content =prefix +" • ".join (cells )

        lower_content =row_content .lower ()
        if any (kw in lower_content for kw in summary_keywords ):
            continue

        instrument =""
        isin =""
        sector =""

        if len (cells )>=8 :

            instrument =cells [1 ]if len (cells )>1 else ""

            for i ,cell in enumerate (cells ):
                if len (cell )==12 and cell .startswith ("INE"):
                    isin =cell
                    break

            sector =cells [-1 ]if cells [-1 ]and cells [-1 ]not in ["Percent",""]else cells [-2 ]if len (cells )>2 else ""

        metadata ={
        "source":os .path .basename (EXCEL_FILE ),
        "fund_code":sheet_name .upper (),
        "fund_name":fund_name ,
        "portfolio_date":PORTFOLIO_DATE ,
        "row_number":int (row_idx )+1 ,
        "isin":isin .strip ()if isin else "",
        "instrument":instrument .strip ()if instrument else "",
        "sector":sector .strip ()if sector else "",
        }

        metadata ["fund_name_normalized"]=(
        fund_name .lower ()
        .replace ("motilal oswal ","")
        .replace ("fund","")
        .replace ("scheme","")
        .replace ("direct","")
        .replace ("growth","")
        .replace ("regular","")
        .strip ()
        )

        doc =Document (page_content =row_content ,metadata =metadata )
        documents .append (doc )

    if documents :
        print (f"→ Created {len (documents )} documents from {sheet_name }")
    else :
        print ("→ No valid rows found")

    return documents

import json

def main ():
    print ("=== Load Motilal Oswal Portfolio to Documents ===")
    print ("Enter fund code (sheet name) e.g. YO07, YO16, YO46\n")

    while True :
        sheet_name =input ("Fund code: ").strip ().upper ()
        if not sheet_name :
            print ("Please enter a code.\n")
            continue

        docs =load_sheet_to_documents (sheet_name )
        if not docs :
            print ("Try again.\n")
            continue

        output_file =OUTPUT_FOLDER /f"{sheet_name }_docs.json"

        docs_for_json =[
        {
        "page_content":doc .page_content ,
        "metadata":doc .metadata
        }
        for doc in docs
        ]

        with open (output_file ,"w",encoding ="utf-8")as f :
            json .dump (docs_for_json ,f ,indent =2 ,ensure_ascii =False )

        print (f"\nSaved {len (docs )} documents as JSON -> {output_file }")
        print ("Next: chunk.py or embed step")
        break

if __name__ =="__main__":
    main ()
