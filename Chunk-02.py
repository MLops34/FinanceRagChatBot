
"""
Step 2: Load .json from temp_pickles → optional chunking → save new .json
"""

import os 
import json 
from typing import List 
from datetime import datetime 
from langchain_core .documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 


INPUT_FOLDER ="temp_pickles"
OUTPUT_FOLDER ="temp_pickles"
CHUNK_SIZE =1000 
CHUNK_OVERLAP =150 
DO_SPLIT =True 

os .makedirs (OUTPUT_FOLDER ,exist_ok =True )


def _clean_value (value :str )->str :
    text =(value or "").strip ()
    return text if text else "Unknown"


def _is_noise_value (value :str )->bool :
    v =(value or "").strip ().lower ()
    if not v :
        return True 

    return all (ch .isdigit ()or ch in ".-+%"for ch in v )


def _extract_portfolio_fields (doc :Document )->dict :
    """
    Extract row-level portfolio fields from metadata/page_content.
    Expected row shape:
    Sr. No. • Name of Instrument • ISIN • Rating / Industry • Quantity •
    Market value (Rs. In lakhs) • % to Net Assets • Sector / Rating • ...
    """
    instrument =_clean_value (doc .metadata .get ("instrument",""))
    rating_industry =_clean_value (doc .metadata .get ("rating_industry",""))
    quantity =_clean_value (doc .metadata .get ("quantity",""))
    market_value =_clean_value (doc .metadata .get ("market_value",""))
    net_assets_pct =_clean_value (doc .metadata .get ("net_assets_pct",""))
    sector =_clean_value (doc .metadata .get ("sector",""))

    parts =[p .strip ()for p in doc .page_content .split ("•")]
    if len (parts )>=7 :
        if instrument =="Unknown":
            instrument =_clean_value (parts [1 ])
        if rating_industry =="Unknown":
            rating_industry =_clean_value (parts [3 ])
        if quantity =="Unknown":
            quantity =_clean_value (parts [4 ])
        if market_value =="Unknown":
            market_value =_clean_value (parts [5 ])
        if net_assets_pct =="Unknown":
            net_assets_pct =_clean_value (parts [6 ])
        if sector =="Unknown"and len (parts )>=8 :
            sector =_clean_value (parts [7 ])

    return {
    "name_of_instrument":instrument ,
    "rating_industry":rating_industry ,
    "quantity":quantity ,
    "market_value":market_value ,
    "net_assets_pct":net_assets_pct ,
    "sector":sector ,
    }


def _build_structured_content (doc :Document )->str :
    fund_name =_clean_value (doc .metadata .get ("fund_name",""))
    fund_code =_clean_value (doc .metadata .get ("fund_code",""))
    portfolio_fields =_extract_portfolio_fields (doc )

    return (
    f"Fund Name: {fund_name }\n"
    f"Fund Code: {fund_code }\n"
    f"Name of Instrument: {portfolio_fields ['name_of_instrument']}\n"
    f"Rating/Industry: {portfolio_fields ['rating_industry']}\n"
    f"Quantity: {portfolio_fields ['quantity']}\n"
    f"Market Value: {portfolio_fields ['market_value']}\n"
    f"% Net Assets: {portfolio_fields ['net_assets_pct']}\n"
    f"Sector: {portfolio_fields ['sector']}"
    )


def load_docs_from_json (file_path :str )->List [Document ]:
    if not os .path .exists (file_path ):
        print (f"ERROR: File not found → {file_path }")
        return []

    try :
        with open (file_path ,"r",encoding ="utf-8")as f :
            data =json .load (f )

        docs =[]
        for item in data :
            if not isinstance (item ,dict )or "page_content"not in item or "metadata"not in item :
                print ("Warning: Invalid item in JSON → skipping")
                continue 
            doc =Document (
            page_content =item ["page_content"],
            metadata =item ["metadata"]
            )
            docs .append (doc )

        print (f"Loaded {len (docs )} documents from {file_path }")
        return docs 

    except Exception as e :
        print (f"ERROR loading JSON: {type (e ).__name__ }: {e }")
        return []


def chunk_documents (docs :List [Document ])->List [Document ]:
    if not DO_SPLIT :
        print ("Splitting is OFF → returning original documents")
        return docs 

    splitter =RecursiveCharacterTextSplitter (
    chunk_size =CHUNK_SIZE ,
    chunk_overlap =CHUNK_OVERLAP ,
    length_function =len ,
    separators =["\n\n","\n"," | ",". "," ",""]
    )

    chunked_docs =[]
    for doc in docs :
        structured_content =_build_structured_content (doc )
        sub_chunks =splitter .split_text (structured_content )
        portfolio_fields =_extract_portfolio_fields (doc )

        for i ,chunk_text in enumerate (sub_chunks ):
            new_doc =Document (
            page_content =chunk_text ,
            metadata ={
            **doc .metadata ,
            "fund_name":_clean_value (doc .metadata .get ("fund_name","")),
            "fund_code":_clean_value (doc .metadata .get ("fund_code","")),
            "instrument":portfolio_fields ["name_of_instrument"],
            "sector":portfolio_fields ["sector"],
            "name_of_instrument":portfolio_fields ["name_of_instrument"],
            "rating_industry":portfolio_fields ["rating_industry"],
            "quantity":portfolio_fields ["quantity"],
            "market_value":portfolio_fields ["market_value"],
            "net_assets_pct":portfolio_fields ["net_assets_pct"],
            "chunk_id":i ,
            "original_chunk_start":i *(CHUNK_SIZE -CHUNK_OVERLAP ),
            "original_row":doc .metadata .get ("row_number"),
            "pipeline_step":"chunked"
            }
            )
            chunked_docs .append (new_doc )

    print (f"Split into {len (chunked_docs )} chunks (from {len (docs )} originals)")
    return chunked_docs 


def save_chunked_to_json (chunked_docs :List [Document ],original_filename :str )->str :
    timestamp =datetime .now ().strftime ("%Y%m%d_%H%M")
    base_name =original_filename .replace (".json","")
    if DO_SPLIT :
        output_name =f"{base_name }_chunked_{timestamp }.json"
    else :
        output_name =f"{base_name }_unchunked_{timestamp }.json"

    output_path =os .path .join (OUTPUT_FOLDER ,output_name )

    serializable_data =[
    {
    "page_content":doc .page_content ,
    "metadata":doc .metadata 
    }
    for doc in chunked_docs 
    ]

    with open (output_path ,"w",encoding ="utf-8")as f :
        json .dump (serializable_data ,f ,ensure_ascii =False ,indent =2 )

    return output_path 


def main ():
    print ("=== Step 2: Chunk Documents (JSON version) ===")
    print (f"Looking in folder: {INPUT_FOLDER }\n")


    json_files =[f for f in os .listdir (INPUT_FOLDER )if f .lower ().endswith (".json")]

    if not json_files :
        print (f"No .json files found in {INPUT_FOLDER }")
        print ("Run data.py first to create row_docs_*.json files.")
        return 

    print ("Available .json files:")
    for i ,fname in enumerate (sorted (json_files ),1 ):
        print (f"{i }. {fname }")

    while True :
        choice =input ("\nEnter number of file to process (or 'q' to quit): ").strip ()

        if choice .lower ()in ['q','quit','exit']:
            print ("Exiting.")
            return 

        try :
            idx =int (choice )-1 
            if 0 <=idx <len (json_files ):
                selected_file =json_files [idx ]
                break 
            else :
                print ("Invalid number.")
        except ValueError :
            print ("Enter a number or 'q'.")

    input_path =os .path .join (INPUT_FOLDER ,selected_file )
    docs =load_docs_from_json (input_path )

    if not docs :
        return 

    chunked_docs =chunk_documents (docs )


    output_path =save_chunked_to_json (chunked_docs ,selected_file )

    print (f"\nSaved {len (chunked_docs )} items to:")
    print (f"  → {output_path }")
    print ("\nNext step: Update 03_embed_and_update_store.py to load from .json and add to FAISS")


if __name__ =="__main__":
    main ()