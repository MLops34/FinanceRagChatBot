"""
Step 3: Load .json from temp_pickles → embed → create/update persistent FAISS index
"""

import os
import json
import time
from typing import List
from datetime import datetime
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core .documents import Document
from langchain_community .vectorstores import FAISS

PROJECT_ROOT =Path (__file__ ).resolve ().parent
INPUT_FOLDER =PROJECT_ROOT /"temp_pickles"
PERSIST_FAISS_DIR =PROJECT_ROOT /"db"/"faiss_motilal"
EMBEDDING_MODEL ="sentence-transformers/all-MiniLM-L6-v2"

os .makedirs (INPUT_FOLDER ,exist_ok =True )
os .makedirs (PERSIST_FAISS_DIR ,exist_ok =True )

def load_docs_from_json (file_path :str )->List [Document ]:
    """Load documents from a .json file"""
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

def load_or_create_faiss (embeddings ):
    """Load existing FAISS index and return tuple (vectorstore, is_new_index)."""
    index_path =os .path .join (PERSIST_FAISS_DIR ,"index.faiss")
    if os .path .exists (index_path ):
        print (f"Loading existing FAISS index from: {PERSIST_FAISS_DIR }")
        try :
            vectorstore =FAISS .load_local (
            PERSIST_FAISS_DIR ,
            embeddings ,
            allow_dangerous_deserialization =True
            )
            print ("FAISS status: Existing index found. New documents will be appended.")
            return vectorstore ,False
        except Exception as e :
            print (f"Failed to load existing FAISS index: {type (e ).__name__ }: {e }")
            print ("Will create a new one.")
    else :
        print ("FAISS status: First run (no index found). A new index will be created.")

    return None ,True

def main ():
    print ("=== Step 3: Embed & Update Persistent FAISS Index (JSON version) ===")
    print ("This step adds your JSON documents to a persistent FAISS index.\n")

    json_files =[f for f in os .listdir (INPUT_FOLDER )if f .lower ().endswith (".json")]
    if not json_files :
        print (f"No .json files found in {INPUT_FOLDER }")
        print ("Run data.py first to create row_docs_*.json files.")
        return

    print ("Available files to embed:")
    for i ,fname in enumerate (sorted (json_files ),1 ):
        path =os .path .join (INPUT_FOLDER ,fname )
        size_mb =os .path .getsize (path )/(1024 *1024 )
        ctime =datetime .fromtimestamp (os .path .getctime (path )).strftime ("%Y-%m-%d %H:%M")
        print (f"{i }. {fname }  ({size_mb :.1f} MB, {ctime })")

    while True :
        choice =input ("\nEnter number of file to add (or 'q' to quit): ").strip ()
        if choice .lower ()in ['q','quit','exit']:
            print ("Exiting Step 3.")
            return

        try :
            idx =int (choice )-1
            if 0 <=idx <len (json_files ):
                selected_file =json_files [idx ]
                break
            else :
                print ("Invalid number.")
        except ValueError :
            print ("Please enter a number or 'q'.")

    input_path =os .path .join (INPUT_FOLDER ,selected_file )
    docs =load_docs_from_json (input_path )

    if not docs :
        print ("No documents loaded → cannot continue.")
        return

    if not os .getenv ("HF_TOKEN"):
        print ("Note: HF_TOKEN is not set. Hugging Face may show an unauthenticated-request warning; this is expected.")
    embeddings =HuggingFaceEmbeddings (model_name =EMBEDDING_MODEL )

    vectorstore ,creating_new_index =load_or_create_faiss (embeddings )

    start_time =time .time ()
    print (f"Starting to embed and add {len (docs )} documents...")

    try :
        if creating_new_index :

            vectorstore =FAISS .from_documents (docs ,embeddings )
            vectorstore .save_local (PERSIST_FAISS_DIR )
            print (f"Created new FAISS index with {len (docs )} documents")
        else :

            vectorstore .add_documents (docs )
            vectorstore .save_local (PERSIST_FAISS_DIR )
            print (f"Added {len (docs )} new documents to existing index")

        elapsed =time .time ()-start_time
        print (f"Operation complete! Took {elapsed :.2f} seconds")
        print (f"Approximate total documents: {len (vectorstore .docstore ._dict )}")
        print (f"FAISS index saved/updated at: {PERSIST_FAISS_DIR }")
        print ("\nNext step: Create 04_query_rag_bot.py to start asking questions!")

    except Exception as e :
        print (f"FAISS operation failed: {type (e ).__name__ }: {str (e )}")
        print ("Try deleting the faiss_motilal folder and running again.")

if __name__ =="__main__":
    main ()
