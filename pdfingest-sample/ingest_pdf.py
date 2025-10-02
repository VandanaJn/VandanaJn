import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from langchain_community.document_loaders import PyPDFLoader
import json

# --- CONFIGURATION ---
load_dotenv()
DOCUMENTS_FOLDER = Path(os.environ["DOCUMENTS_FOLDER"] )
INDEX_NAME = os.environ["PINECONE_INDEX"] 
DIMENSION = 1536
METRIC = "cosine"

SENTENCE_WINDOW_SIZE = 3 # Number of sentences to include on each side

# --- GLOBAL SETTINGS ---
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

def get_pdf_config() -> dict:
    """Load PDF config JSON as a dict."""
    path = Path(__file__).parent / "pdf_config.json"  
        
    if not path.exists():
        raise FileNotFoundError(f"PDF config not found at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- FUNCTIONS ---
def create_pinecone_index_if_not_exists(pc: Pinecone, index_name: str, dimension: int, metric: str):
    """Creates a Pinecone index if it doesn't already exist."""
    try:
        index_exists = any(idx['name'] == index_name for idx in pc.list_indexes())
        if not index_exists:
            print(f"Creating new Pinecone index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Index '{index_name}' created.")
        else:
            print(f"Index '{index_name}' already exists.")
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")
        raise
    
def load_and_store_to_vector_batch(index: VectorStoreIndex, rootfolder: Path, files: list):
    """
    Loads multiple files and inserts them into the index using sentence window retrieval.

    NOTE: This sample code **does not check for duplicates**. In production, you may want
    to track which files or nodes have already been ingested to avoid inserting duplicates.
    """
    documents = []
    for file_name in files:
        documents.extend(get_documents_from_pdf(rootfolder, file_name))

    # Use the SentenceWindowNodeParser
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=SENTENCE_WINDOW_SIZE,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    nodes = node_parser.get_nodes_from_documents(documents)

    # Insert nodes into the index
    index.insert_nodes(nodes)
    print(f"Inserted {len(nodes)} nodes across {len(files)} files.")

def get_documents_from_pdf(root_folder: Path, file_name: str):
    file_path = root_folder / file_name
    pdf_config = get_pdf_config()
    if file_name in pdf_config:
        current_pdf_config = pdf_config[file_name]
        skip_pages = current_pdf_config["skip_pages"]
        
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        filtered_documents = [
            doc for doc in documents
            if doc.metadata.get("page") + 1 not in skip_pages and doc.metadata.get("page")
        ]
        
        llama_docs = [
            Document(
                text=lc_doc.page_content,
                metadata={
                    "source": current_pdf_config["title"],
                    "author": current_pdf_config["author"],
                    "page_label": lc_doc.metadata.get("page_label", "na"),
                    "raw_page_no":lc_doc.metadata.get("page")
                }
            )
            for lc_doc in filtered_documents
        ]
        return llama_docs
    else:
        raise Exception(f"missing configuration for file {file_name}")

def main(files_to_ingest ):
    """Main ingestion pipeline function."""
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    
    create_pinecone_index_if_not_exists(pc, INDEX_NAME, DIMENSION, METRIC)

    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    index = VectorStoreIndex.from_vector_store(vector_store)

    load_and_store_to_vector_batch(index, DOCUMENTS_FOLDER, files_to_ingest)

    print("Ingestion complete.")
    
# --- RUN INGESTION ---
if __name__ == "__main__":
    main(["Sample.pdf"])