import os

# HuggingFace Cache එක project එක ඇතුලටම හරවා යැවීම (Permission error එක විසඳීමට)
os.environ['HF_HOME'] = './.hf_cache'

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. දත්ත ලබා ගැනීම (Loading Documents)
loader = PyPDFDirectoryLoader("./data")
documents = loader.load()
print(f"Loaded {len(documents)} pages from the PDFs.")

# 2. දත්ත කොටස් වලට කැඩීම (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} text chunks.")

# 3. Embeddings සෑදීම
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vector Store එක සෑදීම (ChromaDB)
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
print("Vector Store successfully created and saved to './chroma_db'!")

# --- Evaluation (පරීක්ෂා කිරීම) ---
query = "What are the common artifacts in AI generated images?"
results = vector_store.similarity_search(query, k=3)

print("\n--- Evaluation Result ---")
for i, res in enumerate(results):
    print(f"Result {i+1}:\n {res.page_content}\n")