# build_index.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the extracted text
with open("financial_literacy_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks with overlap to keep context
# Using RecursiveCharacterTextSplitter is better for structured documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
texts = text_splitter.split_text(text)

# Create embeddings using a more powerful HuggingFace model (local, no API cost)
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Build the FAISS vector store index
index = FAISS.from_texts(texts, embeddings)

# Save the index to disk for later use
index.save_local("faiss_index")

print(f"Created and saved index with {len(texts)} chunks. ✅")