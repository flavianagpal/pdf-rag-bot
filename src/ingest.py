from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    CHUNKS_PATH
)

# ------------------
# Step 1: Load PDF
# ------------------
PDF_PATH = "data/isl.pdf"

reader = PdfReader(PDF_PATH)

all_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        all_text += text + "\n"

print(f"Total characters extracted: {len(all_text)}")

# ------------------
# Step 2: Chunking
# ------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(all_text)

print(f"Total chunks created: {len(chunks)}")

# ------------------
# Step 3: Embeddings
# ------------------
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Creating chunk embeddings...")
chunk_embeddings = model.encode(chunks, show_progress_bar=True)

chunk_embeddings = np.array(chunk_embeddings).astype("float32")

# ------------------
# Step 4: Build FAISS index
# ------------------
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print("FAISS index built successfully!")

# ------------------
# Step 5: Save index + chunks
# ------------------
os.makedirs("storage", exist_ok=True)

faiss.write_index(index, FAISS_INDEX_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
print(f"Saved chunks to {CHUNKS_PATH}")