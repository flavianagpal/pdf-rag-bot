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

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found at: {PDF_PATH}")

reader = PdfReader(PDF_PATH)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []
total_characters = 0

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if not text or not text.strip():
        continue

    total_characters += len(text)

    page_chunks = splitter.split_text(text)

    for chunk in page_chunks:
        chunks.append({
            "text": chunk,
            "page": page_number,
            "source": os.path.basename(PDF_PATH)
        })

if not chunks:
    raise ValueError(
        "No chunks were created from the PDF. The PDF may be empty, scanned, or non-text-based."
    )

print(f"Total characters extracted: {total_characters}")
print(f"Total chunks created: {len(chunks)}")

# ------------------
# Step 2: Embeddings
# ------------------
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Creating chunk embeddings...")

chunk_texts = [chunk["text"] for chunk in chunks]

chunk_embeddings = model.encode(
    chunk_texts,
    show_progress_bar=True
)

chunk_embeddings = np.array(chunk_embeddings).astype("float32")

if len(chunk_embeddings) == 0:
    raise ValueError("No embeddings were created from the chunks.")

faiss.normalize_L2(chunk_embeddings)

# ------------------
# Step 3: Build FAISS index
# ------------------
dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(chunk_embeddings)

print("FAISS index built successfully!")

# ------------------
# Step 4: Save index + chunks
# ------------------
os.makedirs("storage", exist_ok=True)

faiss.write_index(index, FAISS_INDEX_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
print(f"Saved chunks to {CHUNKS_PATH}")