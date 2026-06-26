from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ------------------
# Load PDF
# ------------------
reader = PdfReader("data/isl.pdf")

all_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        all_text += text + "\n"

# ------------------
# Chunking
# ------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(all_text)

print(f"Total Chunks: {len(chunks)}")

# ------------------
# Embeddings
# ------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = model.encode(chunks)

# ------------------
# FAISS Index
# ------------------
dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(chunk_embeddings))

print("FAISS index created!")

# ------------------
# Search
# ------------------
question = "What is logistic regression?"

question_embedding = model.encode([question])

k = 3

distances, indices = index.search(
    np.array(question_embedding),
    k
)

print("\nTop Retrieved Chunks:\n")

for i in indices[0]:
    print("=" * 80)
    print(chunks[i][:500])
    print()