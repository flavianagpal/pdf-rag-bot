from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# ------------------
# Step 1: Load saved index + chunks
# ------------------
index = faiss.read_index("storage/faiss_index.bin")

with open("storage/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("FAISS index and chunks loaded successfully!")

# ------------------
# Step 2: Load embedding model
# ------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------
# Step 3: Ask a question
# ------------------
question = input("Enter your question: ")

question_embedding = model.encode([question])
question_embedding = np.array(question_embedding).astype("float32")

# ------------------
# Step 4: Search
# ------------------
k = 3
distances, indices = index.search(question_embedding, k)

print("\nTop retrieved chunks:\n")

for i in indices[0]:
    print("=" * 80)
    print(chunks[i][:1000])
    print()