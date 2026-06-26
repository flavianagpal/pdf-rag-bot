from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example text
text = "Machine learning is a subset of artificial intelligence."

# Generate embedding
embedding = model.encode(text)

print(f"Embedding length: {len(embedding)}")

print("\nFirst 10 values:\n")
print(embedding[:10])