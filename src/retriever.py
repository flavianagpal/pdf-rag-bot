import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PATH

# Load embedding model once
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def load_index_and_chunks():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks

def retrieve_chunks(question, index, chunks, k=3):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, k)

    retrieved = [chunks[i] for i in indices[0]]
    return retrieved