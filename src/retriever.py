import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PATH

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

SIMILARITY_THRESHOLD = 0.6


def load_index_and_chunks():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def retrieve_chunks(question, index, chunks, k=5):
    question_embedding = embedding_model.encode([question])
    question_embedding = np.array(question_embedding).astype("float32")

    faiss.normalize_L2(question_embedding)

    similarities, indices = index.search(question_embedding, k)

    retrieved = []

    for similarity, i in zip(similarities[0], indices[0]):
        if (
            0 <= i < len(chunks)
            and similarity >= SIMILARITY_THRESHOLD
        ):
            retrieved.append(chunks[i])

    return retrieved