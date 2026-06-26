import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

FAISS_INDEX_PATH = "storage/faiss_index.bin"
CHUNKS_PATH = "storage/chunks.pkl"