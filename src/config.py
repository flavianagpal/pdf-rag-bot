import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

FAISS_INDEX_PATH = "storage/faiss_index.bin"
CHUNKS_PATH = "storage/chunks.pkl"