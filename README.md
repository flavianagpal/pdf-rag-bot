# 📄 Conversational PDF RAG Bot

A conversational **Retrieval-Augmented Generation (RAG)** chatbot built using **Python**, **FAISS**, **Sentence Transformers**, and **Google Gemini**. The chatbot answers questions grounded in the contents of a PDF by retrieving the most relevant document chunks before generating a response.

---

## 🚀 Features

* 📖 Extracts text from PDF documents
* ✂️ Splits text into overlapping chunks for better context preservation
* 🧠 Generates semantic embeddings using Sentence Transformers
* 🔍 Performs fast semantic search with FAISS
* 🤖 Generates grounded responses using Google Gemini
* 💬 Supports conversational follow-up questions using chat history
* 📦 Modular project structure for ingestion, retrieval, and chat

---

## 🏗️ Architecture

```
                PDF Document
                      │
                      ▼
             Text Extraction (PyPDF)
                      │
                      ▼
      Recursive Character Chunking
      (Chunk Size = 1000, Overlap = 200)
                      │
                      ▼
     Sentence Transformer Embeddings
      (all-MiniLM-L6-v2)
                      │
                      ▼
          FAISS Vector Index
                      │
        ─────────────────────────
                      │
               User Question
                      │
                      ▼
          Question Embedding
                      │
                      ▼
            Semantic Search
              (Top-k Chunks)
                      │
                      ▼
     Retrieved Context + Chat History
                      │
                      ▼
             Google Gemini API
                      │
                      ▼
               Final Response
```

---

## 📂 Project Structure

```
PDF_RAG_Bot/
│
├── data/
│   └── isl.pdf
│
├── storage/
│   ├── faiss_index.bin
│   └── chunks.pkl
│
├── src/
│   ├── chatbot.py
│   ├── config.py
│   ├── ingest.py
│   └── retriever.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Tech Stack

* Python
* PyPDF
* LangChain Text Splitters
* Sentence Transformers
* FAISS
* Google Gemini API
* NumPy
* python-dotenv

---

## 🧩 How It Works

### 1. Ingestion

The PDF is processed once to create a searchable knowledge base.

Steps performed:

* Extract text from the PDF
* Split text into overlapping chunks
* Generate embeddings for every chunk
* Store embeddings inside a FAISS vector index
* Save both the FAISS index and original chunks

---

### 2. Retrieval

When a user asks a question:

* The question is converted into an embedding.
* FAISS performs semantic similarity search.
* The most relevant chunks are retrieved.

---

### 3. Generation

The retrieved chunks, along with the user's question and conversation history, are sent to Google Gemini.

Gemini generates a grounded response using only the retrieved document context.

---

## 💬 Conversational Memory

Unlike a basic RAG system that answers one question at a time, this chatbot maintains conversation history during the session.

Example:

```
User:
What is logistic regression?

Bot:
...

User:
How is it different from linear regression?

Bot:
...

User:
Explain that in simple words.
```

The chatbot understands references such as **"it"**, **"that"**, and **"previous one"** by combining conversation history with retrieved document context.

---

## ▶️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd PDF_RAG_Bot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Windows

```bash
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## ▶️ Running the Project

### Build the Knowledge Base

```bash
python src/ingest.py
```

This will:

* Read the PDF
* Create chunks
* Generate embeddings
* Build the FAISS index
* Save the knowledge base

---

### Start the Chatbot

```bash
python src/chatbot.py
```

Example:

```
You:
What is logistic regression?

Bot:
Logistic regression is a statistical learning method used for binary classification...
```

---

## 📚 What I Learned

This project helped me understand the complete Retrieval-Augmented Generation (RAG) pipeline, including:

* PDF preprocessing
* Text chunking with overlap
* Embedding generation
* Vector databases
* Semantic similarity search
* Prompt engineering
* Grounded LLM responses
* Conversational memory

---

## 📜 License

This project is intended for educational and learning purposes.
