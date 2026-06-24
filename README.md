PDF RAG Bot with Conversational Memory

A conversational PDF RAG (Retrieval-Augmented Generation) chatbot built in Python.
This project takes a PDF document, chunks it with overlap, converts chunks into embeddings, stores them in a FAISS vector index, and uses Gemini to answer user questions grounded in the document.

It also supports multi-turn conversation, so follow-up questions like “How is it different?” or “Explain that in simple words” work using chat history + retrieval.

Features
PDF text extraction using pypdf
Chunking with overlap using RecursiveCharacterTextSplitter
Semantic embeddings using sentence-transformers
Vector search with FAISS
Grounded answer generation using Gemini
Conversational memory for follow-up questions
Clean modular structure with separate ingestion, retrieval, and chat logic
Project Architecture
1. Ingestion Pipeline

The ingestion step processes the PDF and creates the knowledge base.

PDF
→ extract text
→ chunk with overlap
→ generate embeddings
→ store embeddings in FAISS
→ save chunks separately
2. Retrieval Pipeline

At query time, the user question is embedded and compared against stored chunk embeddings.

User Question
→ question embedding
→ FAISS similarity search
→ top relevant chunks
3. Generation Pipeline

The retrieved chunks are passed to Gemini along with the user question.

Retrieved Chunks + User Question
→ Gemini
→ Final grounded answer
4. Conversational Memory

The chatbot also keeps track of earlier questions and answers in the current session, so follow-up questions can be interpreted correctly.

Example:

Q1: What is logistic regression?
Q2: How is it different from linear regression?
Q3: Explain that in simple words.

This requires both:

Retrieval memory → the PDF knowledge stored in FAISS
Conversation memory → the current session’s previous turns