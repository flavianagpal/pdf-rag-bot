from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from retriever import load_index_and_chunks, retrieve_chunks

# ------------------
# Step 1: Configure Gemini
# ------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------
# Step 2: Load FAISS + chunks
# ------------------
index, chunks = load_index_and_chunks()
print("FAISS index and chunks loaded successfully!")

# ------------------
# Step 3: Chat history
# ------------------
chat_history = []

def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."

    history_text = ""
    for turn in chat_history:
        history_text += f"User: {turn['question']}\n"
        history_text += f"Assistant: {turn['answer']}\n\n"

    return history_text.strip()

def build_prompt(question, retrieved_chunks, chat_history):
    context = "\n\n".join(retrieved_chunks)
    history_text = format_chat_history(chat_history)

    prompt = f"""
You are a helpful conversational assistant answering questions about a PDF document.

Previous conversation:
{history_text}

Document context:
{context}

Current user question:
{question}

Instructions:
- Answer clearly and simply.
- Use the document context to answer.
- Use the previous conversation when the user refers to earlier topics using words like "it", "that", "they", or "previous one".
- If the answer is not in the document context, say: "I could not find that in the document."
- Keep the answer grounded in the document.
"""
    return prompt

def generate_answer(prompt):
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt
    )
    return response.text

# ------------------
# Step 4: Chat loop
# ------------------
print("\nPDF RAG Bot is ready!")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    retrieved_chunks = retrieve_chunks(question, index, chunks, k=3)
    prompt = build_prompt(question, retrieved_chunks, chat_history)
    answer = generate_answer(prompt)

    print("\nBot:", answer)
    print("\n" + "="*100 + "\n")

    chat_history.append({
        "question": question,
        "answer": answer
    })