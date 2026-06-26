from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL_NAME
from retriever import load_index_and_chunks, retrieve_chunks

# ------------------
# Step 1: Configure Groq
# ------------------
client = Groq(api_key=GROQ_API_KEY)

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
    history_text = format_chat_history(chat_history)

    if not retrieved_chunks:
        context = "No relevant context found."
    else:
        context = "\n\n".join(
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
            for chunk in retrieved_chunks
        )

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
- Use only the document context to answer.
- Use the previous conversation when the user refers to earlier topics using words like "it", "that", "they", or "previous one".
- If the answer is not present in the document context, say: "I could not find that in the document."
- If page information is available, mention the page number in your answer.
- Keep the answer grounded in the document.
"""

    return prompt


def generate_answer(prompt):
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error while generating answer from Groq: {e}"


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

    retrieved_chunks = retrieve_chunks(question, index, chunks)

    if not retrieved_chunks:
        print("\nBot: I could not find relevant information in the document.")
        print("\n" + "=" * 100 + "\n")
        continue

    prompt = build_prompt(question, retrieved_chunks, chat_history)

    answer = generate_answer(prompt)

    print("\nBot:", answer)
    print("\n" + "=" * 100 + "\n")

    chat_history.append({
        "question": question,
        "answer": answer
    })