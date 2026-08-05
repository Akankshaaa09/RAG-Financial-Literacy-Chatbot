# qa_bot.py

import cohere
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ignore warnings
warnings.filterwarnings("ignore")

print("Loading models...")

# Set your API key from an environment variable
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable not set.")

# Initialize the Cohere client
client = cohere.Client(COHERE_API_KEY.strip())

# Load embeddings and vector store
# Make sure this model matches the one in build_index.py
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

print("Bot ready! 🚀")

# Short-term memory: only the last exchange, used solely to interpret
# follow-up phrasing — never as a source of facts for the answer itself.
last_question = None
last_answer = None

while True:
    question = input("\nYour question: ").strip()
    if question.lower() == "quit":
        break

    history_note = ""
    if last_question is not None:
        history_note = (
            f"Previous exchange (for context only, not a source of facts):\n"
            f"User asked: {last_question}\n"
            f"You answered: {last_answer}\n\n"
        )

    # Retrieval step: Find relevant document chunks from the FAISS index
    docs = index.similarity_search(question, k=4)
    # The documents retrieved from your FAISS index
    context = " ".join([doc.page_content for doc in docs])[:4000] # Limiting to 4000 chars to avoid token limit

    # Generation step: Use the Cohere API with the retrieved context
    try:
        preamble = (
            "You are a helpful financial literacy chatbot. Answer questions "
            "using only the provided context. If the context does not contain "
            "enough information to answer confidently, say so clearly rather "
            "than guessing or inferring beyond what is stated. Be concise and "
            "conversational. You may be given a previous exchange — use it only "
            "to understand what the current question is referring to (e.g. "
            "pronouns or follow-ups like 'what about X'), never as a source of "
            "facts for your answer."
        )
        message = f"{history_note}Context: {context}\nQuestion: {question}"

        response = client.chat(
            model="command-r-08-2024",
            message=message,
            preamble=preamble,
        )

        answer = response.text
        print(f"\nAnswer: {answer}")

        print("\nSources:")
        for i, doc in enumerate(docs, 1):
            snippet = doc.page_content.strip().replace("\n", " ")[:200]
            print(f"  [{i}] {snippet}...")

        last_question, last_answer = question, answer
    except Exception as e:
        print(f"Error calling Cohere API: {e}")
