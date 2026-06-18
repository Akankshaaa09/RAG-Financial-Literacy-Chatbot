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

while True:
    question = input("\nYour question: ").strip()
    if question.lower() == "quit":
        break

    # Retrieval step: Find relevant document chunks from the FAISS index
    docs = index.similarity_search(question, k=2)
    # The documents retrieved from your FAISS index
    context = " ".join([doc.page_content for doc in docs])[:4000] # Limiting to 4000 chars to avoid token limit

    # Generation step: Use the Cohere API with the retrieved context
    try:
        preamble = preamble = "You are a helpful financial literacy chatbot. Answer questions based primarily on the provided financial context. If the exact answer isn't stated but can be reasonably inferred from the data, say so clearly and provide your best estimate. Be concise and conversational — avoid saying you cannot answer unless you truly have no relevant information at all."
        message = f"Context: {context}\nQuestion: {question}"

        response = client.chat(
            model="command-r-08-2024",
            message=message,
            preamble=preamble,
        )

        answer = response.text
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"Error calling Cohere API: {e}")