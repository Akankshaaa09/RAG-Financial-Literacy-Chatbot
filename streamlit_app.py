import streamlit as st
import cohere
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
from dotenv import load_dotenv

# --- SETUP AND INITIALIZATION (Same as qa_bot.py) ---
# Load environment variables from .env file
load_dotenv()

# Ignore deprecation warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Financial Literacy RAG Chatbot",
    page_icon="🤖"
)
st.title("Financial Literacy RAG Chatbot 🤖")
st.markdown("I am a chatbot that provides answers about financial literacy, based on the **World Bank's Global Findex Database 2025**.")

# Check for API key and hide chat input if not found
cohere_api_key = os.environ.get("COHERE_API_KEY")
if not cohere_api_key:
    st.warning("Please set your COHERE_API_KEY in the .env file.")
else:
    # Initialize the Cohere client
    client = cohere.Client(cohere_api_key.strip())

    # Load embeddings and vector store. This is cached so it only runs once.
    @st.cache_resource
    def load_rag_components():
        """Loads the embeddings model and the FAISS index."""
        # Use the same embeddings model as in build_index.py
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return embeddings, index

    embeddings, index = load_rag_components()

    # --- CHATBOT INTERFACE ---
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if question := st.chat_input("Ask a question about the World Bank's Findex 2025 report..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            # A spinner to show that the bot is "thinking"
            with st.spinner("Searching for an answer..."):
                # Retrieval step: Find relevant document chunks
                docs = index.similarity_search(question, k=2)
                context = " ".join([doc.page_content for doc in docs])[:4000]

                # Generation step: Use the Cohere API
                try:
                    preamble = "You are a helpful financial literacy chatbot. Answer the question truthfully and based ONLY on the provided financial context."
                    message_for_api = f"Context: {context}\nQuestion: {question}"

                    response = client.chat(
                        model="command-r-08-2024",
                        message=message_for_api,
                        preamble=preamble,
                    )

                    answer = response.text
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error calling Cohere API: {e}")
                    answer = "An error occurred while generating the response."
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
