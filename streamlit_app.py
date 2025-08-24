import streamlit as st
import cohere
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
from dotenv import load_dotenv

# --- SETUP AND INITIALIZATION ---
load_dotenv()
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Financial Literacy RAG Chatbot",
    page_icon="🤖"
)
st.title("Financial Literacy RAG Chatbot 🤖")
st.markdown("I am a chatbot that provides answers about financial literacy, based on the **World Bank's Global Findex Database 2025**.")

cohere_api_key = os.environ.get("COHERE_API_KEY")

# --- NEW: Check and rebuild index if it doesn't exist ---
@st.cache_resource
def get_index():
    if not os.path.exists("faiss_index"):
        st.warning("Index not found. Rebuilding from source document...")
        
        # Load the extracted text
        with open("financial_literacy_data.txt", "r", encoding="utf-8") as f:
            text = f.read()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_text(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        
        # Build the FAISS vector store index
        index = FAISS.from_texts(texts, embeddings)
        index.save_local("faiss_index")
        st.success("Index rebuilt successfully! You can now use the chatbot.")
    else:
        # Load embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return embeddings, index

if not cohere_api_key:
    st.warning("Please set your COHERE_API_KEY in the app secrets.")
else:
    embeddings, index = get_index()
    client = cohere.Client(cohere_api_key.strip())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about the World Bank's Findex 2025 report..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Searching for an answer..."):
                docs = index.similarity_search(question, k=2)
                context = " ".join([doc.page_content for doc in docs])[:4000]

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
            st.session_state.messages.append({"role": "assistant", "content": answer})

