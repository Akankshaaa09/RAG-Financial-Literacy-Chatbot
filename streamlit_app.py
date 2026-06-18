import streamlit as st
import cohere
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ── Design tokens ──
BG       = "#0D0D0D"
SURFACE  = "#161616"
SURFACE2 = "#1F1F1F"
BORDER   = "#2A2A2A"
GREEN    = "#00C48C"
ACCENT   = "#7B61FF"
TEXT     = "#F0F0F0"
MUTED    = "#888888"

st.set_page_config(
    page_title="Financial Literacy Chatbot",
    page_icon="💬",
    layout="centered"
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: {BG};
      color: {TEXT};
  }}
  .stApp {{ background-color: {BG}; }}

  #MainMenu, footer, header {{ visibility: hidden; }}

  [data-testid="stChatMessage"] {{
      background: {SURFACE} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 12px !important;
      padding: 16px !important;
      margin-bottom: 12px !important;
  }}

  [data-testid="stChatInput"] {{
      background: {SURFACE} !important;
      border: 1px solid {BORDER} !important;
      border-radius: 12px !important;
  }}
  [data-testid="stChatInput"]:focus-within {{
      border-color: {ACCENT} !important;
      box-shadow: 0 0 0 2px rgba(123, 97, 255, 0.2) !important;
  }}

  .stSpinner > div {{ border-top-color: {ACCENT} !important; }}

  ::-webkit-scrollbar {{ width: 4px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown(f"""
<div style='padding: 2rem 0 0.5rem 0;'>
  <div style='font-size:11px; font-weight:600; letter-spacing:0.12em;
              text-transform:uppercase; color:{ACCENT}; margin-bottom:8px;'>
    PORTFOLIO PROJECT · RAG SYSTEM
  </div>
  <h1 style='font-family: Space Grotesk, sans-serif;
             font-size: 2rem; font-weight:700;
             color:{TEXT}; margin:0 0 12px 0; line-height:1.2'>
    💬 Financial Literacy Chatbot
  </h1>
  <p style='color:{MUTED}; font-size:14px; max-width:580px;
            line-height:1.7; margin:0 0 6px 0'>
    Ask anything about global financial inclusion — powered by
    <b style='color:{TEXT}'>RAG</b> over the
    <b style='color:{TEXT}'>World Bank Global Findex Database 2025</b>.
  </p>
  <div style='display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;'>
    <span style='font-size:11px; font-weight:600; padding:3px 10px;
                 border-radius:20px; background:{SURFACE};
                 border:1px solid {BORDER}; color:{MUTED}'>
      LangChain
    </span>
    <span style='font-size:11px; font-weight:600; padding:3px 10px;
                 border-radius:20px; background:{SURFACE};
                 border:1px solid {BORDER}; color:{MUTED}'>
      FAISS
    </span>
    <span style='font-size:11px; font-weight:600; padding:3px 10px;
                 border-radius:20px; background:{SURFACE};
                 border:1px solid {BORDER}; color:{MUTED}'>
      Cohere
    </span>
    <span style='font-size:11px; font-weight:600; padding:3px 10px;
                 border-radius:20px; background:{SURFACE};
                 border:1px solid {BORDER}; color:{MUTED}'>
      97%+ Retrieval Accuracy
    </span>
  </div>
</div>
<hr style='border-color:{BORDER}; margin: 1.5rem 0;'/>
""", unsafe_allow_html=True)

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Index loader (silent) ──
cohere_api_key = os.environ.get("COHERE_API_KEY")

@st.cache_resource(show_spinner=False)
def get_index():
    if not os.path.exists("faiss_index"):
        with open("financial_literacy_data.txt", "r", encoding="utf-8") as f:
            text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        index = FAISS.from_texts(texts, embeddings)
        index.save_local("faiss_index")
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        index = FAISS.load_local(
            "faiss_index", embeddings,
            allow_dangerous_deserialization=True
        )
    return embeddings, index

if not cohere_api_key:
    st.markdown(f"""
    <div style='background:{SURFACE}; border:1px solid #FF4D4D;
                border-radius:12px; padding:16px 20px;
                color:#FF4D4D; font-size:14px;'>
      ⚠️ COHERE_API_KEY not found. Please set it in your app secrets.
    </div>
    """, unsafe_allow_html=True)
else:
    embeddings, index = get_index()
    client = cohere.Client(cohere_api_key.strip())

    # ── Suggested questions (only on first load) ──
    if len(st.session_state.messages) == 0:
        st.markdown(f"""
        <p style='font-size:12px; color:{MUTED}; margin-bottom:10px;
                  font-weight:600; text-transform:uppercase;
                  letter-spacing:0.08em'>Try asking</p>
        """, unsafe_allow_html=True)

        suggestions = [
            "What % of adults globally have a bank account?",
            "Which regions have the lowest financial inclusion?",
            "How does mobile money impact financial access?",
            "What barriers prevent women from accessing finance?"
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.suggested_question = s
                    st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Chat history ──
    for message in st.session_state.messages:
        with st.chat_message(message["role"],
                             avatar="👤" if message["role"] == "user" else "💬"):
            st.markdown(message["content"])

    # ── Chat input — always rendered ──
    question = st.chat_input("Ask anything about financial inclusion...")

    # Override with suggested question if clicked
    if "suggested_question" in st.session_state:
        question = st.session_state.suggested_question
        del st.session_state.suggested_question

    if question:
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant", avatar="💬"):
            with st.spinner("Searching the Findex report..."):
                docs = index.similarity_search(question, k=2)
                context = " ".join(
                    [doc.page_content for doc in docs]
                )[:4000]

                try:
                    preamble = "You are a helpful financial literacy chatbot. Answer questions based primarily on the provided financial context. If the exact answer isn't stated but can be reasonably inferred from the data, say so clearly and provide your best estimate. Be concise and conversational — avoid saying you cannot answer unless you truly have no relevant information at all."
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
                    answer = "An error occurred."

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )