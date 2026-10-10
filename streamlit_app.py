import html
import os
import warnings
from urllib.parse import quote

import cohere
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
warnings.filterwarnings("ignore")

# ── Design tokens ──
MIST   = "#E9EDF0"   # page
INK    = "#16271F"   # deep forest: all text
MUTED  = "#5C6B66"   # secondary text
LIME   = "#DDF36B"   # the accent: send button, hover, passage numbers
MOSS   = "#3F7A4D"   # links and focus rings
GLASS  = "rgba(255, 255, 255, 0.62)"   # frosted cards
EDGE   = "rgba(255, 255, 255, 0.85)"   # card edge highlight
SHADOW = "0 18px 40px -22px rgba(42, 62, 78, 0.35)"

REPO_URL = "https://github.com/Akankshaaa09/RAG-Financial-Literacy-Chatbot"

# The grainy hill in the header card, drawn as an inline SVG background.
HILL_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 640 420'>
<defs>
<linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>
<stop offset='0' stop-color='#F3FB93'/><stop offset='.26' stop-color='#CDEB6C'/>
<stop offset='.55' stop-color='#7FBA6B'/><stop offset='.82' stop-color='#4F7C70'/>
<stop offset='1' stop-color='#44546F'/>
</linearGradient>
<filter id='f' x='-10%' y='-10%' width='120%' height='120%'>
<feGaussianBlur in='SourceGraphic' stdDeviation='9' result='soft'/>
<feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' seed='7' result='noise'/>
<feColorMatrix in='noise' type='matrix' values='0 0 0 0 .09  0 0 0 0 .15  0 0 0 0 .2  1.1 0 0 0 -.3' result='grain'/>
<feComposite in='grain' in2='soft' operator='in' result='grainIn'/>
<feMerge><feMergeNode in='soft'/><feMergeNode in='grainIn'/></feMerge>
</filter>
</defs>
<g filter='url(#f)'>
<path d='M210 420 C 340 400 410 345 480 285 C 530 245 570 255 680 335 L 680 420 Z' fill='#5A6B85'/>
<path d='M-30 420 C 90 400 170 330 250 215 C 290 155 318 120 352 150 C 420 215 470 330 670 385 L 670 420 Z' fill='url(#g)'/>
</g>
</svg>"""
HILL_URL = "data:image/svg+xml;utf8," + quote(HILL_SVG.replace("\n", ""))

st.set_page_config(
    page_title="Financial Literacy Chatbot",
    page_icon="💬",
    layout="centered",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Instrument+Serif&display=swap');

  :root {{
      --fx-display: 'Instrument Serif', Georgia, 'Times New Roman', serif;
      --fx-body: 'Instrument Sans', 'Helvetica Neue', Arial, sans-serif;
  }}

  html, body, .stApp,
  [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  .stButton button, textarea, input {{
      font-family: var(--fx-body);
  }}
  .stApp {{
      color: {INK};
      background:
          radial-gradient(60rem 30rem at 88% -6%, #C9D7E2 0%, rgba(201, 215, 226, 0) 62%),
          radial-gradient(48rem 26rem at 6% 104%, #E3F0C4 0%, rgba(227, 240, 196, 0) 62%),
          {MIST};
      background-attachment: fixed;
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 2.5rem; padding-bottom: 8rem; max-width: 46rem; }}

  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li {{
      color: {INK}; font-size: 1.02rem; line-height: 1.65;
  }}
  [data-testid="stMarkdownContainer"] a {{ color: {MOSS}; }}

  /* ── Header card ── */
  .fx-hero {{
      position: relative; overflow: hidden;
      border-radius: 30px; border: 1px solid {EDGE};
      background:
          url("{HILL_URL}") right -3.5rem bottom -2.2rem / 64% auto no-repeat,
          linear-gradient(135deg, rgba(255, 255, 255, 0.86), rgba(255, 255, 255, 0.5));
      box-shadow: {SHADOW};
      padding: 1.6rem 1.9rem 2.2rem 1.9rem;
      min-height: 21rem;
      margin-bottom: 0.9rem;
  }}
  [data-testid="stMarkdownContainer"] p.fx-name {{
      font-size: 0.92rem; font-weight: 600; color: {INK}; margin: 0 0 2.4rem 0;
  }}
  [data-testid="stMarkdownContainer"] h1.fx-title {{
      font-family: var(--fx-display); font-weight: 400;
      font-size: clamp(2.35rem, 7.4vw, 3.5rem);
      line-height: 1.02; letter-spacing: -0.012em;
      color: {INK}; max-width: 26rem; padding: 0; margin: 0 0 1.1rem 0;
  }}
  [data-testid="stMarkdownContainer"] p.fx-lede {{
      font-size: 1rem; line-height: 1.55; color: {INK};
      max-width: 21.5rem; margin: 0;
  }}

  /* ── How it works: three small frosted tiles ── */
  .fx-facts {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem;
      margin: 0 0 2.2rem 0;
  }}
  .fx-facts > div {{
      background: {GLASS}; border: 1px solid {EDGE}; border-radius: 20px;
      padding: 0.95rem 1.05rem; box-shadow: {SHADOW};
  }}
  .fx-facts dt {{ font-size: 0.8rem; color: {MUTED}; margin-bottom: 0.25rem; }}
  .fx-facts dd {{ margin: 0; font-size: 0.9rem; line-height: 1.4; color: {INK}; }}
  .fx-facts strong {{ font-weight: 600; }}

  /* ── Suggested questions ── */
  [data-testid="stMarkdownContainer"] p.fx-prompt {{
      font-size: 0.9rem; color: {MUTED}; margin: 0 0 0.5rem 0.2rem;
  }}
  .stButton > button {{
      background: {GLASS}; color: {INK};
      border: 1px solid {EDGE}; border-radius: 20px;
      padding: 0.6rem 1.05rem;
      justify-content: flex-start; text-align: left;
      box-shadow: {SHADOW};
      transition: background .15s ease, border-color .15s ease;
  }}
  .stButton > button p {{
      font-size: 0.95rem; line-height: 1.35; font-weight: 500;
      white-space: normal; overflow: visible; text-overflow: clip; text-align: left;
  }}
  [class*="st-key-suggest_"] .stButton > button {{ min-height: 4.5rem; padding: 0.85rem 1.1rem; }}
  .stButton > button:hover,
  .stButton > button:active,
  .stButton > button:focus:not(:focus-visible) {{
      background: {LIME}; border-color: {LIME}; color: {INK};
  }}
  .stButton > button:focus-visible {{ outline: 2px solid {MOSS}; outline-offset: 2px; }}
  .st-key-reset {{ margin-top: 0.4rem; }}
  .st-key-reset .stButton > button {{ border-radius: 999px; padding: 0.4rem 1rem; }}
  .st-key-reset .stButton > button p {{ font-size: 0.88rem; }}

  /* ── One question and its answer: a frosted card ── */
  [data-testid="stVerticalBlock"]:has(> .element-container p.fx-question) {{
      background: {GLASS}; border: 1px solid {EDGE}; border-radius: 26px;
      box-shadow: {SHADOW};
      padding: 1.5rem 1.6rem 1.1rem 1.6rem;
      margin-bottom: 0.4rem;
  }}
  [data-testid="stMarkdownContainer"] p.fx-question {{
      font-family: var(--fx-display); font-weight: 400;
      font-size: clamp(1.6rem, 4.6vw, 2.05rem);
      line-height: 1.12; letter-spacing: -0.008em;
      color: {INK}; margin: 0 0 0.5rem 0;
  }}
  [data-testid="stMarkdownContainer"] p.fx-problem {{
      font-size: 0.85rem; color: {MUTED}; margin-top: -0.4rem;
  }}

  /* ── Retrieved passages ── */
  [data-testid="stExpander"] details {{
      border: 0; border-top: 1px solid rgba(22, 39, 31, 0.12);
      border-radius: 0; background: transparent;
  }}
  [data-testid="stExpander"] summary {{ padding-left: 0; color: {MUTED}; background: transparent; }}
  [data-testid="stExpander"] summary:hover {{ color: {INK}; background: transparent; }}
  [data-testid="stExpander"] summary p {{ font-size: 0.9rem; color: inherit; }}
  [data-testid="stExpanderDetails"] {{ padding-left: 0; padding-right: 0; }}
  .fx-passage {{
      display: grid; grid-template-columns: 1.7rem 1fr; gap: 0.7rem; align-items: start;
      background: rgba(255, 255, 255, 0.7); border-radius: 16px;
      padding: 0.85rem 1rem; margin: 0 0 0.5rem 0;
      font-size: 0.88rem; line-height: 1.55; color: {INK};
  }}
  .fx-passage b {{
      display: grid; place-items: center; width: 1.7rem; height: 1.7rem;
      border-radius: 50%; background: {LIME}; color: {INK};
      font-size: 0.8rem; font-weight: 600;
  }}

  /* ── Question box ── */
  [data-testid="stBottom"], [data-testid="stBottom"] > div,
  [data-testid="stBottomBlockContainer"] {{ background: transparent; }}
  [data-testid="stBottomBlockContainer"] {{ padding-bottom: 2rem; }}
  [data-testid="stChatInput"] > div {{
      background: rgba(255, 255, 255, 0.92); border: 1px solid {EDGE}; border-radius: 999px;
      box-shadow: 0 16px 36px -16px rgba(42, 62, 78, 0.45);
      padding-left: 0.6rem;
  }}
  [data-testid="stChatInput"] > div:focus-within {{
      border-color: {MOSS}; box-shadow: 0 0 0 3px rgba(63, 122, 77, 0.22);
  }}
  [data-testid="stChatInput"] textarea {{ color: {INK}; font-size: 1rem; }}
  [data-testid="stChatInput"] textarea::placeholder {{ color: {MUTED}; opacity: 1; }}
  [data-testid="stChatInputSubmitButton"] {{
      background: {LIME} !important; color: {INK} !important; border-radius: 50% !important;
  }}
  [data-testid="stChatInputSubmitButton"] svg {{ fill: {INK}; color: {INK}; }}

  [data-testid="stMarkdownContainer"] p.fx-footer {{
      font-size: 0.85rem; color: {MUTED}; margin: 2.2rem 0 0 0.2rem;
  }}
  [data-testid="stMarkdownContainer"] p.fx-footer a {{ color: {INK}; text-underline-offset: 3px; }}

  .stSpinner > div {{ border-top-color: {MOSS} !important; }}

  @media (max-width: 640px) {{
      .block-container {{ padding-top: 1.2rem; }}
      .fx-hero {{
          padding: 1.3rem 1.3rem 11.5rem 1.3rem; border-radius: 26px;
          background:
              url("{HILL_URL}") right -3rem bottom -1.6rem / 118% auto no-repeat,
              linear-gradient(135deg, rgba(255, 255, 255, 0.86), rgba(255, 255, 255, 0.5));
      }}
      [data-testid="stMarkdownContainer"] p.fx-name {{ margin-bottom: 1.6rem; }}
      [data-testid="stMarkdownContainer"] p.fx-lede {{ max-width: none; }}
      .fx-facts {{ grid-template-columns: 1fr; gap: 0.6rem; }}
      [data-testid="stVerticalBlock"]:has(> .element-container p.fx-question) {{
          padding: 1.2rem 1.15rem 0.9rem 1.15rem; border-radius: 22px;
      }}
  }}
  @media (prefers-reduced-motion: reduce) {{
      .stButton > button {{ transition: none; }}
  }}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="fx-hero">
  <p class="fx-name">Financial Literacy Chatbot</p>
  <h1 class="fx-title">Who has a bank account, and who is still left out?</h1>
  <p class="fx-lede">
    Ask about financial inclusion around the world. Answers are written only from
    passages retrieved from the World Bank's Global Findex Database 2025, and the
    passages are listed under each answer so you can check it.
  </p>
</div>
<dl class="fx-facts">
  <div><dt>Source</dt><dd>World Bank Global Findex Database 2025</dd></div>
  <div><dt>Retrieval</dt><dd>Top 4 passages by semantic search. The right one was
       found for <strong>80% of 25 test questions</strong>.</dd></div>
  <div><dt>Answering</dt><dd>Cohere command-r, limited to the retrieved text</dd></div>
</dl>
""", unsafe_allow_html=True)

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Index loader (silent) ──
cohere_api_key = os.environ.get("COHERE_API_KEY")


@st.cache_resource(show_spinner="Loading the report. This takes a few seconds the first time.")
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


def passage_snippets(docs):
    """Shorten each retrieved passage for display under the answer."""
    snippets = []
    for doc in docs:
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320].rsplit(" ", 1)[0] + "…"
        snippets.append(snippet)
    return snippets


def render_question(text):
    st.markdown(
        f"<p class='fx-question'>{html.escape(text)}</p>",
        unsafe_allow_html=True,
    )


def render_answer(message):
    st.markdown(message["content"])
    if message.get("detail"):
        st.markdown(
            f"<p class='fx-problem'>{html.escape(message['detail'])}</p>",
            unsafe_allow_html=True,
        )
    sources = message.get("sources") or []
    if sources:
        with st.expander(f"Read the {len(sources)} passages this answer came from"):
            for rank, snippet in enumerate(sources, 1):
                st.markdown(
                    f"<div class='fx-passage'><b>{rank}</b>"
                    f"<span>{html.escape(snippet)}</span></div>",
                    unsafe_allow_html=True,
                )


if not cohere_api_key:
    st.error(
        "COHERE_API_KEY is missing. Add it to the app's secrets, then restart the app."
    )
else:
    embeddings, index = get_index()
    client = cohere.Client(cohere_api_key.strip())

    # ── Question box, always rendered ──
    typed = st.chat_input("Ask about financial inclusion")
    question = st.session_state.pop("suggested_question", None) or typed

    # ── Suggested questions (only before the first question) ──
    if not st.session_state.messages and not question:
        st.markdown("<p class='fx-prompt'>Try one of these</p>", unsafe_allow_html=True)
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

    # ── Conversation so far: one card per question ──
    history = st.session_state.messages
    for i in range(0, len(history) - 1, 2):
        with st.container():
            render_question(history[i]["content"])
            render_answer(history[i + 1])

    if question:
        pending_card = st.container()
        with pending_card:
            render_question(question)

        # ── Short-term memory: last exchange only, used ONLY to interpret ──
        # ── follow-up phrasing (pronouns, "what about X") — never as a  ──
        # ── source of facts. Answers still must come from retrieved     ──
        # ── context below, so grounding/hallucination-prevention holds. ──
        history_note = ""
        if len(st.session_state.messages) >= 2:
            prev_user = st.session_state.messages[-2]["content"]
            prev_answer = st.session_state.messages[-1]["content"]
            history_note = (
                f"Previous exchange (for context only, not a source of facts):\n"
                f"User asked: {prev_user}\n"
                f"You answered: {prev_answer}\n\n"
            )

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with pending_card:
            with st.spinner("Searching the Findex report..."):
                docs = index.similarity_search(question, k=4)
                context = " ".join(
                    [doc.page_content for doc in docs]
                )[:4000]

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
                    message_for_api = (
                        f"{history_note}Context: {context}\nQuestion: {question}"
                    )
                    response = client.chat(
                        model="command-r-08-2024",
                        message=message_for_api,
                        preamble=preamble,
                    )
                    reply = {
                        "role": "assistant",
                        "content": response.text,
                        "sources": passage_snippets(docs),
                    }
                except Exception as e:
                    reply = {
                        "role": "assistant",
                        "content": "No answer this time: the request to the language "
                                   "model failed. Ask again in a moment.",
                        "detail": f"Cohere API error: {e}",
                    }

        # Saved with its passages, so they stay under the answer on later questions.
        st.session_state.messages.append(reply)
        st.rerun()

    if st.session_state.messages:
        if st.button("Start a new conversation", key="reset"):
            st.session_state.messages = []
            st.rerun()

st.markdown(
    f"<p class='fx-footer'>Built by Akanksha Nayak. "
    f"<a href='{REPO_URL}' target='_blank'>Code on GitHub</a></p>",
    unsafe_allow_html=True,
)
