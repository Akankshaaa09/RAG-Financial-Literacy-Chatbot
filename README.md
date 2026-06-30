# RAG Financial Literacy Chatbot

A Retrieval-Augmented Generation system that answers questions over financial literacy and financial inclusion data using semantic search and LLM-based generation.

🔗 **[Live App](https://rag-financial-literacy-chatbot-dzmtmbgqryzcxengvbo6t4.streamlit.app/)**

---

## What It Does

This chatbot answers plain-language questions about financial inclusion and literacy by retrieving relevant passages from source documents (Global Findex Database 2025 data) and generating grounded, conversational answers — rather than relying on the LLM's general knowledge alone.

---

## Key Result

**Precision@4: 80%** on a 25-question hand-labelled evaluation set, validated via a custom retrieval evaluation script (`evaluate.py`).

---

## Architecture

```
Document (.txt)
      ↓
Chunking (RecursiveCharacterTextSplitter, chunk_size=1000, overlap=200)
      ↓
Embedding (HuggingFace all-mpnet-base-v2, local, no API cost)
      ↓
FAISS Vector Index
      ↓
Query → Semantic Search (top-k=4) → Retrieved Chunks
      ↓
Cohere command-r-08-2024 (Generation, grounded in retrieved context)
      ↓
Answer
```

---

## Evaluation Methodology

Retrieval quality was measured using a custom-built evaluation script (`evaluate.py`):

1. 25 question/answer pairs were hand-written from verified facts in the source document
2. Each pair includes a question and a distinctive keyword/phrase that should appear in a correctly-retrieved chunk
3. For each question, the script runs retrieval and checks whether the target phrase appears in the retrieved context
4. Precision@k is calculated as (correct retrievals / total questions)

**Results:**

| k | Precision |
|---|---|
| k=2 | 72.0% |
| k=4 | 80.0% |

k=4 was selected for production based on this evaluation — it meaningfully improves retrieval without materially increasing context size sent to the LLM.

Run the evaluation yourself:
```bash
python evaluate.py
```

---

## Hallucination Prevention

The system prompt explicitly instructs the model to answer only from retrieved context, and to clearly state when the context does not contain a relevant answer rather than inferring or guessing. This is validated qualitatively — out-of-domain questions reliably trigger an "I don't have that information" response rather than a fabricated answer.

---

## Stack

Python · LangChain · FAISS · HuggingFace Embeddings (all-mpnet-base-v2) · Cohere (command-r-08-2024) · Streamlit

---

## Repo Structure

```
├── build_index.py        # Chunks source text, builds FAISS index
├── evaluate.py            # Retrieval evaluation script (Precision@k)
├── qa_bot.py               # CLI version for local testing
├── streamlit_app.py        # Deployed Streamlit application
├── financial_literacy_data.txt   # Source document
└── requirements.txt
```

---

## How to Run Locally

```bash
git clone https://github.com/Akankshaaa09/RAG-Financial-Literacy-Chatbot
cd RAG-Financial-Literacy-Chatbot
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Set your Cohere API key
echo COHERE_API_KEY=your_key_here > .env

# Build the index (first time only)
python build_index.py

# Run the evaluation (optional)
python evaluate.py

# Run the Streamlit app
streamlit run streamlit_app.py
```

---

**Akanksha Nayak**