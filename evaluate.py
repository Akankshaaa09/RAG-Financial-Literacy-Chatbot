# evaluate.py
# Retrieval evaluation script for the RAG Financial Literacy Chatbot.
#
# Measures retrieval precision@k against a hand-labelled question set
# built from real facts in financial_literacy_data.txt (Global Findex
# Database 2025 content).
#
# Run this AFTER build_index.py has created the faiss_index folder.
# Usage: python evaluate.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore")

print("Loading embeddings and index...")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
index = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
print("Index loaded.\n")

# ════════════════════════════════════════════════════════════
# EVALUATION SET
# 25 question/answer pairs built from real facts in the Global
# Findex Database 2025 text. Each "must_contain" is a distinctive
# number or phrase that proves the correct chunk was retrieved.
# ════════════════════════════════════════════════════════════

EVAL_SET = [
    {"question": "What percentage of adults worldwide own mobile phones?",
     "must_contain": "86 percent"},
    {"question": "How did global account ownership change between 2011 and 2024?",
     "must_contain": "51 percent to 79 percent"},
    {"question": "What percentage of adults in low- and middle-income economies use the internet?",
     "must_contain": "67 percent"},
    {"question": "What percentage of smartphone owners use the internet?",
     "must_contain": "92 percent"},
    {"question": "What percentage of mobile phone owners have a password on their phone in low- and middle-income economies?",
     "must_contain": "60 percent"},
    {"question": "What is the most common reason adults in Sub-Saharan Africa don't have mobile money accounts?",
     "must_contain": "Lack of money"},
    {"question": "In India, what is the most common reason adults without accounts give for not having one?",
     "must_contain": "family member with an account"},
    {"question": "What percentage of adults without an account in low- and middle-income economies own a smartphone?",
     "must_contain": "42 percent"},
    {"question": "Roughly what percentage of adults in low- and middle-income economies pay bills online?",
     "must_contain": "37 percent"},
    {"question": "What percentage of adults paid utility bills digitally?",
     "must_contain": "40 percent"},
    {"question": "What is the most common reason people don't pay merchants digitally?",
     "must_contain": "paying in cash"},
    {"question": "In India, what percentage of account owners do not have an active account?",
     "must_contain": "16 percent"},
    {"question": "What is the average rate of inactive accounts in other low- and middle-income economies excluding India?",
     "must_contain": "4 percent"},
    {"question": "What do adults worry about most financially?",
     "must_contain": "monthly bills and medical expenses"},
    {"question": "What is a major source of financial worry in Sub-Saharan Africa?",
     "must_contain": "School fees"},
    {"question": "What percentage of adults could cover more than two months of expenses if they lost their main income?",
     "must_contain": "one-third"},
    {"question": "What percentage of adults receive remittances from abroad?",
     "must_contain": "10 percent"},
    {"question": "What percentage of bank account owners checked balances through digital channels in the past year?",
     "must_contain": "Forty percent"},
    {"question": "What share of adults without bank accounts would need help using one?",
     "must_contain": "Two-thirds"},
    {"question": "How many economies account for more than half of adults without accounts?",
     "must_contain": "eight economies"},
    {"question": "What percentage of adults in Ghana, Kenya, and Uganda borrowed from mobile money providers in 2024?",
     "must_contain": "20 percent"},
    {"question": "What share of online shoppers also pay for purchases online in low- and middle-income economies?",
     "must_contain": "two-thirds"},
    {"question": "Are women or men more likely to own a mobile phone?",
     "must_contain": "Women"},
    {"question": "What is the biggest barrier to smartphone ownership and internet use?",
     "must_contain": "Lack of money"},
    {"question": "What percentage of mobile phone owners in low- and middle-income economies have a SIM card not registered in their name?",
     "must_contain": "SIM card"},
]

K = 4  # matches production retrieval setting in qa_bot.py / streamlit_app.py

# ════════════════════════════════════════════════════════════
# RUN EVALUATION
# ════════════════════════════════════════════════════════════

results = []
print(f"Running retrieval evaluation (k={K}) on {len(EVAL_SET)} questions...\n")

for i, item in enumerate(EVAL_SET, 1):
    question = item["question"]
    target   = item["must_contain"].lower()

    docs = index.similarity_search(question, k=K)
    retrieved_text = " ".join([d.page_content for d in docs]).lower()

    hit = target in retrieved_text
    results.append({"question": question, "hit": hit})

    status = "HIT " if hit else "MISS"
    print(f"{i:2d}. [{status}]  {question}")

# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════

n_hits  = sum(r["hit"] for r in results)
n_total = len(results)
precision_at_k = n_hits / n_total * 100

print(f"\n{'='*55}")
print(f"RETRIEVAL EVALUATION RESULTS")
print(f"{'='*55}")
print(f"Questions evaluated : {n_total}")
print(f"Correct retrievals  : {n_hits}")
print(f"Precision@{K}          : {precision_at_k:.1f}%")
print(f"{'='*55}")

misses = [r["question"] for r in results if not r["hit"]]
if misses:
    print(f"\nMissed questions (for debugging / chunking improvements):")
    for m in misses:
        print(f"  - {m}")

print(f"\nDone. Use this Precision@{K} number in your README and resume.")