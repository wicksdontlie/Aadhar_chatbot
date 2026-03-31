# ============================================================
# CELL 1 - Install dependencies
# ============================================================
# !pip install streamlit pdfplumber transformers torch sentencepiece nltk pyngrok

# ============================================================
# CELL 2 - Write the app file
# ============================================================
# %%writefile app.py

import streamlit as st
import pdfplumber
import nltk
import warnings
import io

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize
from transformers import pipeline

# ── Page Setup ───────────────────────────────────────────────
st.set_page_config(page_title="Aadhaar Chatbot", page_icon="🪪")
st.title("🪪 Aadhaar RAG Chatbot")
st.caption("Upload your Aadhaar PDF and ask any question about it.")

# ── Load Models (cached so they load only once) ───────────────
@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

# ── Helper: Extract text from PDF ────────────────────────────
@st.cache_data
def extract_text(pdf_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# ── Helper: Split text into passages ─────────────────────────
def get_passages(text, max_words=200):
    sentences = sent_tokenize(text)
    passages, current, count = [], [], 0
    for sent in sentences:
        wc = len(sent.split())
        if count + wc <= max_words:
            current.append(sent)
            count += wc
        else:
            if current:
                passages.append(" ".join(current))
            current, count = [sent], wc
    if current:
        passages.append(" ".join(current))
    return passages

# ── Helper: Find best answer across all passages ──────────────
def get_answer(question, passages, qa_pipe):
    best = {"answer": "I couldn't find an answer.", "score": 0.0}
    for passage in passages:
        try:
            result = qa_pipe({"question": question, "context": passage})
            if result["score"] > best["score"]:
                best = result
        except:
            continue
    return best

# ── Sidebar: Upload PDF ───────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Session state for chat history ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── If no PDF uploaded yet ────────────────────────────────────
if not uploaded_file:
    st.info("👈 Upload an Aadhaar PDF from the sidebar to get started.")
    st.stop()

# ── Process PDF ───────────────────────────────────────────────
with st.spinner("Reading PDF..."):
    text = extract_text(uploaded_file.read())
    passages = get_passages(text)

st.sidebar.markdown(f"**Passages indexed:** {len(passages)}")

# ── Load QA model ─────────────────────────────────────────────
with st.spinner("Loading QA model (first time only)..."):
    qa_pipe = load_qa()

# ── Display chat history ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "confidence" in msg:
            st.caption(f"Confidence: {round(msg['confidence'] * 100, 1)}%")

# ── Chat input ────────────────────────────────────────────────
question = st.chat_input("Ask something about Aadhaar...")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            result = get_answer(question, passages, qa_pipe)

        answer = result["answer"]
        score  = result["score"]

        if score < 0.05:
            answer = "I couldn't find a confident answer. Try rephrasing your question."

        st.write(answer)
        st.caption(f"Confidence: {round(score * 100, 1)}%")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": score
    })


# ============================================================
# CELL 3 - Launch the app
# ============================================================
# from pyngrok import ngrok
# import subprocess, time
#
# # Get free token from: https://dashboard.ngrok.com
# ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")
#
# subprocess.Popen(["streamlit", "run", "app.py",
#     "--server.port=8501", "--server.headless=true"])
# time.sleep(3)
#
# url = ngrok.connect(8501).public_url
# print(f"Open this link: {url}")
