import os
import re
from collections import Counter

import streamlit as st
import PyPDF2
import pandas as pd
import altair as alt
import base64

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI


# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("RAG-based PDF Question Answering (Pinecone + OpenAI)")


# ---------------------------
# Environment variables (Render: set in dashboard)
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# If running locally, allow setting via sidebar
st.sidebar.header("Configuration")

if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
if not PINECONE_API_KEY:
    PINECONE_API_KEY = st.sidebar.text_input("Pinecone API Key", type="password")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.warning("Provide both OpenAI and Pinecone API keys to use the app.")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# ---------------------------
# Initialize OpenAI client
# ---------------------------
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


# ---------------------------
# Helper functions
# ---------------------------

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_text(text_list):
    if not client:
        raise RuntimeError("OpenAI client not initialized.")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [d.embedding for d in response.data]


def show_pdf(file):
    # For Streamlit versions that don't support st.pdf()
    file_bytes = file.read()
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# ---------------------------
# Pinecone setup
# ---------------------------
index_name = "pdf-rag-app"
pc = None
index = None

if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)


# ---------------------------
# Sidebar: PDF upload
# ---------------------------
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Session state to avoid reprocessing on every interaction
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None
if "chunk_lengths" not in st.session_state:
    st.session_state.chunk_lengths = None
if "word_list" not in st.session_state:
    st.session_state.word_list = None


# ---------------------------
# Main content area
# ---------------------------
if uploaded_pdf and OPENAI_API_KEY and PINECONE_API_KEY:
    st.subheader("PDF Preview")

    # Try native st.pdf first; fallback to iframe method
    try:
        st.pdf(uploaded_pdf)  # Available in newer Streamlit versions
    except Exception:
        show_pdf(uploaded_pdf)

    # Extract and chunk text once
    if st.session_state.raw_text is None:
        with st.spinner("Extracting text from PDF..."):
            # Need a fresh file handle for extraction
            uploaded_pdf.seek(0)
            raw_text = extract_text_from_pdf(uploaded_pdf)
            chunks = chunk_text(raw_text)

            st.session_state.raw_text = raw_text
            st.session_state.chunks = chunks
            st.session_state.chunk_lengths = [len(c) for c in chunks]

            word_list = re.findall(r"\b\w+\b", raw_text.lower())
            st.session_state.word_list = word_list

    raw_text = st.session_state.raw_text
    chunks = st.session_state.chunks
    chunk_lengths = st.session_state.chunk_lengths
    word_list = st.session_state.word_list

    st.write(f"Extracted {len(chunks)} text chunks from the document.")

    # ---------------------------
    # Compute key metrics
    # ---------------------------
    pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
    page_count = len(pdf_reader.pages)
    total_words = len(word_list)
    unique_words = len(set(word_list))

    avg_chunk_len = sum(chunk_lengths) // len(chunk_lengths) if chunk_lengths else 0

    stopwords = set([
        "the", "and", "is", "in", "to", "of", "a", "for", "on", "with",
        "as", "by", "an", "be", "this", "that", "it", "are", "or"
    ])
    filtered_words = [w for w in word_list if w not in stopwords]
    filtered_words = [w for w in filtered_words if w isalpha()]
    top_keywords = Counter(filtered_words).most_common(15)

    # ---------------------------
    # Show metrics
    # ---------------------------
    st.subheader("Document Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", page_count)
    col2.metric("Total Words", total_words)
    col3.metric("Unique Words", unique_words)

    col4, col5 = st.columns(2)
    col4.metric("Total Chunks", len(chunks))
    col5.metric("Avg Chunk Length", f"{avg_chunk_len} chars")

    # ---------------------------
    # Visualizations
    # ---------------------------
    st.subheader("Top Keywords")

    if top_keywords:
        df_keywords = pd.DataFrame(top_keywords, columns=["word", "count"])
        chart_keywords = (
            alt.Chart(df_keywords)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Frequency"),
                y=alt.Y("word:N", sort="-x", title="Keyword"),
                color="count:Q",
                tooltip=["word", "count"]
            )
            .properties(height=400)
        )
        st.altair_chart(chart_keywords, use_container_width=True)
    else:
        st.info("Not enough content to compute keyword statistics.")

    st.subheader("Chunk Length Distribution")

    if chunk_lengths:
        df_chunks = pd.DataFrame(
            {
                "chunk_index": range(len(chunks)),
                "length": chunk_lengths
            }
        )
        chart_chunks = (
            alt.Chart(df_chunks)
            .mark_line(point=True)
            .encode(
                x=alt.X("chunk_index:Q", title="Chunk Index"),
                y=alt.Y("length:Q", title="Length (chars)"),
                tooltip=["chunk_index", "length"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart_chunks, use_container_width=True)
    else:
        st.info("No chunks available to visualize.")

    # ---------------------------
    # Process & store in Pinecone
    # ---------------------------
    if index is None:
        st.error("Pinecone index is not initialized. Check your API key and configuration.")
    else:
        if st.button("Process PDF into Pinecone (Embeddings + Indexing)"):
            with st.spinner("Embedding chunks and uploading to Pinecone..."):
                embeddings = embed_text(chunks)
                vectors = []
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    vectors.append(
                        {
                            "id": f"chunk-{i}",
                            "values": emb,
                            "metadata": {"text": chunk}
                        }
                    )
                index.upsert(vectors)
            st.success("PDF has been processed and stored in Pinecone.")

    # ---------------------------
    # Query section
    # ---------------------------
    st.subheader("Ask a Question About the PDF")

    query = st.text_input("Enter your question")

    def retrieve(query_text, k=5):
        query_emb = embed_text([query_text])[0]
        results = index.query(
            vector=query_emb,
            top_k=k,
            include_metadata=True
        )
        return [m["metadata"]["text"] for m in results["matches"]]

    def answer_question(query_text):
        context_chunks = retrieve(query_text)
        context = "\n\n".join(context_chunks)

        prompt = f"""
        You are a helpful assistant that answers questions based only on the given context.

        Context:
        {context}

        Question: {query_text}

        If the answer is not clearly contained in the context, say:
        "The document does not contain that information."
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content

    if st.button("Get Answer"):
        if not query:
            st.error("Please enter a question.")
        elif index is None:
            st.error("Pinecone index is not ready.")
        else:
            with st.spinner("Generating answer..."):
                answer = answer_question(query)
            st.success("Answer:")
            st.write(answer)

else:
    st.info("Upload a PDF and set your API keys to get started.")
