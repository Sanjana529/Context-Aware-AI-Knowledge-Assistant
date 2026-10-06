import os
import tempfile
from pathlib import Path

import streamlit as st
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

DB_PATH = "vectorstore"
COLLECTION_NAME = "knowledge_documents"
MODEL_NAME = "gemini-3.5-flash-lite"

st.set_page_config(
    page_title="Context-Aware AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = set()


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Context-Aware AI Knowledge Assistant")

st.subheader("Intelligent Document-Based Question Answering")

st.write(
    "Upload documents, ask questions, generate summaries, "
    "MCQs and flashcards."
)


# ============================================================
# CHROMADB
# ============================================================

Path(DB_PATH).mkdir(
    parents=True,
    exist_ok=True
)

chroma_client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# LIGHTWEIGHT EMBEDDINGS
# ============================================================

vectorizer = HashingVectorizer(
    n_features=256,
    stop_words="english",
    alternate_sign=False,
    norm="l2"
)


def create_embeddings(texts):

    return vectorizer.transform(
        texts
    ).toarray().tolist()


# ============================================================
# GEMINI
# ============================================================

def ask_gemini(prompt, mode="Balanced"):

    if not API_KEY:
        return "❌ GOOGLE_API_KEY is missing in .env"

    temperature_map = {
        "Precise": 0.1,
        "Balanced": 0.3,
        "Detailed": 0.5
    }

    temperature = temperature_map.get(
        mode,
        0.3
    )

    try:

        client = genai.Client(
            api_key=API_KEY
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,
                top_p=0.9
            )
        )

        if response.text:
            return response.text

        return "No response was generated."

    except Exception as error:

        return f"Gemini Error: {error}"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(path):

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:
            text = page.extract_text()
        except Exception:
            text = None

        if text and text.strip():

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


# ============================================================
# CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=800,
    overlap=150
):

    chunks = []

    start = 0

    while start < len(text):

        chunk = text[
            start:start + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path,
    document_name
):

    pages = extract_pdf_text(
        pdf_path
    )

    texts = []
    ids = []
    metadatas = []

    for page_data in pages:

        chunks = split_text(
            page_data["text"]
        )

        for index, chunk in enumerate(chunks):

            texts.append(chunk)

            ids.append(
                f"{document_name}_{page_data['page']}_{index}"
            )

            metadatas.append({
                "source": document_name,
                "page": page_data["page"],
                "chunk": index
            })

    if not texts:
        return 0

    embeddings = create_embeddings(
        texts
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(texts)


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    question,
    top_k=5
):

    if collection.count() == 0:
        return []

    query_embedding = create_embeddings(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    output = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        output.append({
            "text": document,
            "metadata": metadata
        })

    return output


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for item in results:

        source = item["metadata"]["source"]
        page = item["metadata"]["page"]

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{item["text"]}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    question,
    top_k,
    mode
):

    results = search_documents(
        question,
        top_k
    )

    if not results:

        return (
            "Information not found in the document.",
            []
        )

    context = build_context(
        results
    )

    previous_messages = ""

    if st.session_state.messages:

        previous_messages = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in st.session_state.messages[-6:]
        )

    prompt = f"""
You are a Context-Aware AI Knowledge Assistant.

Answer the user's question using ONLY
the provided document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available, say:
"Information not found in the document."
4. Give a clear answer.
5. Use recent conversation to understand follow-up questions.

Answer mode:

{mode}

Recent conversation:

{previous_messages}

Document context:

{context}

User question:

{question}
"""

    answer = ask_gemini(
        prompt,
        mode
    )

    return answer, results


# ============================================================
# SUMMARY
# ============================================================

def summarize_document():

    if collection.count() == 0:
        return "No documents available."

    results = collection.get(
        include=["documents"]
    )

    documents = results.get(
        "documents",
        []
    )

    if not documents:
        return "No document text available."

    combined_text = "\n\n".join(
        documents[:30]
    )

    prompt = f"""
Summarize the following document.

Give:

1. Executive Summary
2. Main Concepts
3. Important Points
4. Key Takeaways

Use ONLY the provided document.

DOCUMENT:

{combined_text}
"""

    return ask_gemini(
        prompt,
        "Balanced"
    )


# ============================================================
# MCQ GENERATOR
# ============================================================

def generate_mcqs():

    if collection.count() == 0:
        return "No documents available."

    results = collection.get(
        include=["documents"]
    )

    documents = results.get(
        "documents",
        []
    )

    context = "\n\n".join(
        documents[:20]
    )

    prompt = f"""
Create 5 multiple-choice questions
from the document below.

For each question provide:

Question:
A)
B)
C)
D)

Correct Answer:
Explanation:

Use ONLY the document.

DOCUMENT:

{context}
"""

    return ask_gemini(
        prompt,
        "Balanced"
    )


# ============================================================
# FLASHCARDS
# ============================================================

def generate_flashcards():

    if collection.count() == 0:
        return "No documents available."

    results = collection.get(
        include=["documents"]
    )

    documents = results.get(
        "documents",
        []
    )

    context = "\n\n".join(
        documents[:20]
    )

    prompt = f"""
Create 10 study flashcards
from the document.

Format:

CARD 1
Question:
Answer:

CARD 2
Question:
Answer:

Keep answers concise.

Use ONLY the document.

DOCUMENT:

{context}
"""

    return ask_gemini(
        prompt,
        "Precise"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    mode = st.selectbox(
        "🎯 Answer Mode",
        [
            "Precise",
            "Balanced",
            "Detailed"
        ],
        index=1
    )

    top_k = st.slider(
        "🔎 Retrieved Chunks",
        1,
        10,
        5
    )

    st.divider()

    st.subheader("📊 Knowledge Base")

    st.metric(
        "Stored Chunks",
        collection.count()
    )

    document_data = collection.get(
        include=["metadatas"]
    )

    metadata_list = document_data.get(
        "metadatas",
        []
    )

    document_names = set()

    for metadata in metadata_list:

        if metadata:

            document_names.add(
                metadata.get(
                    "source",
                    "Unknown"
                )
            )

    st.metric(
        "Documents",
        len(document_names)
    )

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True
    ):

        chroma_client.delete_collection(
            name=COLLECTION_NAME
        )

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        st.session_state.processed_documents = set()

        st.success(
            "Knowledge base cleared."
        )

        st.rerun()


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header("📚 Document Management")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.write(
        f"📄 {len(uploaded_files)} document(s) selected."
    )

    if st.button(
        "🚀 Process Documents",
        type="primary"
    ):

        total_chunks = 0

        progress = st.progress(0)

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            document_name = uploaded_file.name

            with st.spinner(
                f"Processing {document_name}..."
            ):

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp:

                        temp.write(
                            uploaded_file.getvalue()
                        )

                        temp_path = temp.name

                    count = process_pdf(
                        temp_path,
                        document_name
                    )

                    os.remove(
                        temp_path
                    )

                    total_chunks += count

                    st.success(
                        f"✅ {document_name}: "
                        f"{count} chunks"
                    )

                except Exception as error:

                    st.error(
                        f"❌ {document_name}: {error}"
                    )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        st.success(
            f"🎉 Processing complete! "
            f"{total_chunks} chunks added."
        )


# ============================================================
# CHAT
# ============================================================

st.divider()

st.header("💬 AI Knowledge Assistant")

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


question = st.chat_input(
    "Ask something about your documents..."
)


if question:

    if collection.count() == 0:

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching documents..."
            ):

                answer, sources = answer_question(
                    question,
                    top_k,
                    mode
                )

            st.write(answer)

            if sources:

                st.divider()

                st.subheader(
                    "📚 Sources"
                )

                shown_sources = set()

                for source in sources:

                    source_name = source["metadata"]["source"]
                    page = source["metadata"]["page"]

                    key = (
                        source_name,
                        page
                    )

                    if key not in shown_sources:

                        shown_sources.add(key)

                        st.write(
                            f"📄 **{source_name}** "
                            f"— Page **{page}**"
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# ============================================================
# AI STUDY TOOLS
# ============================================================

st.divider()

st.header("🧠 AI Study Tools")

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("📝 Summary")

    if st.button(
        "Generate Summary",
        use_container_width=True
    ):

        with st.spinner(
            "Generating summary..."
        ):

            summary = summarize_document()

        st.write(summary)


with col2:

    st.subheader("🎓 MCQs")

    if st.button(
        "Generate MCQs",
        use_container_width=True
    ):

        with st.spinner(
            "Generating MCQs..."
        ):

            mcqs = generate_mcqs()

        st.write(mcqs)


with col3:

    st.subheader("🃏 Flashcards")

    if st.button(
        "Generate Flashcards",
        use_container_width=True
    ):

        with st.spinner(
            "Creating flashcards..."
        ):

            flashcards = generate_flashcards()

        st.write(flashcards)


# ============================================================
# SAMPLE QUESTIONS
# ============================================================

st.divider()

st.subheader("💡 Sample Questions")

sample_questions = [
    "What are the main concepts in the document?",
    "Explain the most important topic.",
    "What are the key advantages?",
    "Summarize the important points.",
    "What applications are mentioned?"
]

for sample in sample_questions:

    st.write(
        f"• {sample}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Python • Streamlit • ChromaDB • RAG • "
    "Gemini • AI Study Tools"
)