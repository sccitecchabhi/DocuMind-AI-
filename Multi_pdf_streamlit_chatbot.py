import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI | Multi-PDF RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Header */
    .main-header {
        padding: 1.5rem 0 0.5rem 0;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .info-card {
        background: white;
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .feature-card {
        background: white;
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        height: 100%;
    }

    .feature-title {
        font-weight: 650;
        color: #111827;
        font-size: 1rem;
    }

    .feature-text {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* Source cards */
    .source-card {
        background: #f8fafc;
        border-left: 4px solid #4f46e5;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }

    .source-title {
        font-weight: 600;
        color: #1f2937;
    }

    .source-page {
        color: #6b7280;
        font-size: 0.85rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-header">

<div class="main-title">
📚 DocuMind AI
</div>

<div class="subtitle">
Intelligent Multi-PDF Question Answering powered by RAG + FAISS + Groq
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📄 Documents")

    st.caption(
        "Upload one or multiple PDF files and ask questions "
        "directly from their content."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once."
    )

    st.divider()

    st.markdown("### ⚙️ RAG Configuration")

    chunk_size = st.slider(
        "Chunk size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50
    )

    top_k = st.slider(
        "Retrieved documents",
        min_value=2,
        max_value=8,
        value=4
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_documents(uploaded_files):

    documents = []

    # -----------------------------------------------------
    # Load demo PDFs from pdfs folder
    # -----------------------------------------------------

    demo_folder = "pdfs"

    if os.path.exists(demo_folder):

        for filename in os.listdir(demo_folder):

            if filename.lower().endswith(".pdf"):

                pdf_path = os.path.join(
                    demo_folder,
                    filename
                )

                loader = PyPDFLoader(pdf_path)

                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = filename

                documents.extend(docs)

    # -----------------------------------------------------
    # Load uploaded PDFs
    # -----------------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(
                    uploaded_file.getbuffer()
                )

                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            documents.extend(docs)

            os.unlink(tmp_path)

    return documents


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(
    documents,
    chunk_size,
    chunk_overlap
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embedding
    )

    return vector_store


# =========================================================
# PROCESS DOCUMENT BUTTON
# =========================================================

if st.sidebar.button(
    "🚀 Process Documents",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Processing your documents..."
    ):

        documents = load_documents(
            uploaded_files
        )

        if not documents:

            st.error(
                "No PDF documents found. "
                "Please upload at least one PDF."
            )

        else:

            vector_store = create_vector_store(
                documents,
                chunk_size,
                chunk_overlap
            )

            st.session_state.vector_store = vector_store

            st.session_state.documents_ready = True

            files = sorted(
                set(
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                    for doc in documents
                )
            )

            st.session_state.processed_files = files

            st.session_state.messages = []

            st.success(
                f"Processed {len(files)} PDF file(s) successfully."
            )


# =========================================================
# SIDEBAR FILE STATUS
# =========================================================

if st.session_state.processed_files:

    st.sidebar.divider()

    st.sidebar.markdown(
        "### ✅ Processed PDFs"
    )

    for file in st.session_state.processed_files:

        st.sidebar.markdown(
            f"📄 `{file}`"
        )


# =========================================================
# WELCOME SCREEN
# =========================================================

if not st.session_state.documents_ready:

    st.markdown("""
    <div class="info-card">

    <h3>👋 Welcome to DocuMind AI</h3>

    <p style="color:#6b7280;">
    Upload your PDF documents from the sidebar, click
    <b>Process Documents</b>, and then ask questions about
    the information contained inside them.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">

        <div style="font-size:2rem;">📚</div>

        <div class="feature-title">
        Multi-PDF Support
        </div>

        <div class="feature-text">
        Upload and search across multiple PDF documents
        simultaneously.
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">

        <div style="font-size:2rem;">🔎</div>

        <div class="feature-title">
        Semantic Search
        </div>

        <div class="feature-text">
        FAISS retrieves the most relevant document chunks
        for every question.
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">

        <div style="font-size:2rem;">🤖</div>

        <div class="feature-title">
        AI Answers
        </div>

        <div class="feature-text">
        Groq-powered LLM generates answers using retrieved
        document context.
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.info(
        "💡 Tip: Your repository can contain demo PDFs "
        "inside the `pdfs/` folder. Users can also upload "
        "their own PDFs."
    )


# =========================================================
# CHAT FUNCTION
# =========================================================

def get_answer(question):

    vector_store = st.session_state.vector_store

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = PromptTemplate(
        template="""
You are DocuMind AI, a helpful document assistant.

Answer ONLY using the provided PDF context.

Rules:
1. Do not use outside knowledge.
2. If the answer cannot be found in the context,
   say "I don't know based on the provided documents."
3. Keep the answer clear and concise.
4. Never invent information.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=[
            "context",
            "question"
        ]
    )

    retrieved_docs = retriever.invoke(
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, retrieved_docs


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander(
                "📚 View Sources"
            ):

                for source in message["sources"]:

                    filename = source["filename"]

                    page = source["page"]

                    st.markdown(
                        f"""
                        <div class="source-card">

                        <div class="source-title">
                        📄 {filename}
                        </div>

                        <div class="source-page">
                        Page {page}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# =========================================================
# CHAT INPUT
# =========================================================

if st.session_state.documents_ready:

    question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if question:

        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        # Assistant response
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents..."
            ):

                try:

                    answer, retrieved_docs = get_answer(
                        question
                    )

                    st.markdown(answer)

                    # Collect sources
                    sources = []

                    seen = set()

                    for doc in retrieved_docs:

                        filename = doc.metadata.get(
                            "source",
                            "Unknown PDF"
                        )

                        page = doc.metadata.get(
                            "page",
                            0
                        )

                        page = page + 1

                        key = (
                            filename,
                            page
                        )

                        if key not in seen:

                            seen.add(key)

                            sources.append({
                                "filename": filename,
                                "page": page
                            })

                    # Show sources
                    if sources:

                        with st.expander(
                            "📚 View Sources"
                        ):

                            for source in sources:

                                st.markdown(
                                    f"""
                                    <div class="source-card">

                                    <div class="source-title">
                                    📄 {source["filename"]}
                                    </div>

                                    <div class="source-page">
                                    Page {source["page"]}
                                    </div>

                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:

                    st.error(
                        f"Something went wrong: {str(e)}"
                    )

else:

    st.chat_input(
        "Process your documents first...",
        disabled=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#9ca3af; font-size:0.85rem;">
        Built with LangChain • FAISS • Hugging Face • Groq • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)