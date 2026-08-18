# 📚 DocuMind AI — Multi-PDF RAG Chatbot

DocuMind AI is an AI-powered Multi-PDF Question Answering Chatbot built using  Retrieval-Augmented Generation (RAG) . It allows users to work with multiple PDF documents and ask natural-language questions based specifically on their content.
Instead of sending entire documents directly to an LLM, DocuMind AI retrieves the most relevant document chunks and provides them as context to the language model. This helps generate more relevant and document-grounded answers while reducing the chances of hallucination.

## 🚀 Key Features

* 📄 Multi-PDF Support — Load and process multiple PDF documents.
* ✂️ Intelligent Text Chunking  — Uses RecursiveCharacterTextSplitter to divide documents into manageable chunks.
* 🧠 Semantic Embeddings — Generates embeddings using `entence-transformers/all-MiniLM-L6-v2.
* 🔎 Vector Search  — Uses FAISS for efficient similarity-based document retrieval.
* 🤖 LLM-powered Answers — Uses Groq's GPT-OSS-20B  model for response generation.
* 🔗 LangChain RAG Pipeline — Combines retrieval, prompt construction, LLM generation, and output parsing into a chain.
* 📑 Source-Aware Responses — Provides PDF name and page number in the retrieved context whenever available.
* 🚫 Reduced Hallucination — The model is instructed to answer only from the provided PDF context.
* 🔐 Environment Variables  — API credentials are managed using .env.

## 🏗️ RAG Architecture

                    ┌─────────────────┐
                    │   PDF Documents │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  PyPDFLoader    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Text Splitting  │
                    │ Chunk: 1500     │
                    │ Overlap: 300    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ HuggingFace     │
                    │ Embeddings      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ FAISS Vector DB │
                    └────────┬────────┘
                             ↓
User Question ─────→ Similarity Retrieval
                             ↓
                    ┌─────────────────┐
                    │ Relevant PDF    │
                    │ Context         │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ PromptTemplate  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Groq LLM        │
                    │ GPT-OSS-20B     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Final Answer    │
                    └─────────────────┘
🛠️ Tech Stack

* **Python**
* **LangChain**
* **Groq**
* **Hugging Face Sentence Transformers**
* **FAISS**
* **PyPDF**
* **python-dotenv**

## 🔄 RAG Pipeline

The project follows the standard RAG workflow:

1. **Document Ingestion — Load multiple PDF files using `PyPDFLoader`.
2. **Text Splitting — Split PDF content into chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding Generation — Convert chunks into vector embeddings using Hugging Face.
4. **Vector Storage — Store embeddings in a FAISS vector database.
5. **Retrieval — Retrieve the most relevant chunks using similarity search.
6. **Augmentation — Format retrieved chunks with PDF name and page information.
7. **Prompting — Pass the retrieved context and user question to the prompt.
8. **Generation — Generate the final answer using the Groq LLM.
9. **Output Parsing — Convert the model response into a clean string using `StrOutputParser`.

## 📂 Example Documents

The project can process multiple PDFs such as:

```text
RGPV_ML_Detailed_Answers.pdf
RGPV_Static_Dynamic_Interconnection_Networks_Notes.pdf
story.pdf
```

You can also modify the project to load all PDFs from a dedicated folder using `PyPDFDirectoryLoader`.

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repository-url>
cd <your-project-folder>

pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Then run the application:

```bash
python your_file.py
```

## 💡 Example

**Question:**

```text
What is Linear Regression?
```

**DocuMind AI:**

The chatbot retrieves the most relevant content from the indexed PDFs and generates an answer based on that retrieved context.

## 🎯 Learning Outcomes

Through this project, I implemented and explored:

* Retrieval-Augmented Generation (RAG)
* Multi-document processing
* Document chunking
* Semantic embeddings
* Vector databases
* Similarity search
* LangChain Runnable pipelines
* Prompt engineering
* LLM integration
* Source-aware document retrieval
* Environment variable management

## 🔮 Future Improvements

* [ ] Add a Streamlit web interface
* [ ] Support PDF upload directly from the UI
* [ ] Add conversational memory
* [ ] Display retrieved sources separately
* [ ] Add chat history
* [ ] Support more document formats
* [ ] Add hybrid search
* [ ] Add metadata filtering
* [ ] Deploy the chatbot online

## 👨‍💻 Author

**Abhinay Sah**

Built as a practical **Generative AI / RAG project** to understand how modern document-question-answering systems work using LangChain, vector databases, embeddings, and LLMs.
