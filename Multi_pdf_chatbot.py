# ------------------------------------------------------------ DucuMind AI - Multi Pdf Chatbot ---------------------------------------------------------------------------


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# load pdf 

documents =[]

for pdf_path in ["RGPV_ML_Detailed_Answers.pdf","RGPV_Static_Dynamic_Interconnection_Networks_Notes.pdf","story.pdf"]:
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())

# print(len(documents))  # shows the total pages of all three pdfs

# second way to load  : if the all pdf in pdfs folder
# loader = PyPDFDirectoryLoader("pdfs/")
# documents = loader.load()

# print(type(documents))  # should be list 




# text splitting

splitter = RecursiveCharacterTextSplitter(chunk_size = 1500 , chunk_overlap =300)
chunks = splitter.split_documents(documents)  # use split bcz getting document not plain text
# print(chunks)

# create_documents() → expects a list of strings
# split_documents() → expects a list of Document objects



# embedding and vector store
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding)


# retrieval 
retriever = vector_store.as_retriever(search_type = "similarity" , search_kwargs ={"k":8})


# augmentation

# prompt 
model = ChatGroq(model="openai/gpt-oss-20b")
prompt = PromptTemplate(
    template="""
You are a helpful PDF question-answering assistant.

Answer the question ONLY using the provided PDF context.

Rules:
1. Do not use outside knowledge.
2. If the answer is present in the context, answer it clearly.
3. If the answer is not present, say:
   "I don't know based on the provided documents."
4. Mention the PDF name and page number when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""",
    input_variables=["context", "question"]
)

# format docs 

def format_docs(documents):

    formatted_docs = []

    for doc in documents:

        pdf_name = doc.metadata.get(
            "pdf_name",
            doc.metadata.get("source", "Unknown")
        )

        page = doc.metadata.get("page", 0) + 1

        formatted_docs.append(
            f"""
PDF: {pdf_name}
PAGE: {page}

CONTENT:
{doc.page_content}
"""
        )

    return "\n\n--------------------\n\n".join(formatted_docs)

# chain 

chain = (RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()},
) | prompt | model | StrOutputParser())


# generation 
response = chain.invoke("what is Linear Regression")
print(response)
