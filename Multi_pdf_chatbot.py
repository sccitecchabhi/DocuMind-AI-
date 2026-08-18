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

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap =200)
chunks = splitter.split_documents(documents)  # use split bcz getting document not plain text
# print(chunks)

# create_documents() → expects a list of strings
# split_documents() → expects a list of Document objects



# embedding and vector store
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding)


# retrieval 
retriever = vector_store.as_retriever(search_type = "similarity" , search_kwargs ={"k":4})


# augmentation

# prompt 
model = ChatGroq(model="openai/gpt-oss-20b")
prompt = PromptTemplate (
template= """ You are a helpful assistant.
      Answer ONLY from the provided pdf context.
      If the context is insufficient, just say you don't know.
    context: {context}
    question : {question}""",
    input_variables= ["context","question" ]
)


# format docs 
def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)  # converted all content into one string


# chain 

chain = (RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()},
) | prompt | model | StrOutputParser())


# generation 
response = chain.invoke("what is Linear Regression")
print(response)
