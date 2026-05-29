from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1 - Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Step 2 - Load existing ChromaDB vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Step 3 - Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Step 4 - Load Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

# Step 5 - Create custom prompt
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context 
from the document to answer the question accurately.
If the answer is not in the context, say 
"I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:
""")

# Step 6 - Create chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 7 - Test with a question
query = "What is cyclomatic complexity?"
print(f"Question: {query}\n")
print("Answer:")
print(chain.invoke(query))