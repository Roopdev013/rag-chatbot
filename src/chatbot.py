from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using the context below.
If the answer is not in the context, say 'I dont have enough information to answer this.'

Context:
{context}

Chat History:
{chat_history}"""),
    ("human", "{question}")
])

def format_history(history):
    formatted = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            formatted += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted += f"Assistant: {msg.content}\n"
    return formatted

print("Chatbot ready! Type exit to quit.\n")
chat_history = []

while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break

    docs = retriever.invoke(question)
    context = format_docs(docs)
    history_text = format_history(chat_history)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "chat_history": history_text,
        "question": question
    })

    print(f"\nBot: {answer}\n")

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))