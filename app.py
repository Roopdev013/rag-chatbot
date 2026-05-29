import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()
import os
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="RAG Chatbot", page_icon="Robot", layout="centered")
st.title("RAG Chatbot")
st.caption("Upload PDFs and ask me anything!")

@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant")
    return embeddings, llm

embeddings, llm = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        new_files = [fi.name for fi in uploaded_files if fi.name not in st.session_state.uploaded_files]
        if new_files:
            with st.spinner("Processing PDFs..."):
                all_chunks = []
                splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in new_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path)
                        pages = loader.load()
                        chunks = splitter.split_documents(pages)
                        for chunk in chunks:
                            chunk.metadata["source"] = uploaded_file.name
                        all_chunks.extend(chunks)
                        os.unlink(tmp_path)
                        st.session_state.uploaded_files.append(uploaded_file.name)
                if all_chunks:
                    Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory="chroma_db")
                    st.success(f"Added {len(new_files)} new PDF(s)!")
                else:
                    st.error("Could not extract text. Try a different PDF.")

    if st.session_state.uploaded_files:
        st.subheader("Loaded Documents")
        for fname in st.session_state.uploaded_files:
            st.write("OK " + fname)

    st.divider()
    st.subheader("Bot Settings")
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant. Answer using the context provided. If the answer is not in the context say: I dont have enough information.",
        height=120
    )
    num_chunks = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        history_text = ""
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                history_text += "Human: " + msg.content + "\n"
            elif isinstance(msg, AIMessage):
                history_text += "Assistant: " + msg.content + "\n"

        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nContext:\n{context}\n\nChat History:\n{chat_history}"),
            ("human", "{question}")
        ])

        chain = prompt | llm | StrOutputParser()

        response = st.write_stream(
            chain.stream({
                "context": context,
                "chat_history": history_text,
                "question": question
            })
        )

        with st.expander("Sources used"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}** - Page {doc.metadata.get('page', 0)+1} of {doc.metadata.get('source', 'document')}")
                st.caption(doc.page_content[:200] + "...")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response))