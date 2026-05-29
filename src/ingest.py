from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1 - Load PDF
pdf_path = "data/Software Engineering Test Solutions (5).pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Total pages loaded: {len(pages)}")

# Step 2 - Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(pages)
print(f"Total chunks created: {len(chunks)}")

# Step 3 - Load embedding model
print("\nLoading embedding model (first time downloads ~90MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
print("Embedding model loaded!")

# Step 4 - Store in ChromaDB
print("\nStoring chunks in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")
print("Vector store saved to chroma_db/ folder")