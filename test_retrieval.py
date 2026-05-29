from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load existing vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Ask a question
query = "What is cyclomatic complexity?"
print(f"Question: {query}")
print(f"\nTop 3 relevant chunks found:\n")

results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"Result {i+1} (Page {doc.metadata['page']+1}):")
    print(doc.page_content[:300])
    print("-" * 50)