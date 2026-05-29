from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
    search_kwargs={"k": 5}
)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

# Fixed test cases with correct expected keywords
test_cases = [
    {
        "question": "What is cyclomatic complexity?",
        "expected": "cyclomatic complexity is a software metric"
    },
    {
        "question": "How is cyclomatic complexity calculated?",
        "expected": "edges and nodes"
    },
    {
        "question": "What are Halstead metrics?",
        "expected": "halstead"
    },
    {
        "question": "What is the capital of France?",
        "expected": "i dont have enough information"
    }
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer using context below.
If answer not in context say exactly: I dont have enough information.

Context:
{context}"""),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

print("=" * 60)
print("RAG CHATBOT EVALUATION REPORT")
print("=" * 60)

passed = 0
failed = 0

for i, test in enumerate(test_cases):
    question = test["question"]
    expected_keyword = test["expected"].lower()

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    answer_lower = answer.lower()
    is_correct = expected_keyword in answer_lower

    status = "PASS" if is_correct else "FAIL"
    if is_correct:
        passed += 1
    else:
        failed += 1

    print(f"\nTest {i+1}: {status}")
    print(f"  Question : {question}")
    print(f"  Expected : contains '{expected_keyword}'")
    print(f"  Answer   : {answer[:150]}...")
    print(f"  Chunks   : {len(docs)} retrieved")

print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print(f"SCORE: {round(passed/len(test_cases)*100)}%")
print("=" * 60)