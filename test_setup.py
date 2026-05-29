from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
   model="llama-3.1-8b-instant"
)

response = llm.invoke("Say bhadiya in one word.")
print("LLM works:", response.content)
print("chal be soo ja ")