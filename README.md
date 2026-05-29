# RAG Chatbot

A conversational AI chatbot that answers questions from your own PDF documents. Upload any PDF, ask questions in natural language, and get accurate answers with source citations — powered entirely by free AI models.

---

## What is this?

Most AI chatbots only know what they were trained on. This chatbot is different — it reads **your documents** and answers questions based on them. Upload a textbook, manual, research paper, or any PDF and have a conversation with it.

It uses **Retrieval Augmented Generation (RAG)** — a technique where relevant sections of your document are retrieved and passed to an AI model to generate accurate, grounded answers.

---

## Features

- Upload one or multiple PDF files directly from the browser
- Ask questions in plain English and get accurate answers
- Remembers conversation context for natural follow-up questions
- Answers stream word by word in real time
- Every answer shows which page and document it came from
- Customizable system prompt to change how the bot behaves
- Adjustable retrieval settings for better accuracy
- Completely free to run — no paid APIs required

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| LLM | Groq (Llama 3.1 8B Instant) |
| Embeddings | HuggingFace (BAAI/bge-small-en-v1.5) |
| Vector Database | ChromaDB |
| UI | Streamlit |
| Document Loader | PyPDF |

---

## How It Works

```
You upload a PDF
      ↓
Document split into small chunks
      ↓
Chunks converted into embeddings (numbers representing meaning)
      ↓
Embeddings stored in a local vector database
      ↓
You ask a question
      ↓
Most relevant chunks retrieved from database
      ↓
AI reads those chunks and generates an answer
      ↓
Answer streamed back with source citations
```

---

## Local Setup

### Prerequisites
- Python 3.9 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/Roopdev013/rag-chatbot.git
cd rag-chatbot

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root folder:

```
GROQ_API_KEY=gsk_your-key-here
```

### Add your PDF

Copy any PDF into the `data/` folder:

```bash
cp yourfile.pdf data/
```

### Ingest your documents

```bash
python src/ingest.py
```

### Run the chatbot

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Project Structure

```
rag-chatbot/
├── .env                  ← API keys (never commit!)
├── .gitignore
├── app.py                ← Main Streamlit UI
├── requirements.txt      ← Python dependencies
├── evaluate.py           ← Evaluation script
├── data/                 ← Your PDF documents go here
└── src/
    ├── ingest.py         ← PDF loading and chunking
    ├── retriever.py      ← Retrieval pipeline
    └── chatbot.py        ← Conversational chain
```

---

## Deployment

### Streamlit Cloud (Free)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app** → select this repo → set main file to `app.py`
5. Under **Advanced settings → Secrets** add:

```toml
GROQ_API_KEY = "gsk_your-key-here"
```

6. Click **Deploy**

---

## Customization

From the sidebar in the UI you can:

- Upload new PDF documents at any time
- Edit the system prompt to change how the bot responds
- Adjust how many document chunks to retrieve per question
- Clear the conversation history

---

## Evaluation

Test how accurately the chatbot answers questions:

```bash
python evaluate.py
```

This checks answer accuracy, retrieval quality, and whether the bot correctly says "I don't know" for questions outside the document.

---

## License

MIT License — free to use, modify, and distribute.
