# Simple RAG Chatbot
Building a simple RAG chatbot with LangChain + Hugging Face

# Simple RAG Chatbot
Retrieval demo using LangChain + Hugging Face embeddings + FAISS.

## How to run
```bash
pip install -r requirements.txt
python app.py

First run downloads models (~100MB).

Loading embeddings...
Building vector store...

Query: What is LangChain?
Retrieved: LangChain is a framework for building LLM applications.
...
RAG retrieval demo complete.


## Full RAG
Set token: export HUGGINGFACEHUB_API_TOKEN=your_hf_token
Then run again for generation.

__pycache__/
*.pyc
.env
