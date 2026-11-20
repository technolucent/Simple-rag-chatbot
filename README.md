# Simple RAG Chatbot

Interactive Retrieval-Augmented Generation demo using LangChain, Hugging Face embeddings, and a local LLM.

## Features
- Loads documents and builds a FAISS vector store
- Retrieves relevant context for user questions
- Generates answers with a local DistilGPT-2 model
- Interactive CLI chat

## Setup
```bash
pip install -r requirements.txt

## How to run
python app.py

Type your questions. Type exit to quit.
Note: First run downloads models (~400MB total).

Requirements

Python 3.10+
See requirements.txt
