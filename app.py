from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "LangChain is a framework for building LLM applications.",
    "RAG combines retrieval with generation for better answers.",
    "Hugging Face hosts open source models and datasets.",
    "FAISS is a library for efficient similarity search."
]
splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
chunks = splitter.create_documents(docs)

print("Building vector store...")
vectorstore = FAISS.from_documents(chunks, embeddings)

queries = ["What is LangChain?", "What is RAG?", "What is FAISS?"]
for q in queries:
    results = vectorstore.similarity_search(q, k=1)
    print(f"\nQuery: {q}")
    print(f"Retrieved: {results[0].page_content}")

print("\nRAG retrieval demo complete.")
