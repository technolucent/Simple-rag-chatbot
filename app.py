from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = ["LangChain is a framework for LLM apps.", "RAG combines retrieval with generation."]
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.create_documents(docs)

print("Building vector store...")
vectorstore = FAISS.from_documents(chunks, embeddings)

print("RAG skeleton ready. Add your LLM next.")

query = "What is LangChain?"
results = vectorstore.similarity_search(query, k=1)
print(f"Query: {query}")
print(f"Retrieved: {results[0].page_content}")
