from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "LangChain is a framework for building LLM applications.",
    "RAG combines retrieval with generation for better answers.",
    "Hugging Face hosts open source models and datasets.",
    "FAISS is a library for efficient similarity search.",
    "Python is widely used for AI and machine learning."
]
splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
chunks = splitter.create_documents(docs)

print("Building vector store...")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("Loading LLM...")
model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=pipe)

template = """Answer based on context.
Context: {context}
Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nAsk questions (type 'exit' to quit):")
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit", ""]:
        break
    answer = rag_chain.invoke(query)
    print(f"Bot: {answer}")

print("Done.")
