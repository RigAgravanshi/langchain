import os
import time
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
start = time.time()

pdf_loader = PyMuPDFLoader(file_path="data/Automobile Engineering LAK.pdf")
pdf_doc = pdf_loader.load()
pdf_splitter  = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
pdf_chunks = pdf_splitter.split_documents(pdf_doc)

# models =  | Qwen/Qwen3-Embedding-0.6B | sentence-transformers/all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"token": os.getenv("HF_KEY")})
vector_store = Chroma.from_documents(documents=pdf_chunks, embedding=embeddings,persist_directory="data/chroma_vecstore_retriever")

# using VecStore as retriever
retriever = vector_store.as_retriever(
	search_type='mmr',					#MaxMarginalRetriever-->1st gives the most relevant doc; then gives 2nd most-relevant & LEAST SIMILAR to 1st one
	search_kwargs={'k':3}
	)
query = 'List the types of wheel alignment'
results = retriever.invoke(query)

for i, doc in enumerate(results):
	print("/n/n-----Result-----/n/n")
	print(doc.page_content)
 
print(f"Time taken: {(time.time() - start)} seconds")


# Multi-Query-Retriever: removes ambiguity from the query via an LLM by sending multiple rephrased-queries
# Contextual-Compression-Retriever: doesnt just return the entire doc--> Compressor(LLM) returns only relevant part from the doc/chunk
# Wikipedia, vectorstore, websearch Retrievers
