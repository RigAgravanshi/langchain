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

pdf_loader = PyMuPDFLoader(file_path="data/Profile.pdf")
pdf_doc = pdf_loader.load()
pdf_splitter  = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 20)
pdf_chunks = pdf_splitter.split_documents(pdf_doc)

# models = BAAI/bge-base-en-v1.5 | Qwen/Qwen3-Embedding-0.6B
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"token": os.getenv("HF_KEY")})
vector_store = Chroma.from_documents(documents=pdf_chunks, embedding=embeddings,persist_directory="data/chroma_vector_store")
results = vector_store.similarity_search_with_relevance_scores(query='Education or academic background', k=2)

for doc, score in results:
	print(doc.page_content, "\n\n", score)
# Ok, some okay-ish result has been given. Ofc, this is not a retriever so I dont expect it to. Do SEMANTIC CHUNKING for better results
# On that note, i dont have any simple text based pdfs on my sys :(
# Big pdfs or ones with extreme formatting cause great loading and searching time(18+ minutes in a 21 pg formatttyyy pdf)
print(f"Time taken: {(time.time() - start)} seconds")