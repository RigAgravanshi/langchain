import os
import time
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
start = time.time()
id = 'YXfRsS8MzX4'
api = YouTubeTranscriptApi()

# transcript loading
try:
	transcript_list = api.fetch(video_id=id, languages=["en","hi"])
	txt = " ".join(s.text for s in transcript_list)
	print("Transcript loading Complete...!!/n/n")
	#print(txt[:5000])
except TranscriptsDisabled:
	print("Transcript aint available/n/n")
print(f"Time taken: {(time.time() - start)} seconds/n/n")


# chunking
''' TRY OUT SEMANTIC CHUNKING ONCE AFTER THE ENTIRE THING IS BUILT'''
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
chunk_list = splitter.split_text(txt)
#print(len(chunk_list))
print("Chunking process complete...!!/n/n")
print(f"Time elapsed: {(time.time() - start)} seconds/n/n")


# vector store for embeddings
'''models = BAAI/bge-base-en-v1.5 | Qwen/Qwen3-Embedding-0.6B '''
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"token": os.getenv("HF_KEY")})
vector_store = Chroma.from_documents(documents=chunk_list, embedding=embeddings, persist_directory="data/youtube_vector_store")
print("Vector Storage process complete...!!/n/n")


# initialize retriever
retriever = vector_store.as_retriever(
	search_type = 'similarity',					 
	search_kwargs = {'k' : 4}
	)
query = 'What is the origin and descent of Iranians as per the Rig Veda'
context = retriever.invoke(query)
print("Retrieval Process complete..!!/n/n")
print(f"Time elapsed: {(time.time() - start)} seconds/n/n")


# loading the LLM, Prompt and Output-Parser
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=os.getenv("GOOGLE_API_KEY"))
prompt = PromptTemplate(
    template = """
		You are a Helpful assistant. Answer the Question only from the provided Context.
		If context is insufficient, Just say so. DO NOT make up information that isnt mentioned in the context
		Context: {context}
		Question: {query}
		""",
    input_variables=['context', 'query']
)
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({'context':context, 'query':query})
print(result)

print(f"Total Time Taken: {round(time.time() - start)} seconds")