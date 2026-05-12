from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct:featherless-ai",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_KEY"),

)
model=ChatHuggingFace(llm=llm)

print((model.invoke("""
    You are a pop culture expert. What is JoJo's Bizarre Adventures? 
    If u do not know, do not give incorrect info.    
    Just say you dont know anymore
    """)).content)

# gives a very hallucinating answer, highly unrelated elements are there.
# asking it not to hallucinate gives a very honest feedback.