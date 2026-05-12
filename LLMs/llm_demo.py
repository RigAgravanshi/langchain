from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
# temperature param works like random_state. Temp=0, same o/p each time. 

print((llm.invoke("Who are we? Answer in 10 words")))