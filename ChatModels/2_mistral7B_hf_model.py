from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

from ChatModels.chatbot import chat_history

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    task="conversational",
    huggingfacehub_api_token=os.getenv("HF_KEY"),
)
model = ChatHuggingFace(llm=llm)

chat_history=[]
chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an anime expert. Keep responses clear and engaging.",
        ),
        MessagesPlaceholder(variable_name=chat_history),       # For retaining past convo history
        (
            "human",
            "Summarize season {season_no} of JoJo's Bizarre Adventures.\n"
            "Include spoilers: {yes_no}\n"
            "Target length: {length_input}\n"
            "Style: anime speech tone and include IMDb rating if known.",
        ),
    ]
)

chain = chat_template | model

response = chain.invoke(
    {
        "season_no": "3",
        "yes_no": "No",
        "length_input": "120 words",
    }
)

print(response.content)