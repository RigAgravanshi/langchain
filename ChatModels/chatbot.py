from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

chat_history = []          # for context of prev chats
# chat_history = [SystemMessage(content = "You are a wise sage.")]

while True:
    input_message = input("You: ")
    #chat_history.append(HumanMessage(content=input_message))
    chat_history.append(input_message)
    if input_message == 'exit':
        break
    result = llm.invoke(chat_history)
    #chat_history.append(AIMessage(content=result.content))
    chat_history.append(result.content)
    print("AI: ",result.content)
print(chat_history)

# problem here is: When chat_history grows too large, model doesnt know which message was sent by whom.
# So we use SystemMessage, HumanMessage & AIMessage