from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=os.getenv("GOOGLE_API_KEY"))

class Review(BaseModel):
    sentiment: Literal['positive', 'negative']=Field(description="Sentiment of the given review as positive or negative")

parser = PydanticOutputParser(pydantic_object=Review)
parser2 = StrOutputParser()

prompt = PromptTemplate(
    template = "Classify sentiment of the given review into positive or negative \n{review} in the following format:\n{format}",
    input_variables = ['review'],
    partial_variables= {'format': parser.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template = "Write an apt response to this positive review to the customer:\n {feedback}",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template = "Write an apt response to this negative review to the customer:\n {feedback}",
    input_variables=['feedback']
)

classifier_chain = prompt | model | parser

conditional_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2 | model | parser2),
    (lambda x:x.sentiment=='negative', prompt3 | model | parser2),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | conditional_chain
print(chain.invoke({'review':"What is this app? There's just sponsored content, i cannot find anything that i search for."}))