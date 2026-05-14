# All this is without using with_structured_output(some models dont use it)
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=os.getenv("GOOGLE_API_KEY"))

# 1. StrOutputParser has the benefit of being used by chains seamlessly
# Task: Ask to generate a summary on a topic. Send that back, get a 5 pointer summary.

# template1 = PromptTemplate(
#     template = "Write a brief 100 worded report on {topic}",
#     input_variables=['topic']
# )
# template2 = PromptTemplate(
#     template = "Generate a 5 point summary on the given report: {text}",
#     input_variables=['text']
# )
# parser = StrOutputParser()
# chain = template1 | model | parser | template2 | model | parser
# result = chain.invoke({'topic':'The Rig Veda'})
# print(result)

# 2. Pydantic o/p parser, like before, also helps in validation

class Charact(BaseModel):
    name: str = Field(description = "Name of the character")
    skill: str = Field(description = "The character's ability")
    popularity_score: Optional[int] = Field(le=100, description='A popularity score')
    
parser = PydanticOutputParser(pydantic_object=Charact)

template = PromptTemplate(
    template = "Generate name, skill and arbitrary popularity score of a random mythological character of the country {country} in the following format: {format}",
    input_variables = ['country'],
    partial_variables= {'format': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'country':'Italy'})
print(result)