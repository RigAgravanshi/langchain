from typing import Literal, TypedDict, Annotated, Optional
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=os.getenv("GOOGLE_API_KEY"))

# 1. Annotated Typeddict
class Review1(TypedDict):            
    summary: Annotated[str, 'Summarise the given review']
    sentiment: Annotated[str, 'Analyse the sentiment of the review, in minimum words']
    name: Annotated[Optional[str], 'Mention name od the Reviewer']


# 2. Pydantic: offers type validation as well
class Review2(BaseModel):
    summary: str=Field(description="Summarise the given review in a brief, crisp manner")
    sentiment: Literal["pos","neg"] = Field(description="Analyse the sentiment of the review")
    #cgpa: float = Field(gt=0, lt=10, default=0, description='a value representing my Future')
    name: Optional[str] = Field(default=None, description='Give the name of Reviewer')


# 3. json schema
'''used when other languages also being used in the project'''


structured_model = model.with_structured_output(Review1)
text = """Books: Everyone has had a similar experience when reading a book. What happens is you read a book and you put it down and you rub your sleepy little eyes and you go,"Damn, that really was a book." I defy you to find someone who has not done this. If anyone is around when I finish a book, I personally like to say, "Damn, I could have spent all that time gambling at a bar instead of reading this book." But no one is ever around. I award books four stars for all the stuff in them. Keeps the mind sharp. Review by 'burialgoods'."""

response = structured_model.invoke(text)
print(response)