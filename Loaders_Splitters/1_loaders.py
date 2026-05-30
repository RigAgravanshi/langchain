# https://docs.langchain.com/oss/python/integrations/document_loaders
'''Find different kinds of Loaders from the above website
using langchain_community showx DeprecationWarning: being sunset and no longer maintined (RISKY)
there are loaders for arXiv, various webpages, pdfs(use pypdf for small tasks) and so on 
Whereas, docling takes IMPOSSIBLE AMOUNTS of time incase you pdf/file has some OCR based things,
images, tables and so on'''

import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import TextLoader, ArxivLoader, PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=os.getenv("GOOGLE_API_KEY"))
parser = StrOutputParser()

# We gonna ask whether there is any piece of similarity between Sherlock Holmes and Automobile Engineering pdf
loader1 = TextLoader("data/Sherlock_texts_trimmed.txt", encoding="utf-8")
docs = loader1.load()

loader2 = PyMuPDFLoader("data/Automobile Engineering LAK.pdf")
pdf_doc = loader2.load()

prompt = PromptTemplate(
    template = "Write at least 3 points of similarity or correlation in the contents of {automobile} and {sherlock}. If there is nothing worth mentioning or surprising enough, just give a 5 line story involving the main themes in the 2 content",
    input_variables=['automobile', 'sherlock']
)

chain = prompt | model | parser

result = chain.invoke({
    'automobile' : pdf_doc[0].page_content,     # First page only
    'sherlock': docs[0].page_content[ :10000]
    })
print(result)

# interesting result. But it gave both answers: Use RunnableConditional or improve prompting