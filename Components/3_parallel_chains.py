from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=os.getenv("GOOGLE_API_KEY"))
model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# We Generate summary of a topic. Then parallely generate a Hinglish summary of it & quiz q on it. Then combine it.
prompt1 = PromptTemplate(
    template = "Write a brief 100 worded report on {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = "Generate a 5 point summary on the given report in Hinglish langauge:\n {text}",
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template = "Generate 5 quiz questions on the given report:\n {text}",
    input_variables=['text']
)
prompt4 = PromptTemplate(
    template = "Combine the 5 point summary and 5 quiz questions into one single output\n of {notes} and {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser | prompt2 | model1 | parser,  # RunnableSequence(prompt1,model1,parser,prompt2.......)
    'quiz': prompt1 | model1 | parser | prompt3 | model2 | parser    # RunnableSequence(prompt1,model1..........)
})

merged_chain = prompt4 | model1 | parser
chain = parallel_chain | merged_chain

result = chain.invoke({'topic':'Attention Mechanism'})
print(result)