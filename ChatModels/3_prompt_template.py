# dynamic prompting
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model_name = "Qwen/Qwen2.5-7B-Instruct:featherless-ai"
llm = HuggingFaceEndpoint(
    repo_id=model_name,
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_KEY")
)
model=ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = '''
    Please summarize the Season "{season_no}" of {name} in the following manner:
    Spoilers to be given: {yes_no}
    Length of explanation: {length_input}
    Include IMDB Ratings also.
    If you do not know, do not spout fake nonsense. Just accept that you dont know, DO NOT HALLUCINATE.''',
    input_variables=['season_no', 'name', 'yes_no', 'length_input']
)

prompt = template.invoke({
    'season_no': "4",
    'name': "Sherlock(starring Benedict Cumberbatch)",
    'yes_no': "yes",
    'length_input': "120 words"
})

print(model.invoke(prompt).content)