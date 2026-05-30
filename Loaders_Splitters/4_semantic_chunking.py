from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

txt_loader = TextLoader(file_path="data/mixed_topics.txt", encoding = "utf-8")
txt_doc = txt_loader.load()
semantic_split = SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation", breakpoint_threshold_amount=0.5)
txtresult = semantic_split.split_text(txt_doc[0].page_content)

print(txtresult)

"""Amazing chunking. Very relaxing to see this, try it yourself"""