import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownTextSplitter

#RecursiveCharTxtSplitter 
txt_loader = TextLoader(file_path="data/simple text.txt", encoding = "utf-8")
txt_doc = txt_loader.load()
recursive_split = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 5)
txtresult = recursive_split.split_text(txt_doc[0].page_content)

print(txtresult)
print(len(txtresult))
'''The result is beautiful. Just so comforting to the eyes to see something like this'''


'''Markdown Splitting experimentation'''
# md_loader = TextLoader(file_path="data/the-humanizer.md")
# md_doc = md_loader.load()

# md_splitter = MarkdownTextSplitter(chunk_size = 100, chunk_overlap=0)
# mdresult = md_splitter.split_text(md_doc[0].page_content)

# m_splitter = RecursiveCharacterTextSplitter.from_language(
#     language = Language.MARKDOWN,
#     chunk_size=100, chunk_overlap=0
#     )
# mresult = m_splitter.split_text(md_doc[0].page_content)

# print(len(mdresult))
# print((mresult))

'''Can also try for python code, use Language.PYTHON in that case'''