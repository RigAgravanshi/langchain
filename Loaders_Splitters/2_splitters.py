import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# load text document and split
# load pdf doc and split coontent of its first page

txt_loader = TextLoader(file_path="data/simple text.txt", encoding = "utf-8")
txt_doc = txt_loader.load()
txt_splitter = CharacterTextSplitter(separator='', chunk_size = 100, chunk_overlap=10)
txtresult = txt_splitter.split_text(txt_doc[0].page_content)

pdf_loader = PyMuPDFLoader(file_path="data/Automobile Engineering LAK.pdf")
pdf_doc = pdf_loader.load()
pdf_splitter = CharacterTextSplitter(separator='', chunk_size = 100, chunk_overlap=20)
pdfresult = pdf_splitter.split_documents(pdf_doc)

print(pdfresult[1].page_content, "\n\n----------------------\n\n", txtresult)
#print(txtresult, "\n----------------------\n")