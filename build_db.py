from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from loaders.txt_loader import load_txt
from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from loaders.md_loader import load_md
from config import *
from loaders.excel_loader import load_excel

LOADERS = [
    load_txt,
    load_pdf,
    load_docx,
    load_md,
    load_excel
]

documents = []

for loader in LOADERS:
    documents.extend(loader(KNOWLEDGE_DIR))



print(f"成功读取 {len(documents)} 个文档")

# 2. 切分文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

docs = text_splitter.split_documents(documents)

print(f"共切分为 {len(docs)} 个文本块")

# 3. Embedding 模型
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# 4. 创建 Chroma 数据库
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print("知识库创建成功！")
