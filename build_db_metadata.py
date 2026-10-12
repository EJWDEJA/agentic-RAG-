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

documents = []    #先创一个list用于存储读取到的数据

for loader in LOADERS:
    documents.extend(loader(KNOWLEDGE_DIR))



print(f"成功读取 {len(documents)} 个文档")

# 2. 切分文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


docs = text_splitter.split_documents(documents)       #docs是一个列表列表里面每个元素是doc由Document(page_context="xxx",matedate={"source":"xx","chunk id"=0})组成


for i, doc in enumerate(docs):

    doc.metadata["chunk_id"] = i    #直接修改doc里面的matedate



# 保存Document给BM25使用

import pickle


with open(
    "documents.pkl",     #打开文件名
    "wb"                 #打开模式，全称write binary 二级制写入模式
) as f:        #f对于打开来的文件的简单命名方便调用

    pickle.dump(    #dump转储，用于把docs里面的list列表转为二进制储存在打开的文件里面
        docs,     #待存储对象
        f         #documents.pkl的自定义简称
    )


print(f"共切分为 {len(docs)} 个文本块")

# 3. Embedding 模型
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# 4. 创建 Chroma 数据库
db = Chroma.from_documents(
    documents=docs,      #处理后的list docs
    embedding=embeddings,    
    persist_directory=CHROMA_DIRR     #向量库名字
)

print("知识库创建成功！")
