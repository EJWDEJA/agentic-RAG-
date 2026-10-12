from langchain_community.document_loaders import DirectoryLoader, TextLoader
from functools import partial


def load_txt(folder_path):
    """
    读取指定文件夹中的所有txt文件
    """

    loader = DirectoryLoader(
        folder_path,
        glob="*.txt",
        loader_cls=partial(TextLoader, encoding="utf-8")
    )

    documents = loader.load()

    print(f"读取到 {len(documents)} 个TXT文件")

    return documents
