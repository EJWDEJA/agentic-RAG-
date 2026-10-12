from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader


def load_docx(folder_path):
    """
    读取指定文件夹中的所有 Word 文件
    """

    loader = DirectoryLoader(
        folder_path,
        glob="*.docx",
        loader_cls=Docx2txtLoader
    )

    documents = loader.load()

    print(f"读取到 {len(documents)} 个Word文件")

    return documents
