from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


def load_pdf(folder_path):
    """
    读取指定文件夹中的所有PDF文件
    """

    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    print(f"读取到 {len(documents)} 个PDF页面")

    return documents
