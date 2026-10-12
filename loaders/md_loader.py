from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader


def load_md(folder_path):
    """
    读取指定文件夹中的所有 Markdown 文件
    """

    loader = DirectoryLoader(
        folder_path,
        glob="*.md",
        loader_cls=UnstructuredMarkdownLoader
    )

    documents = loader.load()

    print(f"读取到 {len(documents)} 个Markdown文件")

    return documents
