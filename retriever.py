from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


CHROMA_DIRR="./chroma_db_metadata"


def get_retriever():

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding"
    )


    db = Chroma(
        persist_directory=CHROMA_DIRR,
        embedding_function=embeddings
    )


    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":3,
            "fetch_k":10
        }
    )

    return retriever
