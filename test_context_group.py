from langchain_core.documents import Document    #导入类

from context_group import create_context_group


core_doc = Document(        #这是一个没有用dataclass开发的类但是使用方法和dataclass定义的类一样
    page_content="Chunk 40：学校计算机学院成立于2001年。",
    metadata={
        "chunk_id": 40
    }
)


expanded_docs = [

    Document(
        page_content="Chunk 39：前文介绍学校的发展历史。",
        metadata={
            "chunk_id": 39
        }
    ),

    core_doc,

    Document(
        page_content="Chunk 41：计算机学院随后开始扩大招生。",
        metadata={
            "chunk_id": 41
        }
    )
]


group = create_context_group(         #这里面的group就是对象 
    core_doc,
    expanded_docs,
    score=0.92
)


print("\n===== Context Group =====")

print("Score:", group.score  )    #这里的group.score就是用来实现类的价值：即用对象.字段名快速调取字段值

print("Documents:")

for doc in group.docs:      

    print(
        "chunk_id:",
        doc.metadata.get("chunk_id")
    )
