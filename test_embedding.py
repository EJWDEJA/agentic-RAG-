from langchain_community.embeddings import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="qwen3-embedding"
)


text = "人工智能在教育领域的应用"


print("开始生成 Embedding...")

vector = embeddings.embed_query(text)

print("Embedding 成功")
print("向量维度：", len(vector))
print("前 10 个值：", vector[:10])