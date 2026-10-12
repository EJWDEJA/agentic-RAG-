from retrieval_chain_group import retrieve_context


question = "计算机科学与技术学院有哪些教师？"


result = retrieve_context(
    question
)


print("\n===== FINAL CONTEXT =====")

print(
    result["context"]
)


print("\n===== SOURCES =====")

for source in result["sources"]:

    print(
        source
    )