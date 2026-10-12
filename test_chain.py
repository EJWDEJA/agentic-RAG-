from rag_chain import rag_chain
from memory import show_memory



while True:

    question = input("\n用户：")


    if question == "exit":
        break


    result = rag_chain.invoke(

        {
            "question": question
        },

        config={
            "configurable":
            {
                "session_id":"user001"
            }   #自定义用户AI在实际项目这个ID来自浏览器微信号随机生成等等由于构建store键值对存贮不同ID用户的记忆
        }

    )


    print(
    "\nAI：",
    result["answer"].content
    )
    print("\nSources:")

    for source in result["sources"]:
        print(source)

    show_memory("user001")

