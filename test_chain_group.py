from rag_chain_group import rag_chain

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
                "session_id": "user001"
            }
        }
    )


    print(
        "\nAI：",
        result["answer"].content
    )


    show_memory(
        "user001"
    )