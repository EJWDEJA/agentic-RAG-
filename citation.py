import os

def build_reference(docs):

    references = []

    for doc in docs:

        metadata = doc.metadata

        source = metadata.get("source", "未知文件")

        filename = os.path.basename(source)

        if "sheet" in metadata:

            sheet = metadata["sheet"]

            references.append(
                f"{filename}（工作表：{sheet}）"
            )

        else:

            references.append(filename)

    references = list(dict.fromkeys(references))

    text = "\n\n 来源：\n"

    for i, ref in enumerate(references, 1):

        text += f"{i}. {ref}\n"

    return text
