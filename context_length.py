from token_counter import count_tokens        #按token数多少控制上下文依次加和doc超长直接停止后续识别


def limit_context_tokens(
    docs,
    max_tokens=3000
):

    selected_docs = []

    total_tokens = 0

    for doc in docs:

        doc_tokens = count_tokens(
            doc.page_content
        )

        if total_tokens + doc_tokens > max_tokens:
            break

        selected_docs.append(doc)

        total_tokens += doc_tokens

    return selected_docs
