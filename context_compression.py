import re    #导入正则表达式模块regex



from langchain_community.embeddings import OllamaEmbeddings   #导入类

from sklearn.metrics.pairwise import cosine_similarity     #导入余弦相似度函数

from token_counter import count_tokens

def split_sentences(text):

    sentences = re.split(       #按正则表达式切分句子         #|是或者的意思\s用于识别单个空白符号回车\n\t都可以\n+用于识别一个或多个换行符
         r"(?<=[。！？])|(?<=[.!?])(?=\s|$)|\n+",   #规则   其中r是python内置函数raw string初始化字符目标是不执行方括号内常见的\d  \n这种字符。？<=在符号后切割保留符号不加正则会把符号删除【】内是正则识别的符号用于切分
        text      #文本
    )

    return [
        sentence.strip()   #strip删除首尾空白  3保存
        for sentence in sentences   #一个个检索句子列表内元素 1
        if sentence.strip()      #判断句子元素是否为空围困不执行不会出现在返回列表 2
    ]





embeddings = OllamaEmbeddings(
    model="qwen3-embedding"
)


def calculate_sentence_scores(     
    question,
    sentences
):

    question_vector = embeddings.embed_query(    #把question向量化，embed_query为OllamaEmbeddings这个列内置方法
        question
    )

    sentence_vectors = embeddings.embed_documents(     #把sentences向量化并放入列表，每句话一个单向量是一个列表元素
        sentences
    )

    scores = cosine_similarity(
        [question_vector],   #把向量化以后的问题放入列表和 sentence_vectors格式一样
        sentence_vectors
    )[0]     #把各socre放进一个列表并把这个列表方进一个列表取出第一个元素作为socre

    return list(
        zip(     #把两个列表元素一一对应
            scores,   #socres列表
            sentences   #sentence列表
        )
    )









def select_relevant_sentences(
    scored_sentences,
    threshold=0.5,
    min_sentences=1
):

    selected_sentences = []

    for score, sentence in scored_sentences:

        if score >= threshold:

            selected_sentences.append(
                sentence
            )


    if len(selected_sentences) >= min_sentences:

        return selected_sentences


    sorted_sentences = sorted(
        scored_sentences,
        key=lambda x: x[0],
        reverse=True
    )


    selected_sentences = [
        sentence
        for score, sentence
        in sorted_sentences[:min_sentences]
    ]


    return selected_sentences








def compress_text(
    question,
    text,
    threshold=0.5,
    min_sentences=1
):

    sentences = split_sentences(
        text
    )

    scored_sentences = calculate_sentence_scores(
        question,
        sentences
    )

    selected_sentences = select_relevant_sentences(
        scored_sentences,
        threshold,
        min_sentences
    )

    return "\n".join(
        selected_sentences
    )




def compress_group(
    question,
    group,
    threshold=0.5,
    min_sentences=1
):

    compressed_docs = []

    for doc in group.docs:

        source = doc.metadata.get(
            "source",
            ""
        ).lower()


        # =========================
        # Excel 文件
        # 不进行句子级 Compression
        # =========================

        if source.endswith(".xlsx") or source.endswith(".xls"):

            compressed_docs.append(doc)

            continue


        # =========================
        # 普通文本
        # 使用原来的 Compression
        # =========================

        compressed_text = compress_text(
            question,
            doc.page_content,
            threshold=threshold,
            min_sentences=min_sentences
        )


        if compressed_text:

            doc.page_content = compressed_text

            compressed_docs.append(doc)


    group.docs = compressed_docs

    return group



def recalculate_group_tokens(group):

    text = "\n".join(
        doc.page_content
        for doc in group.docs
    )

    return count_tokens(text)
