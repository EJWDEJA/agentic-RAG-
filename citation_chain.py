def get_sources(docs):

    sources = []


    for doc in docs:

        metadata = doc.metadata


        # 文件来源
        source = metadata.get(
            "source",
            "未知来源"
        )


        # Excel Sheet
        sheet = metadata.get(
            "sheet",
            ""
        )


        # 文本块编号
        chunk_id = metadata.get(
            "chunk_id",
            ""
        )


        # 组合来源信息

        if sheet:       

            source_info = (
                f"{source}（{sheet}）"
            )

        else:

            source_info = source



        if chunk_id != "":         #两个if是连贯的这个里面的source info基于上一个if产出的source info加和

            source_info += (
                f"  chunk_id:{chunk_id}"
            )



        sources.append(
            source_info        #把source info存进source开始创造的list
        )


    return list(set(sources))
 
