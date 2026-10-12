from dataclasses import dataclass    #导入定义书局类的工具


@dataclass              #启动工具
class ContextGroup:     #定义类的名字

    docs: list          #定义字段名字：字段类型
    score: float        #定义字段名字：字段类型
#完成类的定义，跨文件调用需要和函数一样的格式提前1导入


#类本身不会执行当他用来如下创建对象的时候才会执行


def create_context_group(
    core_doc,
    expanded_docs,
    score
):

    return ContextGroup(    #使用类，这里类似于Document（）
    
        docs=expanded_docs,
        score=score
    )     #这里就是创建对象












def create_context_groups(             #用于拓展rerank输出的core doc ,并把它们用类模板创立成对象放进groups列表里面
    scored_docs,
    all_docs,
    neighbor_size=1
):

    groups = []

    for score, core_doc in scored_docs:

        core_source = core_doc.metadata.get("source")
        core_sheet = core_doc.metadata.get("sheet")
        core_chunk_id = core_doc.metadata.get("chunk_id")

        expanded_docs = []    #列表在循环内部每次循环都会用空列表覆盖上次循环的列表消息，想保存多次for数据就把列表放在for循环外面

        for doc in all_docs:

            source = doc.metadata.get("source")
            sheet = doc.metadata.get("sheet")
            chunk_id = doc.metadata.get("chunk_id")

            if (
                source == core_source
                and sheet == core_sheet
                and abs(chunk_id - core_chunk_id) <= neighbor_size
            ):

                expanded_docs.append(doc)

        group = ContextGroup(
            docs=expanded_docs,
            score=score
        )

        groups.append(group)

    return groups








def merge_overlapping_groups(groups):        #不去重硬合并有重叠chunk的group

    merged_groups = []

    for group in groups:

        merged = False            #给当前group做标记方便后续if判断

        group_keys = {            #集合推导式
            (
                doc.metadata.get("source"),
                doc.metadata.get("sheet"),
                doc.metadata.get("chunk_id")
            )
            for doc in group.docs
        }

        for existing_group in merged_groups:     

            existing_keys = {
                (
                    doc.metadata.get("source"),
                    doc.metadata.get("sheet"),
                    doc.metadata.get("chunk_id")
                )
                for doc in existing_group.docs
            }

            if group_keys & existing_keys:       #&这个是两集合求交集的符号，有交集1为true，if执行

                existing_group.docs.extend(group.docs)          #extend是把后列表即括号里面列表里的元素一个个加入前列表，append是把后列表即括号里面的的列表以一个元素整体的形式加入前列表

                existing_group.score = max(         #取两元素里面分数较大的为存在组分数
                    existing_group.score,
                    group.score
                )

                merged = True            #覆盖前面的  merged = False 

                break

        if not merged:      #if not false=if true执行

            merged_groups.append(group)

    return merged_groups                  
#此函数有缺陷因为当1，2号group都无重叠对比的merged group的时候他们会加入merged group后面的三号group和1，2号都有重叠的时候因为和一号加和重叠以后就break了审议本来123都可以合并成一个列表现在只能12，3了              








def merge_overlapping_groups_transitively(groups):          #这段代码解决上面代码无传递性重叠的问题让123可以合成一个列表

    changed = True            #布尔标记表示有合并

    while changed:           #外层循环用作new group里面各元素的重叠合并

        changed = False       #嘉定无合并   

        new_groups = []

        while groups:         #内层循环用作以各位置元素为起始current group的循环比如【A,B,C，D】一次循环起始是A和剩余B,C，D依次测重合并，二层起始是B和C,D依次测重合并三层起始C...当然以remain group为基础

            current_group = groups.pop(0)   #groups,pop（）从groups里面抓取第一个元素并删除这个元素

            current_keys = {
                (
                    doc.metadata.get("source"),
                    doc.metadata.get("sheet"),
                    doc.metadata.get("chunk_id")
                )
                for doc in current_group.docs
            }

            merged_any = False     #没什么用

            remaining_groups = []      #一轮for循环完整结束以后一次内while循环以后没有合并的剩余group

            for group in groups:    #这里的groups是被Pop抽走一个元素以后的groups

                group_keys = {
                    (
                        doc.metadata.get("source"),
                        doc.metadata.get("sheet"),
                        doc.metadata.get("chunk_id")
                    )
                    for doc in group.docs
                }

                if current_keys & group_keys:

                    current_group.docs.extend(
                        group.docs
                    )

                    current_group.score = max(
                        current_group.score,
                        group.score
                    )

                    current_keys |= group_keys    #这里的   1   是并集的意思

                    changed = True         #changee只有一个只要这里发生一次就会修改changed的布尔值外层wwhile直接可以读取用作二次外循环
                    merged_any = True

                else:

                    remaining_groups.append(group)     

            groups = remaining_groups     #这里的remaining group用作内循环while的原料

            new_groups.append(current_group)

        groups = new_groups   #用作外循环的原料

    for group in groups:

        unique_docs = {}

        for doc in group.docs:

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("sheet"),
                doc.metadata.get("chunk_id")
            )

            unique_docs[key] = doc

        group.docs = list(
            unique_docs.values()
        )

    return groups








def sort_group_docs(group):  #merged 以后group的组内升序排序

    group.docs = sorted(
        group.docs,
        key=lambda doc: (
            doc.metadata.get("source", ""),
            doc.metadata.get("sheet", ""),
            int(doc.metadata.get("chunk_id", 0))
        )
    )

    return group




def sort_all_group_docs(groups):

    for group in groups:

        sort_group_docs(group)

    return groups
