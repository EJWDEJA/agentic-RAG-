from langchain_core.documents import Document    #用于制作虚假的测试集来测试上下文工程召回精准率（未计算版）


test_docs = [

    # ==============================
    # 场景 1：相邻 Chunk
    # ==============================

    Document(
        page_content="项目使用 MySQL 存储业务数据。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_1",
            "chunk_id": 100
        }
    ),

    Document(
        page_content="用户表保存用户基本信息，包括用户编号、姓名和联系方式。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_1",
            "chunk_id": 101
        }
    ),

    Document(
        page_content="用户表通过 user_id 与订单表进行关联，用于查询某个用户对应的订单记录。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_1",
            "chunk_id": 102
        }
    ),

    Document(
        page_content="Redis 用于缓存热点数据，以减少数据库的访问压力。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_1",
            "chunk_id": 103
        }
    ),


    # ==============================
    # 场景 2：多个 Chunk 共同回答
    # ==============================

    Document(
        page_content="系统采用 MySQL 作为主要关系数据库。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_2",
            "chunk_id": 200
        }
    ),

    Document(
        page_content="订单数据主要存储在 orders 表中。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_2",
            "chunk_id": 201
        }
    ),

    Document(
        page_content="orders 表通过 user_id 与 users 表关联，从而建立订单与用户之间的对应关系。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_2",
            "chunk_id": 202
        }
    ),

    Document(
        page_content="Redis 用于缓存热门商品信息，提高商品查询速度。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_2",
            "chunk_id": 203
        }
    ),


    # ==============================
    # 场景 3：相关 Chunk 距离较远
    # ==============================

    Document(
        page_content="系统使用 MySQL 保存订单数据。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_3",
            "chunk_id": 300
        }
    ),

    Document(
        page_content="系统提供用户注册、登录和权限管理功能。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_3",
            "chunk_id": 301
        }
    ),

    Document(
        page_content="系统使用 Redis 保存部分高频访问数据。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_3",
            "chunk_id": 302
        }
    ),

    Document(
        page_content="系统通过日志服务记录用户操作行为。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_3",
            "chunk_id": 303
        }
    ),

    Document(
        page_content="系统前端采用 Vue 构建管理后台。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_3",
            "chunk_id": 304
        }
    ),

    Document(
        page_content="订单表建立 user_id 索引，以提高按照用户查询订单记录时的检索效率。",
        metadata={
            "source": "synthetic_context_test",
            "sheet": "scene_3",
            "chunk_id": 305
        }
    )
]


golden_context_dataset = [

    {
        "question": "用户表和订单表是如何关联的？",
        "relevant_chunk_ids": {102}
    },

    {
        "question": "系统的订单数据存储在哪里，以及订单和用户是如何关联的？",
        "relevant_chunk_ids": {201, 202}
    },

    {
        "question": "MySQL 中订单数据的查询效率是如何优化的？",
        "relevant_chunk_ids": {300, 305}
    }
]
