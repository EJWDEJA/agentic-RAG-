test_cases = [    #普通recall precision测试（即混合检索，重排序相关）上下文工程recall也用这个测试

    {
        "question": "T2019033是谁？",
        "expected_answer": "王建国",
        "expected_chunks": [38],
        "question_type": "实体查询"
    },

    {
        "question": "王建国的职称是什么？",
        "expected_answer": "教授/博导",
        "expected_chunks": [38],
        "question_type": "属性查询"
    },

    {
        "question": "王建国的研究方向是什么？",
        "expected_answer": "大数据分析、分布式系统",
        "expected_chunks": [38],
        "question_type": "属性查询"
    },

    {
        "question": "赵雪丽的研究方向是什么？",
        "expected_answer": "大模型与RAG应用、NLP",
        "expected_chunks": [39],
        "question_type": "属性查询"
    },

    {
        "question": "谁是计算机科学与技术2024级2班的班主任？",
        "expected_answer": "周浩",
        "expected_chunks": [40],
        "question_type": "实体查询"
    },

    {
        "question": "孙艺萌负责哪些工作？",
        "expected_answer": "负责2024级全体本科生的日常综合事务、考勤管理、奖学金评定、宿舍安全检查及党团组织发展。",
        "expected_chunks": [41],
        "question_type": "属性查询"
    }

]