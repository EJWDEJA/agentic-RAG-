import sys     #测试hybird search的命中率
import os

sys.path.append(      #sys.path是一个列表，内部默认第一个元素即可为当前.py文件所在目录（文件夹）的地址 另外还有python标准库第三方库...          这一行的作用是添加新的目录地址
    os.path.dirname(    #当前文件所在文件夹所在文件夹的地址
        os.path.dirname(        #当前.py文件所在文件夹的地址
            os.path.abspath(__file__)   #当前.py文件的绝对地址
        )
    )
)


from test_dataset import test_cases
from hybrid_retriever import hybrid_search


def evaluate_retrieval(test_cases):

    hit_count = 0

    total_count = len(test_cases)

    for i, case in enumerate(test_cases, start=1):

        question = case["question"]

        expected_chunks = set(
            case["expected_chunks"]
        )

        docs = hybrid_search(question)

        retrieved_chunks = {
            doc.metadata.get("chunk_id")
            for doc in docs
        }

        hit = bool(    #把一个东西转换成true or flash   判断有无或者是否成立
            expected_chunks & retrieved_chunks
        )

        if hit:
            hit_count += 1

        print("\n==============================")
        print(f"测试 {i}")
        print("问题:", question)
        print("正确 Chunk:", expected_chunks)
        print("检索 Chunk:", retrieved_chunks)
        print("结果:", "✓ 命中" if hit else "✗ 未命中")

    hit_rate = hit_count / total_count

    print("\n==============================")
    print("Retrieval Evaluation")
    print("==============================")
    print("总问题数:", total_count)
    print("命中数量:", hit_count)
    print("Hit Rate:", hit_rate)


if __name__ == "__main__":   #这是一个函数开关直接执行脚本可以执行，进行脚本调用不能执行因为每个,py文件python都会给他一个name，直接调用name和main一样，跨文件调用时name是脚本名字脚本不执行

    evaluate_retrieval(test_cases)
