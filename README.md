# agentic-RAG-
LangGraph 构建 Agentic RAG 系统，在传统 RAG 检索流程的基础上，引入证据充分性评估、定向补充检索、引用校验、失败分类与答案修复机制，形成具备自我检查与错误恢复能力的检索增强生成闭环。 项目不仅关注系统能否生成答案，还关注答案是否正确、证据是否充分、引用是否有效，以及系统失败的原因。进一步构建 Evaluation → Failure Analysis → Hypothesis Management → Experiment Design → Experiment Execution → Result Analysis 的实验优化流程，通过可控实验验证优化假设，为后续系统决策提供依据。
