# 🧪 RAG System for TCM Knowledge Base
> **Experiment 4: Retrieval-Augmented Generation based on Streamlit & Milvus/GraphRAG**

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)
![Milvus](https://img.shields.io/badge/VectorDB-Milvus_Lite-00a1ea.svg)
![GraphRAG](https://img.shields.io/badge/Architecture-GraphRAG-success.svg)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 项目简介 (Introduction)

本项目是 **“数据挖掘与应用”** 课程实验作品。旨在构建一个垂直领域（中医/血液病）的检索增强生成（RAG）系统，解决大语言模型在特定领域产生“幻觉”的问题。

系统实现了两种不同的检索架构，分别探索了**密集向量检索**与**图谱增强检索**在知识关联挖掘上的差异：

1.  **Milvus 版本**: 基于 `pymilvus` 实现标准的向量存储与 ANN 检索。
2.  **GraphRAG 版本**: 基于 `NetworkX` 构建内存语义图谱，实现“锚点+邻居扩展”的 NodeRAG 检索。

---

## ✨ 核心功能 (Key Features)

* **多模式检索引擎**:
    * 🧩 **Vector Search**: 使用 BAAI/bge-m3 生成高维向量，利用余弦相似度进行快速召回。
    * 🕸️ **Graph Traversal**: 基于语义相似度构建文档切片图谱，通过图遍历发现隐性关联。
* **混合重排序 (Hybrid Re-ranking)**: 集成 Cross-Encoder 模型，对召回结果进行精细打分，大幅提升 Top-3 准确率。
* **全流程可视化**: 从数据清洗、切片、入库到检索生成，所有步骤均通过 Streamlit 前端实时展示。
* **零幻觉机制**: 严格基于检索到的 `data` 上下文回答，若无相关资料则拒绝回答。

---

## 📂 目录结构 (Directory Structure)

```text
TCM-RAG-System/
├── data/                   # 知识库源数据 (.txt)
│   └── all.txt             # 示例：吴银根/邱佳信学术经验集
├── app_milvus.py           # 启动入口：Milvus 向量数据库版
├── app_graph.py            # 启动入口：GraphRAG 图谱增强版
├── requirements.txt        # 项目依赖库
└── README.md               # 项目说明文档
