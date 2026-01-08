import os
# 1. 解决 Keras 3 兼容性
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import re
import numpy as np
import pickle
import networkx as nx
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

# 真实依赖
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dashscope
from dashscope import Generation

# ================= 配置与初始化 =================
st.set_page_config(page_title="RAG System - 批量文档知识库", layout="wide")

# 常量定义
DATA_DIR = "data"
GRAPH_SAVE_PATH = "graph_knowledge_base.pkl"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# GraphRAG 参数
GRAPH_SIM_THRESHOLD = 0.75  # 相似度 > 0.75 则建立语义边
Top_K_ANCHORS = 5           # 锚点数量
Top_N_FINAL = 3             # 最终输出数量

# ================= 核心类定义 =================

class GraphNode:
    """图节点结构"""
    def __init__(self, node_id: int, text: str, source: str, vector: np.ndarray):
        self.id = node_id
        self.text = text
        self.source = source
        self.vector = vector

class GraphRAGEngine:
    """基于 NetworkX 的本地 GraphRAG 引擎 (真实逻辑)"""
    
    def __init__(self):
        self.embedding_model = None
        self.reranker_model = None
        self.graph = None
        self.nodes = {}
        
        self._init_models()
        self._load_graph()

    @st.cache_resource
    def _init_models(_self):
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        reranker_model = CrossEncoder(RERANK_MODEL_NAME)
        return embedding_model, reranker_model

    def _load_graph(self):
        """加载图谱"""
        if os.path.exists(GRAPH_SAVE_PATH):
            try:
                with open(GRAPH_SAVE_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.nodes = data['nodes']
            except Exception:
                self.graph = nx.Graph()
                self.nodes = {}
        else:
            self.graph = nx.Graph()
            self.nodes = {}

    def clean_text(self, text: str) -> str:
        #text = re.sub(r'\', '', text)
        text = re.sub(r'---\s*PAGE\s*\d+\s*---', '', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        return splitter.split_text(text)

    def ingest_data(self, directory_path: str):
        """全流程真实入库：Chunk -> Vector -> Graph Construction"""
        emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME) # 重新获取避免缓存冲突
        
        if not os.path.exists(directory_path):
            st.error(f"目录 {directory_path} 不存在")
            return

        files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
        if not files:
            st.error("数据目录下没有 .txt 文件")
            return

        all_chunks = []
        all_sources = []
        
        progress_bar = st.progress(0, text="开始读取文档...")
        status_text = st.empty()

        # 1. 读取与处理
        for i, filename in enumerate(files):
            status_text.text(f"正在解析 ({i+1}/{len(files)}): {filename}")
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = self.clean_text(f.read())
                chunks = self.chunk_text(content)
                all_chunks.extend(chunks)
                all_sources.extend([filename] * len(chunks))
            progress_bar.progress((i + 1) / (len(files) * 3))

        if not all_chunks:
            st.warning("文档为空")
            return

        # 2. 批量向量化
        status_text.text("正在计算节点向量...")
        vectors = emb_model.encode(all_chunks, normalize_embeddings=True)
        progress_bar.progress(0.6)

        # 3. 构建图谱
        status_text.text("正在构建语义关联图谱...")
        self.graph = nx.Graph()
        self.nodes = {}

        # 添加节点
        for idx, (text, src, vec) in enumerate(zip(all_chunks, all_sources, vectors)):
            node = GraphNode(idx, text, src, vec)
            self.nodes[idx] = node
            self.graph.add_node(idx)
        
        # 计算相似度并建边 (真实计算)
        sim_matrix = cosine_similarity(vectors)
        rows, cols = np.where(sim_matrix > GRAPH_SIM_THRESHOLD)
        
        for r, c in zip(rows, cols):
            if r < c: # 避免重复和自环
                self.graph.add_edge(r, c, weight=sim_matrix[r][c])
        
        progress_bar.progress(1.0)
        status_text.text("图谱构建完成并保存")

        # 持久化
        with open(GRAPH_SAVE_PATH, 'wb') as f:
            pickle.dump({'graph': self.graph, 'nodes': self.nodes}, f)
        
        st.toast(f"成功构建包含 {len(self.nodes)} 个节点的图谱！", icon="✅")
        st.rerun()

    def search(self, query: str) -> List[Dict]:
        """GraphRAG 检索 (真实逻辑: 锚点+邻居)"""
        if not self.graph or not self.nodes:
            return []

        emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        reranker = CrossEncoder(RERANK_MODEL_NAME)

        # 1. 向量检索找锚点
        query_vec = emb_model.encode([query], normalize_embeddings=True)[0]
        node_vectors = np.array([n.vector for n in self.nodes.values()])
        sims = cosine_similarity([query_vec], node_vectors)[0]
        
        anchor_indices = np.argsort(sims)[::-1][:Top_K_ANCHORS]
        
        # 2. 图扩展 (NodeRAG)
        candidate_ids = set(anchor_indices)
        for nid in anchor_indices:
            neighbors = list(self.graph.neighbors(nid))
            candidate_ids.update(neighbors)
        
        candidate_nodes = [self.nodes[nid] for nid in candidate_ids]
        candidate_texts = [n.text for n in candidate_nodes]
        
        if not candidate_texts:
            return []

        # 3. 重排序
        pairs = [[query, text] for text in candidate_texts]
        scores = reranker.predict(pairs)
        
        sorted_indices = np.argsort(scores)[::-1][:Top_N_FINAL]
        
        final_results = []
        for idx in sorted_indices:
            node = candidate_nodes[idx]
            final_results.append({
                "text": node.text,
                "source": node.source,
                "score": float(scores[idx])
            })
            
        return final_results

    def generate_answer(self, query: str, context: List[Dict], api_key: str = None, history: List = None) -> str:
        """真实 LLM 生成"""
        if not context:
            return "未检索到相关资料。"

        context_str = "\n\n".join([f"[来源: {item['source']}]\n{item['text']}" for item in context])
        
        prompt = f"""基于以下参考资料回答用户问题。
        
参考资料：
{context_str}

用户问题：{query}
"""
        
        if not api_key:
            return "【提示】请在侧边栏输入 DashScope API Key 以开启 AI 回答。\n\n" \
                   f"**检索到的背景知识：**\n{context_str}"

        dashscope.api_key = api_key
        messages = []
        if history:
            for h in history:
                messages.append({'role': h['role'], 'content': h['content']})
        messages.append({'role': 'user', 'content': prompt})

        try:
            response = Generation.call(
                Generation.Models.qwen_turbo,
                messages=messages,
                result_format='message',
            )
            if response.status_code == 200:
                return response.output.choices[0]['message']['content']
            else:
                return f"API Error: {response.code} - {response.message}"
        except Exception as e:
            return f"LLM 调用出错: {str(e)}"

# ================= Streamlit UI 逻辑 =================

def main():
    st.title("🧪 实验4：基于 GraphRAG 的多文档知识库")
    
    with st.sidebar:
        st.header("系统设置")
        api_key = st.text_input("DashScope API Key", type="password", help="阿里云百炼 Qwen API Key")
        
        st.divider()
        st.header("1. 知识库构建 (Ingestion)")
        
        dir_exists = os.path.exists(DATA_DIR)
        status_color = "green" if dir_exists else "red"
        st.markdown(f"检测数据文件夹 `{DATA_DIR}/`: :{status_color}[{'存在' if dir_exists else '缺失'}]")
        
        if dir_exists:
            files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
            st.caption(f"共发现 {len(files)} 个文档")

        if st.button("批量处理并构建知识图谱", disabled=not dir_exists):
            rag = GraphRAGEngine()
            rag.ingest_data(DATA_DIR)

        if st.button("🚨 删除图谱缓存"):
            if os.path.exists(GRAPH_SAVE_PATH):
                os.remove(GRAPH_SAVE_PATH)
                st.success("缓存已删除")
                st.rerun()
            else:
                st.info("暂无缓存")

    # --- 主界面 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        rag = GraphRAGEngine()
        
        with st.spinner("正在执行图谱检索 (Graph Traversal)..."):
            relevant_docs = rag.search(prompt)
            
            if not relevant_docs:
                st.warning("未检索到相关内容，请先构建知识库。")
            else:
                with st.expander(f"查看检索到的上下文 (Top {Top_N_FINAL} from Graph)"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**图节点片段 {i+1}** (Rel-Score: {doc['score']:.4f}) - *{doc['source']}*")
                        st.text(doc['text'])
                        st.divider()

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = rag.generate_answer(
                query=prompt, 
                context=relevant_docs, 
                api_key=api_key,
                history=st.session_state.messages[:-1] 
            )
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()