import os
# 1. 解决 Keras 3 兼容性
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import re
from typing import List, Dict
import dashscope
from dashscope import Generation

# 真实依赖导入
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 配置与初始化 =================
st.set_page_config(page_title="RAG System - 批量文档知识库", layout="wide")

# 常量定义
DATA_DIR = "data"
DB_NAME = "rag_experiment_4.db"
COLLECTION_NAME = "tcm_knowledge_base"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
Top_K_RETRIEVAL = 10
Top_N_RERANK = 3

# ================= 核心类定义 =================

class DataProcessor:
    """处理文档加载、清洗和切分"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        #text = re.sub(r'\', '', text)
        text = re.sub(r'---\s*PAGE\s*\d+\s*---', '', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        return splitter.split_text(text)

class RAGEngine:
    """RAG 核心引擎：Milvus + Embedding + Reranking"""
    
    def __init__(self):
        # 初始化 Milvus Lite 本地数据库
        self.client = MilvusClient(DB_NAME)
        self._init_models()
        self._init_collection()

    @st.cache_resource
    def _init_models(_self):
        # 加载真实模型
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        reranker_model = CrossEncoder(RERANK_MODEL_NAME)
        return embedding_model, reranker_model

    def _init_collection(self):
        if not self.client.has_collection(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=1024,  # BGE-M3 维度
                metric_type="COSINE",
                auto_id=True
            )

    def ingest_data(self, directory_path: str):
        """批量数据入库流程 (真实处理)"""
        processor = DataProcessor()
        emb_model, _ = self._init_models()
        
        if not os.path.exists(directory_path):
            st.error(f"在 {directory_path} 文件夹中未找到文件！")
            return

        files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
        if not files:
            st.error("数据目录下没有 .txt 文件")
            return

        total_files = len(files)
        progress_bar = st.progress(0, text="准备开始处理...")
        status_text = st.empty()
        
        for i, filename in enumerate(files):
            file_path = os.path.join(directory_path, filename)
            status_text.text(f"正在处理 ({i+1}/{total_files}): {filename}")
            progress_bar.progress((i / total_files))
            
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            # 清洗与切分
            cleaned_text = processor.clean_text(raw_text)
            chunks = processor.chunk_text(cleaned_text)
            
            if not chunks:
                continue

            # 向量化
            embeddings = emb_model.encode(chunks, normalize_embeddings=True)
            
            # 插入 Milvus
            data = [
                {"vector": emb, "text": chunk, "source_file": filename} 
                for emb, chunk in zip(embeddings, chunks)
            ]
            self.client.insert(collection_name=COLLECTION_NAME, data=data)
            
        progress_bar.progress(1.0, text="所有文件处理完成！")
        status_text.text("处理完毕")
        st.toast(f"成功导入 {total_files} 个文件到向量库！", icon="✅")

    def search(self, query: str) -> List[Dict]:
        """检索 + 重排序 (真实逻辑)"""
        emb_model, reranker_model = self._init_models()
        
        # 1. 向量检索
        query_vector = emb_model.encode([query], normalize_embeddings=True)[0]
        
        search_res = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=Top_K_RETRIEVAL,
            output_fields=["text", "source_file"]
        )
        
        hits = search_res[0]
        if not hits:
            return []
            
        retrieved_items = [hit['entity'] for hit in hits]
        retrieved_texts = [item['text'] for item in retrieved_items]

        # 2. 重排序
        pairs = [[query, text] for text in retrieved_texts]
        scores = reranker_model.predict(pairs)
        
        # 排序
        results = sorted(zip(retrieved_items, scores), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for item, score in results[:Top_N_RERANK]:
            final_results.append({
                "text": item['text'],
                "source": item['source_file'],
                "score": float(score)
            })
        
        return final_results

    def generate_answer(self, query: str, context: List[Dict], api_key: str = None, history: List = None) -> str:
        """调用大模型生成回答"""
        if not context:
            return "根据现有知识库，未检索到相关信息。"

        context_str = "\n\n".join([f"[来源: {item['source']}]\n{item['text']}" for item in context])
        
        prompt = f"""基于以下参考资料回答用户问题。如果参考资料中没有答案，请根据常识回答但要说明资料未提及。
        
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
    st.title("🧪 实验4：基于 Streamlit 和 Milvus 的多文档 RAG 系统")
    
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
            st.caption(f"文件夹内共发现 {len(files)} 个 txt 文件")

        if st.button("批量处理并写入向量库", disabled=not dir_exists):
            rag = RAGEngine()
            if rag.client.has_collection(COLLECTION_NAME):
                 st.info("正在追加数据...")
            rag.ingest_data(DATA_DIR)

        if st.button("🚨 清空数据库"):
             rag = RAGEngine()
             if rag.client.has_collection(COLLECTION_NAME):
                rag.client.drop_collection(COLLECTION_NAME)
                st.success("数据库已清空。")
                st.rerun()

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

        rag = RAGEngine()
        
        with st.spinner("正在检索并重排序 (Retrieval & Re-ranking)..."):
            relevant_docs = rag.search(prompt)
            
            if not relevant_docs:
                st.warning("未检索到相关文档。")
            else:
                with st.expander(f"查看检索到的上下文 (Top {Top_N_RERANK} Re-ranked)"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**片段 {i+1}** (Score: {doc['score']:.4f}) - *来源: {doc['source']}*")
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