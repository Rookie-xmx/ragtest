import yaml
import time
import os
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import (
    utility, FieldSchema, CollectionSchema, DataType,
    Collection, connections, db, AnnSearchRequest, RRFRanker
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class RAGSystem:
    def __init__(self, config_path, prompt_config_path):
        self.load_configs(config_path, prompt_config_path)
        self.initialize_models()
        self.connect_milvus()
        self.initialize_llm()
        self.time_stats = {
            'search': 0.0,
            'rerank': 0.0,
            'llm': 0.0
        }

    def load_configs(self, config_path, prompt_config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(prompt_config_path, 'r', encoding='utf-8') as f:
            self.prompt_config = yaml.safe_load(f)

    def initialize_models(self):
        embedding_args = self.config['embedding_model']
        print("Loading embedding model...") 
        self.embedding_model = BGEM3FlagModel(
            embedding_args["root"] + embedding_args["base"], 
            use_fp16=False
        )
        print("Loading reranker model...") 
        self.reranker = FlagReranker(
            embedding_args["root"] + embedding_args["reranker"],
            use_fp16=True,
            normalize=True
        )
        print("Models loaded successfully.") 

    def connect_milvus(self):
        try:
            conn_args = self.config['milvus_conn_args']
            connections.connect(
                alias="default",
                host=conn_args['host'],
                port=conn_args['port'],
                user=conn_args['user'],
                password=conn_args['password']
            )
            db.using_database("test")
            self.collection = Collection(name='insurance_clauses_FLAT')
            print("Connected to Milvus successfully.")
        except Exception as e:
            raise ConnectionError(f"Milvus connection failed: {e}")

    def initialize_llm(self):
        llm_config = self.config['llm']
        # 读取 api_base 配置
        api_base = llm_config.get("api_base", None)
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_API_KEY"] = self.config['llm']['api_key']

        self.llm = ChatOpenAI(
            model_name=self.config['llm']['model_name']
        )
    
    def _generate_prompt(self, query, references):
        return self.prompt_config["rag_Prompt"].format(
            query=query,
            references=references
        )

    def _search_milvus(self, query, top_k=100):
        start = time.time()
        
        embeddings = self.embedding_model.encode([query], 
                                                return_dense=True,
                                                return_sparse=True)
        embeddings["dense"] = embeddings["dense_vecs"]
        embeddings["sparse"] = [dict(embeddings["lexical_weights"][0])]

        sparse_req = AnnSearchRequest(
            embeddings["sparse"],
            "sparse_vector",
            {"metric_type": "IP"},
            limit=top_k
        )
        dense_req = AnnSearchRequest(
            embeddings["dense"],
            "dense_vector",
            {"metric_type": "IP"},
            limit=top_k
        )

        results = self.collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=RRFRanker(k=50),
            limit=50,
            output_fields=['text']
        )

        self.time_stats['search'] += time.time() - start
        return [(hit.fields['text'], hit.distance, hit.id) for hit in results[0]]

    def _rerank_results(self, query, search_results, top_n=20):
        start = time.time()
        
        passages = [res[0] for res in search_results]
        pairs = [[query, passage] for passage in passages]
        
        scores = self.reranker.compute_score(pairs, normalize=False)
        combined = [{
            'id': search_results[i][2],
            'text': passages[i],
            'score': scores[i]
        } for i in range(len(scores)) if scores[i]> -2.5]
        
        sorted_results = sorted(combined, key=lambda x: x['score'], reverse=True)
        self.time_stats['rerank'] += time.time() - start
        return sorted_results[:top_n]

    def _generate_response(self, prompt):
        start = time.time()
        
        template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "你是一个专业的保险领域专家，从参考资料中回答用户问题。"
            ),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])
        messages = template.format_messages(user_input=prompt)
        response = self.llm.invoke(messages)
        
        self.time_stats['llm'] += time.time() - start
        return response.content

    def get_answer(self, query):
        try:
            # 执行搜索
            search_results = self._search_milvus(query)
            
            # 重新排序
            reranked = self._rerank_results(query, search_results)
            self.last_retrieved_docs = reranked 

            # print(f"reranked的资料如下:                      {reranked}")
            
            # 构建提示
            references = "\n".join(
                [f"- 文档{i+1}: {res['text']}" for i, res in enumerate(reranked)]
            )
            prompt = self._generate_prompt(query, references)

            print(prompt)
            
            # 生成最终响应
            return self._generate_response(prompt)
        except Exception as e:
            return f"处理请求时发生错误: {str(e)}"

    def get_time_stats(self):
        return self.time_stats.copy()

    def reset_time_stats(self):
        self.time_stats = {k: 0.0 for k in self.time_stats}

import streamlit as st
import time

# 假设RAGSystem类和相关依赖已正确导入

@st.cache_resource
def load_rag_system():
    return RAGSystem(
        config_path='/data/myba/ragtest/config/ragtest.yaml',
        prompt_config_path='/data/myba/ragtest/config/prompt.yaml'
    )

def main():
    st.set_page_config(page_title="保险条款智能问答系统", layout="wide")
    
    # 初始化系统
    try:
        rag = load_rag_system()
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        st.stop()

    st.title("📑 保险条款智能问答系统")
    
    # 输入区域
    with st.form("query_form"):
        query = st.text_input("请输入您的问题：", placeholder="例如：意外险的保障范围包括哪些？")
        submitted = st.form_submit_button("发送")
    
    if submitted and query:
        start_time = time.time()
        
        with st.spinner("正在全力为您解答..."):
            try:
                # 获取答案和参考资料
                answer = rag.get_answer(query)
                reranked_docs = rag.last_retrieved_docs
                process_time = time.time() - start_time
                
                # 显示结果
                col1, col2 = st.columns([1, 1], gap="large")
                
                with col1:
                    st.subheader("🔍 检索到的相关资料")
                    if reranked_docs:
                        references = "\n\n".join(
                            [f"📄 文档{i+1}（相关度：{doc['score']:.2f}）:\n{doc['text'][:500]}..." 
                             for i, doc in enumerate(reranked_docs)]
                        )
                        st.text_area(label="参考资料", 
                                    value=references,
                                    height=400,
                                    label_visibility="collapsed")
                    else:
                        st.warning("未找到相关文档")
                
                with col2:
                    st.subheader("💡 最终答案")
                    st.text_area(label="答案", 
                                value=answer,
                                height=1000
                                ,
                                label_visibility="collapsed")
                    
                    # 显示统计信息
                    st.markdown("---")
                    times = rag.get_time_stats()
                    total_time = times['search'] + times['rerank'] + times['llm']
                    
                    cols = st.columns(4)
                    cols[0].metric("搜索耗时", f"{times['search']:.2f}s")
                    cols[1].metric("重排序耗时", f"{times['rerank']:.2f}s")
                    cols[2].metric("大模型响应", f"{times['llm']:.2f}s")
                    cols[3].metric("总耗时", f"{process_time:.2f}s")
                    
                rag.reset_time_stats()
                
            except Exception as e:
                st.error(f"处理请求时发生错误: {str(e)}")

if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     # 使用示例
#     rag = RAGSystem(
#         config_path='/data/myba/ragtest/config/ragtest.yaml',
#         prompt_config_path='/data/myba/ragtest/config/prompt.yaml'
#     )

#     try:
#         while True:
#             query = input("\n请输入您的问题（输入'exit'退出）: ")
#             if query.lower() == 'exit':
#                 break
            
#             start_total = time.time()
#             answer = rag.get_answer(query)
#             total_time = time.time() - start_total
            
#             print("\n答案：")
#             print(answer)
            
#             # 打印时间统计
#             times = rag.get_time_stats()
#             print("\n时间统计（秒）:")
#             print(f"搜索耗时: {times['search']:.2f}")
#             print(f"重排序耗时: {times['rerank']:.2f}")
#             print(f"大模型响应耗时: {times['llm']:.2f}")
#             print(f"总耗时: {total_time:.2f}")
            
#             rag.reset_time_stats()
            
#     except KeyboardInterrupt:
#         print("\n程序已退出。")