from typing import List
import requests
import yaml
import time
import os
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from pymilvus import (
    utility, FieldSchema, CollectionSchema, DataType,
    Collection, connections, db, AnnSearchRequest, RRFRanker
)
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
import streamlit as st

# 配置日志记录器
import logging
logger = logging.getLogger(__name__)

class SimpleOpenAIEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        try:
            response = requests.post(
                "http://182.43.102.111:8090/v1/embeddings",
                headers=self._headers,
                json={
                    "input": texts,
                    "model": "Conan-embedding-v1",
                },
            )
            response = response.json()
            batched_embeddings.extend(r["embedding"] for r in response["data"])
        except Exception as e:
            logger.error('Exception: %s', e)
        return batched_embeddings

conan_embeddings = SimpleOpenAIEmbeddings(
    api_key="5F5D63CA-477A-4B65-9D8C-336D43CB1D53",
    model_name="Conan-embedding-v1",
    api_url="http://182.43.102.111:8090/v1/embeddings",
)

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
        self.last_retrieved_docs = [] # 用于存储最近一次检索到的文档

    def load_configs(self, config_path, prompt_config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(prompt_config_path, 'r', encoding='utf-8') as f:
            self.prompt_config = yaml.safe_load(f)

    def initialize_models(self):
        embedding_args = self.config['embedding_model']
        print("Loading embedding model...") 
        self.bge_embedding_model = BGEM3FlagModel(
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
            self.examples_collection = Collection(name='examples')
            self.standard_collection = Collection(name='standard')
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
        return self.prompt_config["Underwriting_Prompt"].format(
            query=query,
            references=references
        )

    def _search_milvus(self, query, top_k=200): ##分为3步骤，第一步按照companyname vector搜索，第二步搜索examples collection，第三步搜索standard collection
        start = time.time()
        
        ##query 的稠密向量，conan，dims len 1792
        query_dense_vector = conan_embeddings.embed_documents([query])

        ##query 的稀疏向量，bge-m3的embedding model
        bge_embeddings = self.bge_embedding_model.encode([query], 
                                                return_dense=False,
                                                return_sparse=True)

        bge_embeddings["sparse"] = [dict(bge_embeddings["lexical_weights"][0])]
        print(bge_embeddings["sparse"])

        ##先按照company_name进行搜索两个collection，找出top2的company_name，划定范围。
        # Step 1: 获取两个collection的top2 company_name
        def get_top_companies(collection, vector_field,similarity_threshold=0.8):
            results = collection.search(
                data=[query_dense_vector[0]],
                anns_field=vector_field,
                param={"metric_type": "IP"},
                limit=100,
                output_fields=["company_name"]
            )

            #过滤掉相似度低于similarity_threshold的结果
            filtered_results = []
            for hit in results[0]:
                if hit.distance >= similarity_threshold:  # 根据相似度分数过滤结果
                    filtered_results.append(hit)
            unique_companies = set()  # 用于存储不同的公司名

            for hit in filtered_results:
                unique_companies.add(hit.entity.company_name)
                if len(unique_companies) >= 2:  # 当找到两个不同的公司名后停止
                    break
            return list(unique_companies)[:2]  # 返回前两个不同的公司名
        
        examples_companies = get_top_companies(self.examples_collection, "company_name_dense_vector")
        standard_companies = get_top_companies(self.standard_collection, "company_name_dense_vector")

        print(f"examples_companies: {examples_companies}")
        print(f"standard_companies: {standard_companies}")


        # Step 2: 执行混合搜索

        def hybrid_search(collection, dense_field, sparse_field, expr, fields,top_k=50,k_rrf=50):

            sparse_req = AnnSearchRequest(
                bge_embeddings["sparse"] ,
                sparse_field,
                {"metric_type": "IP"},
                limit=top_k,
                expr=expr
            )

            dense_req = AnnSearchRequest(
                query_dense_vector,
                dense_field,
                {"metric_type": "IP"},
                limit=top_k,
                expr=expr
            )


            return collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=RRFRanker(k=k_rrf),
            limit=50,
            output_fields=fields
            ) 
        
        def format_companies(companies):
            return "['" + "', '".join(companies) + "']"
        # 搜索examples collection

        examples_results = hybrid_search(
            self.examples_collection,
            "true_example_dense_vector",
            "true_example_sparse_vector",
            f"company_name in {format_companies(examples_companies)}" if examples_companies else "",
            ["company_name", "product_name", "true_example_text"]
            )

        # 搜索standard collection
        standard_results = hybrid_search(
            collection=self.standard_collection,
            dense_field="standard_dense_vector",
            sparse_field="standard_sparse_vector",
            expr=f"company_name in {format_companies(standard_companies)}" if standard_companies else "",
            fields=["company_name", "underwriting_standard_text"],
            top_k=50,
            k_rrf=50
        )

        # 合并结果
        combined = []
        for hit in examples_results[0]:
            combined.append((
                hit.fields['true_example_text'],
                hit.distance,
                hit.id,
                'example'  # 结果类型标识
            ))
        
        for hit in standard_results[0]:
            combined.append((
                hit.fields['underwriting_standard_text'],
                hit.distance,
                hit.id,
                'standard'
            ))

        self.time_stats['search'] += time.time() - start
        return combined


    def _rerank_results(self, query, search_results, top_n=10):
        start = time.time()
        
        passages = [res[0] for res in search_results]
        pairs = [[query, passage] for passage in passages]
        
        scores = self.reranker.compute_score(pairs, normalize=False)
        combined = []
        for i in range(len(scores)):
            if scores[i] > -4:
                res = search_results[i]
                item = {
                    'text': passages[i],
                    'score': scores[i],
                    'type': res[-1],  # 结果类型
                }
                combined.append(item)

        sorted_results = sorted(combined, key=lambda x: x['score'], reverse=True)
        self.time_stats['rerank'] += time.time() - start
        return sorted_results[:top_n]

    def _generate_response(self, prompt):
        start = time.time()
        
        template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "你是一个专业的保险领域预核保专家，善于从参考资料中找到相关依据并根据自身专业的核保领域知识来回答用户问题。"
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
            self.last_retrieved_docs = reranked  # 保存检索到的文档

            print(f"reranked的资料如下:                      {reranked}")
            
            # 构建提示
            references = "\n".join(
            [f"文档{i+1}：{('真实预核保案例' if res['type'] == 'example' else '核保标准')}，{res['text'][0:800]}" for i, res in enumerate(reranked)]  )
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
    st.set_page_config(page_title="保险核保智能助手", layout="wide")
    
    try:
        rag = load_rag_system()
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        st.stop()

    st.title("📑 保险核保智能助手")
    
    with st.form("query_form"):
        query = st.text_input("请输入您的问题：", placeholder="例如：我有心脏病可以买中意的哪些产品")
        submitted = st.form_submit_button("发送")
    
    if submitted and query:
        start_time = time.time()
        
        with st.spinner("正在全力为您解答..."):
            try:
                answer = rag.get_answer(query)
                reranked_docs = rag.last_retrieved_docs
                process_time = time.time() - start_time
                
                col1, col2 = st.columns([1, 1], gap="large")
                
                with col1:
                    st.subheader("🔍 检索到的相关资料")
                    if reranked_docs:
                        references = "\n\n".join(
                            [f"📄 文档{i+1}（类型：{'案例' if doc['type'] == 'example' else '标准'}，相关度：{doc['score']:.2f}）:\n{doc['text'][:500]}..." 
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
                    st.markdown(f"```\n{answer}\n```", unsafe_allow_html=True)
                    
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