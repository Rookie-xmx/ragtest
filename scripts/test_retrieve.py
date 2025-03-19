from typing import List
from venv import logger
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
            self.collection = Collection(name='test_standard')
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

    def _search_milvus(self, query, top_k=200):
        start = time.time()
        
        embeddings = self.embedding_model.encode([query], 
                                                return_dense=True,
                                                return_sparse=True)
        embeddings["dense"] = conan_embeddings.embed_documents([query])
        print(embeddings["dense"])
        print(len(embeddings["dense"][0]))
        # embeddings["dense"] = embeddings["dense_vecs"]
        # print(embeddings["dense"])
        # print(len(embeddings["dense"][0]))
        embeddings["sparse"] = [dict(embeddings["lexical_weights"][0])]
        print(embeddings["sparse"])

        sparse_req = AnnSearchRequest(
            embeddings["sparse"],
            "standard_sparse_vector",
            {"metric_type": "IP"},
            limit=top_k
        )
        dense_req = AnnSearchRequest(
            embeddings["dense"],
            "standard_dense_vector",
            {"metric_type": "IP"},
            limit=top_k
        )

        results = self.collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=RRFRanker(k=50),
            limit=50,
            output_fields=['company_name',"underwriting_standard_text"]
        )
        print("混合搜索召回的资料",results)

        self.time_stats['search'] += time.time() - start
        return [(hit.fields['text'], hit.distance, hit.id, hit.fields['product_id'], hit.fields['file_name']) for hit in results[0]]

    def _rerank_results(self, query, search_results, top_n=30):
        start = time.time()
        
        passages = [res[0] for res in search_results]
        pairs = [[query, passage] for passage in passages]
        
        scores = self.reranker.compute_score(pairs, normalize=False)
        combined = [{
            'product_id': search_results[i][3],
            'file_name':search_results[i][4],
            'text': passages[i],
            'score': scores[i]
        } for i in range(len(scores)) if scores[i]>-4]
        
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

            print(f"reranked的资料如下:                      {reranked}")
            
            # 构建提示
            references = "\n".join(
            [f"- 文档{i+1} (产品ID: {res['product_id']}): {res['text'][0:500]}" for i, res in enumerate(reranked)]  )
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

if __name__ == '__main__':
    # 使用示例
    rag = RAGSystem(
        config_path='/data/myba/ragtest/config/ragtest.yaml',
        prompt_config_path='/data/myba/ragtest/config/prompt.yaml'
    )

    try:
        while True:
            query = input("\n请输入您的问题（输入'exit'退出）: ")
            if query.lower() == 'exit':
                break
            
            start_total = time.time()
            answer = rag.get_answer(query)
            total_time = time.time() - start_total
            
            print("\n答案：")
            print(answer)
            
            # 打印时间统计
            times = rag.get_time_stats()
            print("\n时间统计（秒）:")
            print(f"搜索耗时: {times['search']:.2f}")
            print(f"重排序耗时: {times['rerank']:.2f}")
            print(f"大模型响应耗时: {times['llm']:.2f}")
            print(f"总耗时: {total_time:.2f}")
            
            rag.reset_time_stats()
            
    except KeyboardInterrupt:
        print("\n程序已退出。")