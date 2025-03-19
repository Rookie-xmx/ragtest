from venv import logger
import requests
import yaml
import psycopg2
from FlagEmbedding import BGEM3FlagModel
from typing import List
from multiprocessing import freeze_support
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, connections, db
)
import os
import numpy as np
import pandas as pd
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


embeddings = SimpleOpenAIEmbeddings(
    api_key="5F5D63CA-477A-4B65-9D8C-336D43CB1D53",
    model_name="Conan-embedding-v1",
    api_url="http://182.43.102.111:8090/v1/embeddings",
)



from pymilvus import utility, FieldSchema, DataType, CollectionSchema, Collection

def create_milvus_collection(collection_type):
    if collection_type == "standard":
        collection_name = "standard"
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="company_name_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1792, nullable=True),
            FieldSchema(name="underwriting_standard_text", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="standard_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1792, nullable=True),
            FieldSchema(name="standard_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR, nullable=True),
        ]
    elif collection_type == "example":
        collection_name = "examples"
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="company_name_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1792, nullable=True),
            FieldSchema(name="product_name", dtype=DataType.VARCHAR, max_length=200, nullable=True), 
            FieldSchema(name="true_example_text", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
            FieldSchema(name="true_example_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1792, nullable=True),
            FieldSchema(name="true_example_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR, nullable=True),
        ]
    else:
        raise ValueError("Invalid collection type. Use 'standard' or 'example'.")

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    schema = CollectionSchema(
        fields=fields,
        description=f"{collection_type} collection",
        enable_dynamic_field=False
    )

    collection = Collection(name=collection_name, schema=schema)

    # 创建索引
    if collection_type == "standard":
        sparse_index = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        }
        collection.create_index("standard_sparse_vector", sparse_index)

        dense_index = {
            "index_type": "FLAT",
            "metric_type": "IP",
        }
        collection.create_index("standard_dense_vector", dense_index)
        collection.create_index("company_name_dense_vector", dense_index)
    elif collection_type == "example":
        sparse_index = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        }
        collection.create_index("true_example_sparse_vector", sparse_index)

        dense_index = {
            "index_type": "FLAT",
            "metric_type": "IP",
        }
        collection.create_index("true_example_dense_vector", dense_index)
        collection.create_index("company_name_dense_vector", dense_index)


    return collection

def process_example(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 初始化一个空列表来存储处理后的数据
    processed_data = []
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        # 提取需要的字段
        company_name = row['保险公司']
        product_name = row['投保险种']
        
        # 拼接true_example_text
        true_example_text = (
            f"以下是{row['保险公司']}的{row['投保险种']}相关健康告知的真实案例。\n"
            f"该投保人情况为：{row['标的性别']}，{row['标的年龄']}岁，总保额{row['总保额(单位:万元)']}万元。\n"
            f"相关健康告知细节：{row['健康告知']}\n"
            f"核保意见为：{row['核保意见']}\n"
            f"核保结果：{row['核保结果']}\n"
            f"特约承保说明：{row['特约承保说明']}\n"
            f"核保说明：{row['核保说明']}"
        )
        
        # 将处理后的数据添加到列表中
        processed_data.append({
            'company_name': company_name,
            'product_name': product_name,
            'true_example_text': true_example_text
        })
    
    return processed_data


import pandas as pd

def process_underwriting_conclusions(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    print(df.columns.tolist())

    # 定义需要处理的保险公司列名
    companies = ['中英人寿', '同方全球', '瑞泰人寿', '中意人寿', '君龙人寿']

    # 初始化结果列表
    result = []

    # 遍历每一行数据
    for index, row in df.iterrows():
        disease_category = row['疾病大类']
        disease_name = row['疾病名称']

        # 遍历每个保险公司
        for company in companies:
            underwriting_standard = row[company]

            # 拼接核保标准文本
            underwriting_standard_text = f"以下是{company}针对于{disease_category}下的{disease_name}的核保标准，标准如下：{underwriting_standard}"

            # 生成字典并添加到结果列表
            result.append({
                'company_name': company,
                'underwriting_standard_text': underwriting_standard_text
            })

    return result



def main():
    with open('../config/ragtest.yaml', 'r', encoding='utf-8') as stream:
        config_api = yaml.safe_load(stream)

    embedding_model_args = config_api['embedding_model']

    # 初始化 BGE 模型用于生成 Sparse Vector
    bge_model = BGEM3FlagModel(embedding_model_args["root"] + embedding_model_args["base"], use_fp16=False)

    milvus_conn_args = config_api['milvus_conn_args']

    # 连接到 Milvus 服务
    try:
        connections.connect(
            alias="default",
            host=milvus_conn_args['host'],
            port=milvus_conn_args['port'],
            user=milvus_conn_args['user'],
            password=milvus_conn_args['password']
        )
        db.using_database("test")
        print("Connected to Milvus successfully.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    #先处理核保的案例
    examples_list = process_example("/data/myba/ragtest/data/核保前置健康告知.xlsx")
        # 确保集合存在
    
    example_collection = create_milvus_collection("example")

    for example in examples_list:
        company_name = example['company_name']
        product_name = example['product_name']
        true_example_text = example['true_example_text']

        company_name_dense_vector = embeddings.embed_documents([company_name])[0]
        # 使用 Conan 生成 Dense Vector
        true_example_dense_vector = embeddings.embed_documents([true_example_text])[0]

        # 使用 BGE 生成 Sparse Vector
        sparse_embedding = bge_model.encode([true_example_text], return_dense=False, return_sparse=True)
        true_example_sparse_vector = sparse_embedding["lexical_weights"][0]

        # print(company_name_dense_vector, true_example_dense_vector, true_example_sparse_vector)

        example_collection.insert([
            {
                "company_name": company_name,
                "company_name_dense_vector": company_name_dense_vector,
                "product_name": product_name,
                "true_example_text": true_example_text,
                "true_example_dense_vector": true_example_dense_vector,
                "true_example_sparse_vector": true_example_sparse_vector,
            }
        ])
        print(f"Inserted data to example_collection for product_name: {product_name}")
    
    # 再处理官方的核保结论
    file_path = '/data/myba/ragtest/data/合作保司重疾险常见疾病核保结论1.xlsx'
    standard_list = process_underwriting_conclusions(file_path)
    standard_collection = create_milvus_collection("standard")

    for standard in standard_list:
        company_name = standard['company_name']
        company_name_dense_vector = embeddings.embed_documents([company_name])[0]
        underwriting_standard_text = standard['underwriting_standard_text']

        # 使用 Conan 生成 Dense Vector
        standard_dense_vector = embeddings.embed_documents([underwriting_standard_text])[0]

        # 使用 BGE 生成 Sparse Vector
        sparse_embedding = bge_model.encode([underwriting_standard_text], return_dense=False, return_sparse=True)
        standard_sparse_vector = sparse_embedding["lexical_weights"][0]

        # print(company_name_dense_vector, true_example_dense_vector, true_example_sparse_vector)

        standard_collection.insert([
            {
                "company_name": company_name,
                "company_name_dense_vector": company_name_dense_vector,
                "underwriting_standard_text": underwriting_standard_text,
                "standard_dense_vector": standard_dense_vector,
                "standard_sparse_vector": standard_sparse_vector,
            }
        ])
        print(f"Inserted data to standard_collection forproduct_name: {underwriting_standard_text}")
    
    


if __name__ == '__main__':
    freeze_support()
    main()