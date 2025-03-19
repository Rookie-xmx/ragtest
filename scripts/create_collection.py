import yaml
import psycopg2
from FlagEmbedding import BGEM3FlagModel
from multiprocessing import freeze_support
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, connections,db
)

from pymilvus.model.hybrid import BGEM3EmbeddingFunction


def create_milvus_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="uuid", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="product_code", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="pdf_path", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),  # BGE-M3 dense维度
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)  # BGE-M3 sparse向量
    ]
    

    schema = CollectionSchema(
        fields=fields,
        description="Insurance所有条款检索系统",
        enable_dynamic_field=False
    )
    
    collection = Collection(name="test_sparse_insuranc", schema=schema)
    
    # 创建混合索引
    sparse_index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
    }
    collection.create_index("sparse_vector", sparse_index)
    
    dense_index = {
        "index_type": "FLAT",
        "metric_type": "IP", 
    }
    collection.create_index("dense_vector", dense_index)
    
    return collection



def main():
    with open('../config/ragtest.yaml', 'r', encoding='utf-8') as stream:
        config_api = yaml.safe_load(stream)

    embedding_model_args = config_api['embedding_model']

#     bge_m3_ef = BGEM3EmbeddingFunction(
#     model_name= embedding_model_args["root"] + embedding_model_args["base"], # Specify the model name
#     device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
#     use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
# )
    embedding_model = BGEM3FlagModel(embedding_model_args["root"] + embedding_model_args["base"], use_fp16=False)

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
    
    # 确保集合存在
    if not utility.has_collection("test"):
        create_milvus_collection()

    collection = create_milvus_collection()

    pg_conn_args = config_api['pg_conn_args']
    tables = [
        # "kb_chunk_version_0724",
        # "kb_chunk_version_0906",
        # "kb_chunk_version_0906_2",
        # "kb_chunk_version_20240919",
        # "kb_chunk_version_20241008",
        # "kb_chunk_version_20241014",
        # "kb_chunk_version_20241016",
        # "kb_chunk_version_20241021",
        # "kb_chunk_version_all",
        "myba_query_raw_kb_chunk"
    ]
    for table_name in tables:
        TABLE_NAME = table_name
        try:
            conn = psycopg2.connect(
                host=pg_conn_args["DB_HOST"],
                port=pg_conn_args["DB_PORT"],
                user=pg_conn_args["DB_USERNAME"],
                password=pg_conn_args["DB_PASSWORD"],
                dbname=pg_conn_args["DB_DATABASE"]
            )
            print("Connected to PostgreSQL successfully.")


            # TABLE_NAME = pg_conn_args["TABLE_NAME"]
            cursor = conn.cursor()

            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
            total = cursor.fetchone()[0]
            print(f"开始处理表 {TABLE_NAME}, 共 {total} 条数据")

            cursor.execute(f"SELECT product_id, uuid, product_code, key,value, pdf_path, file_name FROM {TABLE_NAME};")
            rows = cursor.fetchall()

            for row in rows:
                try:
                    product_id, uuid, product_code, key, value, pdf_path, file_name = row
                    print(row)
                    combined_text = f"该段文字是产品名称为{file_name}中，章节标题为{key}]的内容。内容如下 {value}"
                    
                    combined_text = combined_text[:5000] if len(combined_text) >= 5000 else combined_text

                    docs_embeddings = embedding_model.encode([combined_text],    
                                                            return_dense=True, 
                                                            return_sparse=True)

                    # Print embeddings
                    # print("Embeddings:", docs_embeddings)
                    # # Print dimension of dense embeddings
                    # print("Dense document dim:", docs_embeddings["dense_vecs"][0].shape)

                    dense_vector=docs_embeddings["dense_vecs"][0]
                    sparse_vector=docs_embeddings["lexical_weights"][0]

                    # print(dense_vector)

                    # print("@@@@@@@@@@@@@@@")

                    print(sparse_vector)

                    collection.insert([
                        {
                            "product_id": product_id,
                            "uuid": uuid,
                            "product_code": product_code,
                            "key": key,
                            "pdf_path": pdf_path,
                            "file_name": file_name,
                            "text": combined_text,
                            "dense_vector": dense_vector,
                            "sparse_vector": sparse_vector
                        }
                    ])
                    print(f"Inserted data for product_id: {product_id}")

                except Exception as e:
                    print(f"Error processing row with product_id {row[0]}: {e}")
                    continue  # 跳过当前行，继续处理下一行

            print("Data insertion completed.")

        except psycopg2.Error as e:
            print(f"Failed to connect to PostgreSQL or execute query: {e}")
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                print("PostgreSQL connection closed.")


if __name__ == '__main__':
    freeze_support()
    main()