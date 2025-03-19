import psycopg2
import pandas as pd
from ast import literal_eval
import yaml


def safe_literal_eval(x):
    try:
        return literal_eval(x)
    except (ValueError, SyntaxError):  # 捕获可能的异常
        print(f"Warning: Could not parse value '{x}'. The containing row will be dropped.")
        return None  # 返回None或其他标识符表示这一行应该被删除
    
def filter_existing_products(input_file, output_file, db_config):

    with open('../config/ragtest.yaml', 'r', encoding='utf-8') as stream:
        config_api = yaml.safe_load(stream)

    pg_conn_args = config_api['pg_conn_args']
    # 读取Excel文件
    df = pd.read_excel(input_file)
    
    # 应用safe_literal_eval函数，并使用结果更新DataFrame
    for col in ['产品id','产品名称','维度']:
        df[col] = df[col].apply(safe_literal_eval)

    # 根据是否为None（或你选择的其他标识符）来过滤掉那些包含错误的行
    df = df.dropna(subset=['产品id','维度','产品名称'])
    print(f"共有{len(df)}行数据")
    # 收集所有唯一产品ID

    all_ids = set()
    for ids in df['产品id']:
        all_ids.update(ids)
    
    if not all_ids:
        print("没有需要处理的产品ID")
        return
    
    # 连接数据库查询存在的ID
    try:
        conn = psycopg2.connect(
            host=pg_conn_args["DB_HOST"],
            port=pg_conn_args["DB_PORT"],
            user=pg_conn_args["DB_USERNAME"],
            password=pg_conn_args["DB_PASSWORD"],
            dbname=pg_conn_args["DB_DATABASE"]
        )
        cursor = conn.cursor()
        
        # 分批次查询（避免SQL过长）
        batch_size = 1000
        existing_ids = set()
        id_list = list(all_ids)
        print(f"共有{len(id_list)}个产品ID")
        id_list_str = ','.join(f"'{str(id)}'" for id in id_list)    


        query = f"SELECT product_id FROM kb_chunk_version_all WHERE product_id IN ({id_list_str})"
        # print(query)
        cursor.execute(query, id_list)
        existing_ids.update(row[0] for row in cursor.fetchall())
        print(f"共有{len(existing_ids)}个产品ID在数据库中找到")
            

        tables = [
            "kb_chunk_version_0724",
            "kb_chunk_version_0906",
            "kb_chunk_version_0906_2",
            "kb_chunk_version_20240919",
            "kb_chunk_version_20241008",
            "kb_chunk_version_20241014",
            "kb_chunk_version_20241016",
            "kb_chunk_version_20241021",
            "kb_chunk_version_all",
            "myba_query_raw_kb_chunk"
        ]
        for table_name in tables:
            query = f"SELECT product_id FROM {table_name} WHERE product_id IN ({id_list_str})"
            # print(query)
            cursor.execute(query, id_list)
            existing_ids.update(row[0] for row in cursor.fetchall())
        print(f"共有{len(existing_ids)}个产品ID在数据库中找到")

    finally:
        cursor.close()
        conn.close()
    
    # 过滤数据
    def filter_row(row):
        valid_products = []
        valid_names = []
        valid_dims = []
        
        for pid, name in zip(row['产品id'], row['产品名称']):
            if pid in existing_ids:
                valid_products.append(pid)
                valid_names.append(name)
        
        return pd.Series([
            row['问题'],
            valid_products,
            valid_names,
            row['维度']
        ])
    
    # 应用过滤

    new_df = df.apply(filter_row, axis=1)
    new_df.columns = df.columns
    
    # 保存结果
    new_df.to_excel(output_file, index=False)
    print(f"处理后的文件已保存到{output_file}")

if __name__ == "__main__":
    # 配置参数（需要修改为你的实际配置）
    db_config = {
        'host': '182.44.1.89',
        'user': 'your_username',
        'password': 'your_password',
        'database': 'your_database'
    }


    
    filter_existing_products(
        input_file='问题+产品.xlsx',
        output_file='处理后的_问题+产品.xlsx',
        db_config=db_config
    )