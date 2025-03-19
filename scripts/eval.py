import pandas as pd
import ast
from tqdm import tqdm
from retrieve import RAGSystem  # 假设RAGSystem类在retrieve.py中

class RAGEvaluator:
    def __init__(self, excel_path, rag_system):
        self.df = pd.read_excel(excel_path)
        self.rag = rag_system
        
        # 转换列数据格式
        self.df['产品id'] = self.df['产品id'].apply(ast.literal_eval)
        self.df['产品名称'] = self.df['产品名称'].apply(ast.literal_eval)
        
        # 添加结果列
        for n in [1, 5, 10, 20, 50]:
            self.df[f'前{n}命中数'] = 0
        self.df['filename'] = None 

    def _get_unique_product_ids(self, reranked_results):
        seen_ids = set()
        seen_files = set()
        unique_ids = []
        unique_files = []

        for res in reranked_results:
            # 处理product_id去重
            if res['product_id'] not in seen_ids:
                seen_ids.add(res['product_id'])
                unique_ids.append(res['product_id'])
            
            # 处理filename去重（注意字段名拼写）
            filename = res.get('file_name')  # 与代码中的拼写保持一致
            if filename and filename not in seen_files:
                seen_files.add(filename)
                unique_files.append(filename)
        
        return unique_ids, unique_files

    def _calculate_hits(self, expected_ids, retrieved_ids, top_n):
        actual_top = min(top_n, len(retrieved_ids))
        return len(set(retrieved_ids[:actual_top]) & set(expected_ids))

    def process_all_queries(self, output_path):
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                # 获取召回结果
                search_results = self.rag._search_milvus(row['问题'])
                reranked = self.rag._rerank_results(row['问题'], search_results)
                
                # 获取唯一product_id列表
                unique_ids,unique_files = self._get_unique_product_ids(reranked)
                
                # 计算各top值命中数
                expected = list(map(str, row['产品id']))  # 统一转为字符串格式
                for n in [1, 5, 10, 20, 50]:
                    self.df.at[idx, f'前{n}命中数'] = self._calculate_hits(expected, unique_ids, n)
                
                self.df.at[idx, 'filename'] = str(unique_files)
            
            except Exception as e:
                print(f"处理第{idx}行时发生错误：{str(e)}")
                continue
        
        # 保存结果
        self.df.to_excel(output_path, index=False)
        print(f"评测完成，结果已保存至：{output_path}")

if __name__ == "__main__":
    # 初始化RAG系统（根据实际配置修改参数）
    rag_system = RAGSystem(
        config_path='/data/myba/ragtest/config/ragtest.yaml',
        prompt_config_path='/data/myba/ragtest/config/prompt.yaml'
    )
    
    # 初始化评测器
    evaluator = RAGEvaluator("处理后的_问题+产品.xlsx", rag_system)
    
    # 执行评测并保存结果
    evaluator.process_all_queries("评测结果.xlsx")