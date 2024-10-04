import os
import numpy as np
import pandas as pd
import fitz  # PyMuPDF 用於處理 PDF 文件
from sentence_transformers import SentenceTransformer
import faiss
from glucose_analysis import calculate_metrics, create_agp, create_daily_clusters  # 從 glucose_analysis.py 匯入所需的函數
import openai  # 確保在文件頂部添加這行導入

class ReferenceDatabase:
    def __init__(self, directory_path):
        """
        初始化 ReferenceDatabase，從指定目錄下的 PDF 文件中提取文本並將其嵌入向量空間中。

        :param directory_path: 包含 PDF 文件的目錄路徑。
        """
        self.directory_path = directory_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.vectors = []

        # 從目錄下的 PDF 中提取文本
        self._extract_text_from_pdfs()

        # 將提取的文本嵌入向量空間
        self._create_vector_index()

    def _extract_text_from_pdfs(self):
        """
        使用 PyMuPDF 從指定目錄下的所有 PDF 文件中提取文本。
        """
        for file_name in os.listdir(self.directory_path):
            if file_name.endswith('.pdf'):
                file_path = os.path.join(self.directory_path, file_name)
                with fitz.open(file_path) as pdf_document:
                    for page_num in range(pdf_document.page_count):
                        page = pdf_document.load_page(page_num)
                        text = page.get_text()
                        self.documents.append(text)

    def _create_vector_index(self):
        """
        將提取的文本嵌入為向量，並使用 FAISS 建立向量索引。
        """
        # 使用模型將文檔文本嵌入向量
        self.vectors = self.model.encode(self.documents)

        # 建立 FAISS 索引
        dimension = self.vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.vectors)

    def search(self, query, k=3):
        """
        根據查詢返回最相關的參考文檔。

        :param query: 要查詢的字串。
        :param k: 要返回的最相關文檔數量。
        :return: 最相關的文檔列表。
        """
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector, k)
        return [self.documents[idx] for idx in indices[0]]

class GRIAnalyzer:
    def __init__(self, glucose_data, vector_database):
        """
        使用血糖數據和向量資料庫初始化 GRIAnalyzer。

        :param glucose_data: 包含 'timestamp' 和 'glucose' 欄位的 pandas DataFrame。
        :param vector_database: 用於檢索信息的向量資料庫。
        """
        self.glucose_data = glucose_data
        self.vector_database = vector_database
        # 加載預訓練的句子轉換模型，用於向量嵌入
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # 不再需要初始化 GlucoseAnalysis

    def retrieve_references(self, query):
        """
        根據查詢從向量資料庫中檢索參考信息。

        :param query: 查詢字串，用於搜索相關信息。
        :return: 檢索到的參考文本。
        """
        # 使用 ReferenceDatabase 搜索最相關的文檔
        return self.vector_database.search(query, k=3)

    def analyze_with_references(self):
        """
        執行完整分析，包括範圍內時間、GRI、血糖指標以及相關參考資料。

        :return: 包含所有分析結果和相關參考資料的字典。
        """
        # 執行基本的血糖分析並計算 GRI
        analysis = self.analyze()
        # 定查詢以檢索相關的參考文章
        query = "glucose metrics and their clinical significance"
        # 根據查詢檢索參考信息
        references = self.retrieve_references(query)
        # 將檢索到的參考資添加到分析結果中
        analysis['References'] = references
        return analysis

    def analyze(self):
        """
        執行完整分析，包括範圍內時間、GRI 和血糖指標。

        :return: 包含所有分析結果的字典。
        """
        analysis = {}
        # 直接使用導入的函數
        analysis.update(calculate_metrics(self.glucose_data))
        # GRI 已經包含在 calculate_metrics 中，不需要單獨計算
        # 範圍內時間也已經包含在 calculate_metrics 中
        return analysis

    def gpt4_analysis(self, openai_api_key, query):
        """
        使用 OpenAI 的 GPT-4 API 進行進一步的文本分析。

        :param openai_api_key: 用於驗證的 OpenAI API 金鑰。
        :param query: 查詢字串，用於 GPT-4 分析。
        :return: GPT-4 的分析結果。
        """
        openai.api_key = openai_api_key

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in glucose analysis and diabetes management."},
                {"role": "user", "content": query}
            ],
            max_tokens=500,
            temperature=0.7,
            top_p=1,
            n=1
        )
        
        return response.choices[0].message['content']

def perform_gri_rag_analysis(cgm_df, reference_db):
    gri_analyzer = GRIAnalyzer(cgm_df, reference_db)
    analysis_result = gri_analyzer.analyze()
    
    # 只提取 GRI 相關的指標，並四捨五入到小數點後一位
    gri_value = round(analysis_result.get('GRI', 0), 1)
    
    # 構建針對 GRI 的查詢
    query = f"""
    基於血糖風險指數 (GRI) 為 {gri_value}，請提供分析和臨床建議。
    
    請用中文（繁體）精簡準確回答，並包含以下幾點：
    1. GRI 值的含義和臨床重要性
    
    請直接給出專業的分析，以連貫的段落形式呈現。在適當的地方使用粗體來強調重要信息, 不用總結或建議。
    """
    
    # 使用 GPT-4 生成分析結果
    gpt4_analysis = gri_analyzer.gpt4_analysis(openai.api_key, query)
    
    # 直接返回完整的分析結果，使用 Markdown 格式
    return f"""

**GRI 值：{gri_value}**

{gpt4_analysis}
"""