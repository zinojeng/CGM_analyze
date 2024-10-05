import os
import numpy as np
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from openai import OpenAI

class ReferenceDatabase:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.file_path = os.path.join(directory_path, "Glycemia Risk Index.pdf")
        self.extracted_text = ""
        self._extract_text_from_pdf()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = self._split_text_into_chunks()
        self._create_index()

    def _extract_text_from_pdf(self):
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                self.extracted_text += page.extract_text()

    def _split_text_into_chunks(self, chunk_size=200):
        words = self.extracted_text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def _create_index(self):
        embeddings = self.model.encode(self.documents)
        self.index = AnnoyIndex(embeddings.shape[1], 'angular')
        for i, embedding in enumerate(embeddings):
            self.index.add_item(i, embedding)
        self.index.build(10)  # 使用 10 棵樹來構建索引

    def search(self, query, k=3):
        query_vector = self.model.encode([query])[0]
        indices = self.index.get_nns_by_vector(query_vector, k, include_distances=False)
        return [self.documents[idx] for idx in indices]

class GRIAnalyzer:
    def __init__(self, cgm_df, reference_db):
        self.cgm_df = cgm_df
        self.reference_db = reference_db
        self.glucose_column = 'Sensor Glucose (mg/dL)'

    def analyze(self):
        glucose_values = pd.to_numeric(self.cgm_df[self.glucose_column], errors='coerce').dropna()
        gri = np.log(glucose_values / 100) ** 2
        mean_gri = gri.mean() * 100
        
        analysis_result = {
            'Mean GRI': mean_gri,
            'Hypoglycemia Component': (glucose_values < 70).sum() / len(glucose_values) * 100,
            'Hyperglycemia Component': (glucose_values > 180).sum() / len(glucose_values) * 100
        }
        
        return analysis_result

def perform_gri_rag_analysis(cgm_df, reference_db, openai_api_key):
    gri_analyzer = GRIAnalyzer(cgm_df, reference_db)
    analysis_result = gri_analyzer.analyze()
    
    gri_value = round(analysis_result.get('Mean GRI', 0), 1)
    
    # 使用 RAG 從參考文獻中檢索相關信息
    query = f"Glycemia Risk Index (GRI) value of {gri_value} interpretation and clinical significance"
    relevant_passages = reference_db.search(query, k=2)
    
    # 構建提示
    prompt = f"""
    Based on a Glycemia Risk Index (GRI) value of {gri_value} and the following relevant information from medical literature:

    {' '.join(relevant_passages)}

    Please provide an analysis and clinical interpretation in Traditional Chinese. Include:
    1. The meaning and clinical importance of this GRI value
    2. Potential implications for the patient's glycemic control
    3. Any recommendations for management or further monitoring

    Present your analysis in a cohesive paragraph format, using bold text to emphasize key points. Do not summarize or provide additional recommendations beyond what's supported by the given information.
    """

    # 使用 GPT-4 生成分析結果
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a diabetes management expert providing analysis based on GRI data and medical literature."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    gpt4_analysis = response.choices[0].message.content
    
    # 返回完整的分析結果，使用 Markdown 格式
    return f"""

**GRI 值：{gri_value}**

{gpt4_analysis}
"""