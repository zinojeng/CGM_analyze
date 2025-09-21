import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

from llm_utils import DEFAULT_FALLBACK_MODELS, LLMCallResult, request_llm_text


@dataclass
class GRIRAGResult:
    content: str
    notice: str | None = None


def _format_gri_fallback_notice(primary_model: str, result: LLMCallResult) -> str:
    failure_reason = result.failures[0][1] if result.failures else "未知錯誤"
    return (
        f"[注意] 主模型 {primary_model} 調用失敗：{failure_reason}\n"
        f"已自動改用 {result.model_used}."
    )


def _strip_notice_prefix(text: str, notice: str | None) -> str:
    if not text or not notice:
        return text
    cleaned_notice = notice.strip()
    if not cleaned_notice:
        return text
    if text.startswith(cleaned_notice):
        text = text[len(cleaned_notice):].lstrip()
    return text


class ReferenceDatabase:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.file_path = os.path.join(directory_path, "Glycemia Risk Index.pdf")
        self.extracted_text = ""
        try:
            self._extract_text_from_pdf()
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.documents = self._split_text_into_chunks()
            self._create_index()
        except FileNotFoundError:
            print(f"Warning: Reference file not found at {self.file_path}")
            self.documents = []
            self.model = None

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
        self.index.build(10)

    def search(self, query, k=3):
        if not self.documents or not self.model:
            return ["Reference database not available"]
        query_vector = self.model.encode([query])[0]
        indices = self.index.get_nns_by_vector(query_vector, k, include_distances=False)
        return [self.documents[idx] for idx in indices]


class GRIAnalyzer:
    def __init__(self, cgm_df, reference_db=None):
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


def perform_gri_rag_analysis(gri_analysis, api_key=None, model_name="o3") -> GRIRAGResult:
    """
    使用 RAG 方法分析 GRI 數據。
    """
    prompt = f"""
    請分析以下 GRI (Glycemic Risk Index) 數據：
    {gri_analysis}

    請提供專業的解釋和建議，包括：
    1. GRI 指標的含義
    2. 當前數值的風險評估
    3. 改善建議

    請用中文（繁體）回答，並確保回答準確、專業且易於理解。
    """

    messages = [
        {"role": "system", "content": "You are a diabetes management expert analyzing GRI data."},
        {"role": "user", "content": prompt}
    ]

    content, error_message, _, notice = request_llm_text(
        api_key,
        primary_model=model_name,
        messages=messages,
        max_tokens=1000,
        fallback_models=DEFAULT_FALLBACK_MODELS,
        missing_key_error="錯誤：需要提供 API 金鑰",
        error_formatter=lambda model, exc: f"OpenAI API 調用錯誤 ({model}): {exc}",
        fallback_notice_formatter=_format_gri_fallback_notice,
    )

    cleaned_notice = notice.strip() if notice else None
    if error_message:
        return GRIRAGResult(content=error_message, notice=cleaned_notice)

    clean_content = _strip_notice_prefix(content or "", cleaned_notice)
    return GRIRAGResult(content=clean_content, notice=cleaned_notice)
