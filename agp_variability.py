"""Generate AGP variability analysis via OpenAI models."""

from __future__ import annotations

import numpy as np

from glucose_analysis import calculate_mage
from dataclasses import dataclass

from llm_utils import DEFAULT_FALLBACK_MODELS, LLMCallResult, request_llm_text


@dataclass
class AGPVariabilityResult:
    agp_analysis: str
    hypo_hyper_analysis: str
    sd: float
    cv: float
    mage: float
    notice: str | None = None

    def __iter__(self):
        yield self.agp_analysis
        yield self.hypo_hyper_analysis
        yield self.sd
        yield self.cv
        yield self.mage


def _strip_notice_prefix(text: str, notice: str | None) -> str:
    if not text or not notice:
        return text
    cleaned_notice = notice.strip()
    if not cleaned_notice:
        return text
    if text.startswith(cleaned_notice):
        text = text[len(cleaned_notice):].lstrip()
    return text


def _format_fallback_notice(primary_model: str, result: LLMCallResult) -> str:
    """Return a localized notice when the primary model fails."""

    failure_reason = result.failures[0][1] if result.failures else "未知錯誤"
    return (
        f"[注意] 主模型 {primary_model} 調用失敗：{failure_reason}\n"
        f"已自動改用 {result.model_used}."
    )


def agp_variability(cgm_df, api_key=None, model_name="o3"):
    """分析 CGM 數據的 AGP 變異性並回傳解析結果。"""

    glucose_values = cgm_df['Sensor Glucose (mg/dL)'].dropna()
    sd = glucose_values.std()
    cv = (sd / glucose_values.mean()) * 100
    mage = calculate_mage(glucose_values)

    prompt = f"""
    請分析以下血糖監測數據的變異性指標：

    標準差 (SD): {sd:.2f}
    變異係數 (CV): {cv:.2f}%
    平均血糖波動幅度 (MAGE): {mage:.2f}

    請提供：
    1. AGP 整體分析
    2. 低血糖和高血糖風險分析

    請用中文（繁體）回答，並確保回答準確、專業且易於理解。
    """

    messages = [
        {"role": "system", "content": "You are a diabetes management expert analyzing CGM data."},
        {"role": "user", "content": prompt},
    ]

    analysis, error_message, _, notice = request_llm_text(
        api_key,
        primary_model=model_name,
        messages=messages,
        max_tokens=1000,
        fallback_models=DEFAULT_FALLBACK_MODELS,
        missing_key_error="錯誤：需要提供 API 金鑰",
        error_formatter=lambda model, exc: f"OpenAI API 調用錯誤 ({model}): {exc}",
        fallback_notice_formatter=_format_fallback_notice,
    )

    if error_message:
        return AGPVariabilityResult(error_message, "", sd, cv, mage, notice=None)

    analysis = analysis or ""

    try:
        parts = analysis.split("\n\n")
        agp_analysis = parts[0] if parts else "無法獲取 AGP 分析"
        hypo_hyper_analysis = parts[1] if len(parts) > 1 else "無法獲取低血糖和高血糖分析"
    except Exception as error:  # pylint: disable=broad-except
        return AGPVariabilityResult(f"分析結果處理錯誤: {str(error)}", "", sd, cv, mage, notice=None)

    cleaned_notice = notice.strip() if notice else None
    agp_analysis = _strip_notice_prefix(agp_analysis, cleaned_notice)
    hypo_hyper_analysis = _strip_notice_prefix(hypo_hyper_analysis, cleaned_notice)

    return AGPVariabilityResult(agp_analysis, hypo_hyper_analysis, sd, cv, mage, cleaned_notice)
