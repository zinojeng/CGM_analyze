"""Generate AGP variability analysis via OpenAI models."""

from __future__ import annotations

from glucose_analysis import calculate_mage
from dataclasses import dataclass



def _categorize_width(value: float, *, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "moderate"


def _build_envelope_summary(
    iqr_avg: float,
    idr_avg: float,
    iqr_status: str,
    idr_status: str,
    iqr_peak_time: str | None,
    iqr_peak_value: float | None,
    idr_peak_time: str | None,
    idr_peak_value: float | None,
) -> str:
    status_label = {"low": "窄", "moderate": "中等", "high": "偏寬"}
    header = f"**IQR/IDR 觀察**：IQR 約 {iqr_avg:.0f} mg/dL（{status_label[iqr_status]}），IDR 約 {idr_avg:.0f} mg/dL（{status_label[idr_status]}）。"

    details: list[str] = []
    if iqr_peak_time and iqr_peak_value is not None:
        details.append(f"- 最大 IQR 時段：{iqr_peak_time}（約 {iqr_peak_value:.0f} mg/dL）")
    if idr_peak_time and idr_peak_value is not None:
        details.append(f"- 最大 IDR 時段：{idr_peak_time}（約 {idr_peak_value:.0f} mg/dL）")

    if iqr_status == "high" and idr_status == "high":
        lines = [
            "• IQR 與 IDR 均偏寬，顯示治療設定與生活型態同時造成大幅波動。",
            "  - 治療面：檢查基礎/餐前胰島素總量、CEU/BEU 碳水換算因子與矯正因子設定。",
            "  - 行為面：強化餐食覆蓋、規律用餐、運動與飲酒的安排與補償策略。",
        ]
    elif iqr_status == "high":
        lines = [
            "• IQR 偏寬但 IDR 未顯著擴張，推測治療參數需微調。",
            "  - 檢視基礎與餐前胰島素劑量是否貼合需求。",
            "  - 檢查 CEU/BEU 碳水換算因子與矯正因子是否設定過弱或過強。",
            "  - 評估注射時間、作息變動或換班是否頻繁影響整體圖型。",
        ]
    elif idr_status == "high":
        lines = [
            "• IQR 維持穩定但 IDR 偏寬，代表偶發性事件造成尖波。",
            "  - 確認餐食是否完整覆蓋、IMI 注射時機是否合適。",
            "  - 注意不規律用餐、臨時加餐、運動或飲酒是否缺乏對應調整。",
            "  - 若偶有藥物（如類固醇）或壓力事件，需記錄並調整策略。",
        ]
    else:
        lines = [
            "• IQR 與 IDR 均處於窄幅，圖型穩定。",
            "• 維持現行治療與生活模式，持續定期檢視即可。",
        ]

    return "\n".join([header] + details + lines)


def _analyze_agp_envelope(cgm_df) -> tuple[str | None, float | None, float | None, str | None, str | None]:
    if 'Timestamp' not in cgm_df.columns:
        return None, None, None, None, None

    working = cgm_df[['Timestamp', 'Sensor Glucose (mg/dL)']].dropna()
    if working.empty:
        return None, None, None, None, None

    working = working.copy()
    working['TimeOfDay'] = working['Timestamp'].dt.strftime('%H:%M')
    grouped = working.groupby('TimeOfDay')['Sensor Glucose (mg/dL)']
    percentiles = grouped.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    if percentiles.empty:
        return None, None, None, None, None

    iqr_width = (percentiles[0.75] - percentiles[0.25]).dropna()
    idr_width = (percentiles[0.95] - percentiles[0.05]).dropna()
    if iqr_width.empty or idr_width.empty:
        return None, None, None, None, None, None, None

    iqr_avg = float(iqr_width.mean())
    idr_avg = float(idr_width.mean())

    iqr_status = _categorize_width(iqr_avg, low=30.0, high=45.0)
    idr_status = _categorize_width(idr_avg, low=80.0, high=120.0)
    iqr_peak_idx = iqr_width.idxmax() if not iqr_width.empty else None
    idr_peak_idx = idr_width.idxmax() if not idr_width.empty else None
    iqr_peak_time = str(iqr_peak_idx) if iqr_peak_idx is not None else None
    idr_peak_time = str(idr_peak_idx) if idr_peak_idx is not None else None
    iqr_peak_value = float(iqr_width.loc[iqr_peak_idx]) if iqr_peak_idx is not None else None
    idr_peak_value = float(idr_width.loc[idr_peak_idx]) if idr_peak_idx is not None else None

    summary = _build_envelope_summary(
        iqr_avg,
        idr_avg,
        iqr_status,
        idr_status,
        iqr_peak_time,
        iqr_peak_value,
        idr_peak_time,
        idr_peak_value,
    )
    return summary, iqr_avg, idr_avg, iqr_status, idr_status, iqr_peak_time, idr_peak_time

from llm_utils import DEFAULT_FALLBACK_MODELS, LLMCallResult, request_llm_text


@dataclass
class AGPVariabilityResult:
    agp_analysis: str
    hypo_hyper_analysis: str
    sd: float
    cv: float
    mage: float
    notice: str | None = None
    envelope_summary: str | None = None

    def __iter__(self):
        yield self.agp_analysis
        yield self.hypo_hyper_analysis
        yield self.sd
        yield self.cv
        yield self.mage
        yield self.envelope_summary


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

    envelope_summary, _, _, _, _, iqr_peak_time, idr_peak_time = _analyze_agp_envelope(cgm_df)
    if envelope_summary is None:
        envelope_summary = "IQR/IDR 數據不足，無法提供圖型判讀。"
        iqr_peak_time = idr_peak_time = None

    prompt = f"""
    請分析以下血糖監測數據的變異性指標：

    標準差 (SD): {sd:.2f}
    變異係數 (CV): {cv:.2f}%

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
        return AGPVariabilityResult(error_message, "", sd, cv, mage, notice=None, envelope_summary=envelope_summary)

    analysis = analysis or ""

    try:
        parts = analysis.split("\n\n")
        agp_analysis = parts[0] if parts else "無法獲取 AGP 分析"
        hypo_hyper_analysis = parts[1] if len(parts) > 1 else "無法獲取低血糖和高血糖分析"
    except Exception as error:  # pylint: disable=broad-except
        return AGPVariabilityResult(f"分析結果處理錯誤: {str(error)}", "", sd, cv, mage, notice=None, envelope_summary=envelope_summary)

    cleaned_notice = notice.strip() if notice else None
    agp_analysis = _strip_notice_prefix(agp_analysis, cleaned_notice)
    hypo_hyper_analysis = _strip_notice_prefix(hypo_hyper_analysis, cleaned_notice)

    return AGPVariabilityResult(agp_analysis, hypo_hyper_analysis, sd, cv, mage, cleaned_notice, envelope_summary=envelope_summary)
