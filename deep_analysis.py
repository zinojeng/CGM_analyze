import json
import pandas as pd
import numpy as np
from gri_rag import GRIAnalyzer, perform_gri_rag_analysis, ReferenceDatabase
from gri_plotting import plot_gri
import os
from agp_variability import agp_variability
from llm_utils import DEFAULT_FALLBACK_MODELS, LLMCallResult, request_llm_text


def _strip_notice_prefix(text: str | None, notice: str | None) -> str:
    if not text:
        return ""
    cleaned_text = str(text).strip()
    if not notice:
        return cleaned_text
    cleaned_notice = notice.strip()
    if not cleaned_notice:
        return cleaned_text
    if cleaned_text.startswith(cleaned_notice):
        cleaned_text = cleaned_text[len(cleaned_notice):].lstrip()
    return cleaned_text


def _format_fallback_notice_en(primary_model: str, result: LLMCallResult) -> str:
    reason = result.failures[0][1] if result.failures else "unknown error"
    return f"[注意] 主模型 {primary_model} 調用失敗：{reason} 已自動改用 {result.model_used}."


def analyze_insulin_pharmacokinetics(cgm_df, insulin_data):
    if not isinstance(insulin_data, pd.DataFrame):
        if isinstance(insulin_data, list) and len(insulin_data) > 0:
            insulin_data = pd.DataFrame(insulin_data)
        else:
            return "無法進行胰島素藥代動力學分析，因為數據格式不正確。"

    if 'Insulin' not in insulin_data.columns:
        return "無法進行胰島素藥代動力學分析，因為數據中沒有胰島素注射記錄。"

    if 'Timestamp' not in cgm_df.columns:
        cgm_df['Timestamp'] = pd.to_datetime(cgm_df['Date'].astype(str) + ' ' + cgm_df['Time'].astype(str))
    if 'Timestamp' not in insulin_data.columns:
        insulin_data['Timestamp'] = pd.to_datetime(insulin_data['Date'].astype(str) + ' ' + insulin_data['Time'].astype(str))

    merged_data = pd.merge_asof(
        cgm_df.sort_values('Timestamp'),
        insulin_data.sort_values('Timestamp'),
        on='Timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )

    merged_data['Glucose_Change'] = merged_data['Sensor Glucose (mg/dL)'].diff()
    merged_data['Time_Since_Insulin'] = (
        merged_data['Timestamp'] - merged_data['Timestamp'].where(merged_data['Insulin'].notna()).ffill()
    ).dt.total_seconds() / 3600

    def safe_min(series):
        return series.min() if not series.isna().all() else np.nan

    analysis_results = merged_data.groupby('Timestamp').apply(
        lambda group: pd.Series({
            'Action_Time': safe_min(group['Time_Since_Insulin']),
            'Peak_Time': group['Time_Since_Insulin'][group['Glucose_Change'].idxmin()] if not group['Glucose_Change'].isna().all() else np.nan,
            'Duration': group['Time_Since_Insulin'].max(),
            'Insulin_Sensitivity': group['Glucose_Change'].min() / group['Insulin'].max() if group['Insulin'].max() > 0 else np.nan
        })
    )

    return analysis_results.mean()


def analyze_meal_impact(cgm_df, meal_data):
    if not isinstance(meal_data, pd.DataFrame) or meal_data.empty:
        return "無法分析飲食對血糖的影響，因為沒有有效的飲食數據。"

    if 'Timestamp' not in cgm_df.columns:
        cgm_df['Timestamp'] = pd.to_datetime(cgm_df['Date'].astype(str) + ' ' + cgm_df['Time'].astype(str))
    if 'Timestamp' not in meal_data.columns:
        meal_data['Timestamp'] = pd.to_datetime(meal_data['Date'].astype(str) + ' ' + meal_data['Time'].astype(str))

    merged_data = pd.merge_asof(
        cgm_df.sort_values('Timestamp'),
        meal_data.sort_values('Timestamp'),
        on='Timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )

    merged_data['Glucose_Change'] = merged_data['Sensor Glucose (mg/dL)'].diff()
    merged_data['Time_Since_Meal'] = (
        merged_data['Timestamp'] - merged_data['Timestamp'].where(merged_data.index.isin(meal_data.index)).ffill()
    ).dt.total_seconds() / 3600

    def safe_max(series):
        return series.max() if not series.isna().all() else np.nan

    analysis_results = merged_data.groupby('Timestamp').apply(
        lambda group: pd.Series({
            'Peak_Glucose_Time': group['Time_Since_Meal'][group['Glucose_Change'].idxmax()] if not group['Glucose_Change'].isna().all() else np.nan,
            'Peak_Glucose_Change': safe_max(group['Glucose_Change']),
            'Return_To_Baseline_Time': group['Time_Since_Meal'][group['Glucose_Change'].abs().idxmin()] if not group['Glucose_Change'].isna().all() else np.nan
        })
    )

    return analysis_results.mean()


def _ensure_serializable(value):
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return {k: (None if (isinstance(v, float) and np.isnan(v)) else _ensure_serializable(v)) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_ensure_serializable(v) for v in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    return value


def _format_json_block(label, data):
    try:
        serializable = _ensure_serializable(data)
        payload = json.dumps(serializable, ensure_ascii=False, indent=2)
    except Exception:  # pylint: disable=broad-except
        payload = str(data)
    return f"### {label}\n```json\n{payload}\n```"


def _format_percentage(value, decimals=1):
    if value is None:
        return "資料不足"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "資料不足"
    number = float(value)
    if number <= 1:
        number *= 100
    format_spec = f".{decimals}f"
    return f"{number:{format_spec}}%"


def _format_float(value, unit="", decimals=1):
    if value is None:
        return "資料不足"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "資料不足"
    formatted = f"{float(value):.{decimals}f}"
    return f"{formatted}{unit}" if unit else formatted


def _format_time_label(hour_value):
    if hour_value is None:
        return None
    if isinstance(hour_value, (float, np.floating)) and np.isnan(hour_value):
        return None
    total_minutes = int(round(float(hour_value) * 60))
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _summarize_cgm_metrics(cgm_metrics, profile_config=None):
    if not isinstance(cgm_metrics, dict) or not cgm_metrics:
        return "尚無有效 CGM 指標資料。"

    target_range = profile_config['target_range'] if profile_config else (70, 180)
    lines = []

    mean_glucose = cgm_metrics.get("Mean Glucose (mg/dL)")
    if mean_glucose is not None:
        lines.append(
            f"- 平均血糖約 **{_format_float(mean_glucose, ' mg/dL', decimals=1)}**，目標區間為 `{target_range[0]}-{target_range[1]} mg/dL`。"
        )

    tir = cgm_metrics.get("TIR (70-180 mg/dL)")
    if tir is not None:
        lines.append(f"- 目標範圍內時間 (TIR) 為 **{_format_percentage(tir)}**。")

    low = (cgm_metrics.get("Low (54-<70 mg/dL)") or 0) + (cgm_metrics.get("VLow (<54 mg/dL)") or 0)
    if low:
        lines.append(f"- 低血糖時間合計 **{_format_percentage(low)}**，需要留意是否符合低血糖管理門檻。")

    high = (cgm_metrics.get("High (>180-250 mg/dL)") or 0) + (cgm_metrics.get("VHigh (>250 mg/dL)") or 0)
    if high:
        lines.append(f"- 高血糖時間合計 **{_format_percentage(high)}**，顯示整體高血糖暴露偏多。")

    cv = cgm_metrics.get("CV")
    if cv is not None:
        lines.append(f"- 變異係數 (CV) 為 **{_format_percentage(cv)}**，用以評估波動度是否接近 36% 門檻。")

    gmi = cgm_metrics.get("GMI")
    if gmi is not None:
        lines.append(f"- GMI (估算 A1c) 約 **{_format_float(gmi, decimals=2)}%**。")

    mage = cgm_metrics.get("MAGE")
    if mage is not None:
        lines.append(f"- MAGE (平均血糖波動幅度) 為 **{_format_float(mage)}**，可對應波動是否劇烈。")

    if not lines:
        return "尚無足夠的 CGM 指標可供分析。"

    return "\n".join(lines)


def _summarize_insulin_stats(insulin_stats):
    if not isinstance(insulin_stats, dict) or not insulin_stats:
        return "尚無有效的胰島素統計資料。"

    lines = []
    for category, data in insulin_stats.items():
        if not isinstance(data, dict) or not data:
            continue
        avg_dose = data.get('平均劑量')
        count = data.get('注射次數')
        min_dose = data.get('最小劑量')
        max_dose = data.get('最大劑量')

        headline = f"- **{category}**："
        details = []
        if avg_dose is not None:
            details.append(f"平均 {avg_dose} 單位")
        if count is not None:
            details.append(f"共 {count} 次")
        if min_dose is not None and max_dose is not None:
            details.append(f"範圍 {min_dose}-{max_dose} 單位")
        headline += "，".join(details) if details else "資料待補。"
        lines.append(headline)

        common_times = data.get('常見注射時間')
        if common_times:
            formatted_slots = []
            for entry in common_times:
                if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                    continue
                window = _format_time_label(entry[0])
                avg = entry[1]
                freq = entry[2]
                if window is None:
                    continue
                formatted_slots.append(f"{window} 附近 ~ {avg} 單位 (共 {freq} 次)")
            if formatted_slots:
                lines.append("  • 常見時段：" + "；".join(formatted_slots))

        grouped = data.get('劑量分組')
        if grouped and not common_times:
            formatted_groups = []
            for entry in grouped:
                if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                    continue
                window = _format_time_label(entry[0])
                avg = entry[1]
                freq = entry[2]
                if window is None:
                    continue
                formatted_groups.append(f"{window} 附近 ~ {avg} 單位 (共 {freq} 次)")
            if formatted_groups:
                lines.append("  • 劑量分組：" + "；".join(formatted_groups))

        for name, info in data.items():
            if name in {'平均劑量', '注射次數', '最小劑量', '最大劑量', '常見注射時間', '劑量分組'}:
                continue
            if isinstance(info, dict):
                avg = info.get('平均劑量')
                freq = info.get('注射次數')
                lines.append(
                    f"  • {name}：平均 {avg} 單位，共 {freq} 次"
                )

    if not lines:
        return "目前無法彙整胰島素注射趨勢，請確認事件資料是否完整。"

    return "\n".join(lines)


def _summarize_agp_variability(agp_analysis, hypo_hyper_analysis, sd, cv, mage):
    lines = []
    if agp_analysis:
        lines.append(f"- AGP 評析：{agp_analysis}")
    if hypo_hyper_analysis:
        lines.append(f"- 低/高血糖風險：{hypo_hyper_analysis}")
    sd_text = _format_float(sd, ' mg/dL')
    cv_text = _format_percentage(cv)
    mage_text = _format_float(mage)
    lines.append(f"- 波動指標：SD {sd_text}｜CV {cv_text}｜MAGE {mage_text}")
    return "\n".join(lines)


def _summarize_gri(gri_analysis, gri_gpt4_analysis):
    if not isinstance(gri_analysis, dict) or not gri_analysis:
        return "尚無 GRI 指標可供分析。"

    mean_gri = gri_analysis.get('Mean GRI')
    hypo_component = gri_analysis.get('Hypoglycemia Component')
    hyper_component = gri_analysis.get('Hyperglycemia Component')

    lines = [
        f"- Mean GRI 約 **{_format_float(mean_gri)}**，綜合評估整體風險。",
        f"- 低血糖風險成分：{_format_percentage(hypo_component)}，高血糖成分：{_format_percentage(hyper_component)}。"
    ]

    interpretation = gri_gpt4_analysis.strip() if isinstance(gri_gpt4_analysis, str) else ""
    if interpretation:
        lines.append(f"- LLM 解讀：{interpretation}")
    else:
        lines.append("- 尚未取得 LLM 解讀，可檢查 RAG 模組或 API 呼叫狀態。")

    return "\n".join(lines)


def _summarize_insulin_pharmacokinetics(insulin_pharmacokinetics):
    if isinstance(insulin_pharmacokinetics, pd.Series):
        data = insulin_pharmacokinetics.to_dict()
    elif isinstance(insulin_pharmacokinetics, dict):
        data = insulin_pharmacokinetics
    else:
        return str(insulin_pharmacokinetics) if insulin_pharmacokinetics else "尚無足夠資料解析胰島素藥代特徵。"

    action = data.get('Action_Time')
    peak = data.get('Peak_Time')
    duration = data.get('Duration')
    sensitivity = data.get('Insulin_Sensitivity')

    lines = []
    if action is not None:
        lines.append(f"- 平均起始作用時間約 `{_format_float(action, ' 小時')}`。")
    if peak is not None:
        lines.append(f"- 推估峰值時間落在 `{_format_float(peak, ' 小時')}`。")
    if duration is not None:
        lines.append(f"- 作用持續時間約 `{_format_float(duration, ' 小時')}`。")
    if sensitivity is not None and not (isinstance(sensitivity, float) and np.isnan(sensitivity)):
        lines.append(f"- 估計胰島素敏感度：{_format_float(sensitivity)} (血糖變化 / 劑量)。")

    if not lines:
        return "尚無足夠資料解析胰島素藥代特徵。"
    return "\n".join(lines)


def _summarize_meal_impact(meal_impact):
    if isinstance(meal_impact, pd.Series):
        data = meal_impact.to_dict()
    elif isinstance(meal_impact, dict):
        data = meal_impact
    else:
        return str(meal_impact) if meal_impact else "尚無足夠資料分析餐食影響。"

    peak_time = data.get('Peak_Glucose_Time')
    peak_change = data.get('Peak_Glucose_Change')
    return_time = data.get('Return_To_Baseline_Time')

    lines = []
    if peak_time is not None and not (isinstance(peak_time, float) and np.isnan(peak_time)):
        lines.append(f"- 血糖峰值通常出現在進食後 `{_format_float(peak_time, ' 小時')}`。")
    if peak_change is not None and not (isinstance(peak_change, float) and np.isnan(peak_change)):
        lines.append(f"- 峰值幅度約 `{_format_float(peak_change, ' mg/dL', decimals=1)}`。")
    if return_time is not None and not (isinstance(return_time, float) and np.isnan(return_time)):
        lines.append(f"- 回到基線約需 `{_format_float(return_time, ' 小時')}`。")

    if not lines:
        return "尚無足夠資料分析餐食影響。"
    return "\n".join(lines)


def _summarize_profile_guidance(profile_config):
    if not profile_config:
        return "尚未載入族群建議設定。"

    target_lower, target_upper = profile_config.get('target_range', (70, 180))
    display_name = profile_config.get('display_name', '自訂族群')
    summary = profile_config.get('targets_summary', '')
    recommendation = profile_config.get('recommendation', '')

    lines = [
        f"- 族群設定：**{display_name}**",
        f"- 建議目標範圍：`{target_lower}-{target_upper} mg/dL`",
    ]
    if summary:
        lines.append(f"- 目標重點：{summary}")
    if recommendation:
        lines.append(f"- 建議重點：{recommendation}")
    return "\n".join(lines)


def insulin_glucose_interaction(
    cgm_metrics,
    insulin_stats,
    profile_config=None,
    openai_api_key=None,
    model_name="o3"
):
    profile_name = profile_config['display_name'] if profile_config else "一般族群"
    target_range = profile_config['target_range'] if profile_config else (70, 180)
    targets_summary = profile_config.get('targets_summary', '') if profile_config else ""

    prompt = f"""
    Patient profile: {profile_name}
    Recommended target range: {target_range[0]}-{target_range[1]} mg/dL
    Key targets: {targets_summary}

    CGM 指標：
    {cgm_metrics}

    胰島素統計：
    {insulin_stats}

    請提供以下方面的分析：
    1. 血糖控制的整體評估，相對於上述目標是否達標
    2. 胰島素使用的效果與優化建議
    3. 胰島素劑量與血糖反應的關聯性與改善方向

    請用繁體中文回答，語氣專業、具體且容易執行。
    """

    messages = [
        {
            "role": "system",
            "content": "You are a diabetes management expert providing analysis based on CGM and insulin data."
        },
        {"role": "user", "content": prompt}
    ]

    content, error_message, _, notice = request_llm_text(
        openai_api_key,
        primary_model=model_name,
        messages=messages,
        max_tokens=1024,
        fallback_models=DEFAULT_FALLBACK_MODELS,
        missing_key_error="錯誤：使用 OpenAI 模型需要提供 OpenAI API 金鑰",
        error_formatter=lambda model, exc: f"分析時發生錯誤 ({model}): {exc}",
        fallback_notice_formatter=_format_fallback_notice_en,
    )

    if error_message:
        cleaned_notice = notice.strip() if notice else None
        return error_message, cleaned_notice

    cleaned_notice = notice.strip() if notice else None
    normalized_content = _strip_notice_prefix(content, cleaned_notice)
    return normalized_content, cleaned_notice


def generate_integrated_summary(
    cgm_metrics,
    insulin_stats,
    agp_analysis,
    hypo_hyper_analysis,
    sd,
    cv,
    mage,
    gri_analysis,
    gri_gpt4_analysis,
    insulin_glucose_analysis,
    insulin_pharmacokinetics,
    meal_impact,
    profile_config=None,
    openai_api_key=None,
    model_name="o3",
    cgm_summary_text=None,
    insulin_summary_text=None,
    agp_summary_text=None,
    gri_summary_text=None,
    insulin_pk_summary_text=None,
    meal_summary_text=None
):
    profile_name = profile_config['display_name'] if profile_config else "一般族群"
    target_range = profile_config['target_range'] if profile_config else (70, 180)
    targets_summary = profile_config.get('targets_summary', '') if profile_config else ""

    cgm_block = ""
    if cgm_summary_text:
        cgm_block += f"### CGM 指標摘要 (文字)\n{cgm_summary_text}\n\n"
    cgm_block += _format_json_block("CGM 指標 (JSON)", cgm_metrics)

    insulin_block = ""
    if insulin_summary_text:
        insulin_block += f"### 胰島素統計摘要 (文字)\n{insulin_summary_text}\n\n"
    insulin_block += _format_json_block("胰島素統計 (JSON)", insulin_stats)

    agp_block = "### AGP 與變異解析\n"
    if agp_summary_text:
        agp_block += f"{agp_summary_text}\n\n"
    sd_text = _format_float(sd, ' mg/dL')
    cv_text = _format_percentage(cv)
    mage_text = _format_float(mage)
    agp_block += (
        f"- 原始文字解析：{agp_analysis}\n"
        f"- 低/高血糖評估：{hypo_hyper_analysis}\n"
        f"- SD: `{sd_text}` | CV: `{cv_text}` | MAGE: `{mage_text}`"
    )

    gri_block = "### GRI 解析\n"
    if gri_summary_text:
        gri_block += f"{gri_summary_text}\n\n"
    gri_block += _format_json_block("GRI 數值", gri_analysis)
    gri_interpret_text = (
        gri_gpt4_analysis.strip()
        if isinstance(gri_gpt4_analysis, str) and gri_gpt4_analysis.strip()
        else "尚未取得 LLM 解讀"
    )
    gri_block += f"\n- LLM 解讀：{gri_interpret_text}"

    insulin_pk_block = ""
    if insulin_pk_summary_text:
        insulin_pk_block += f"### 胰島素藥代摘要 (文字)\n{insulin_pk_summary_text}\n\n"
    insulin_pk_block += _format_json_block("胰島素藥代特徵 (JSON)", insulin_pharmacokinetics)

    meal_block = ""
    if meal_summary_text:
        meal_block += f"### 餐食影響摘要 (文字)\n{meal_summary_text}\n\n"
    meal_block += _format_json_block("餐食影響 (JSON)", meal_impact)

    base_prompt = (
        f"患者族群：{profile_name}\n"
        f"建議目標範圍：{target_range[0]}-{target_range[1]} mg/dL\n"
        f"目標摘要：{targets_summary}\n\n"
        f"以下為整合後的分析輸入：\n\n"
        f"{cgm_block}\n\n"
        f"{insulin_block}\n\n"
        f"{agp_block}\n\n"
        f"{gri_block}\n\n"
        f"### 胰島素與血糖互動\n{str(insulin_glucose_analysis)}\n\n"
        f"{insulin_pk_block}\n\n"
        f"{meal_block}"
    )

    instructions = (
        "請以繁體中文提供整合分析，格式要求：\n"
        "- 以二級標題 (##) 或三級標題 (###) 組織段落。\n"
        "- 段落順序依序為：整體控制現況、主要風險與可能成因、胰島素調整建議、餐食與生活型態洞察、優先行動清單、後續監測指引。\n"
        "- 優先行動清單需列出 3-5 點，標註優先等級與建議時程 (例如：1-2 週內)。\n"
        "- 重要數值請使用粗體或反引號標示，並清楚引用來源指標。\n"
        "- 若觀察到資料不足或需要人工確認的地方，請於監測指引段落標註。"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior diabetes educator who synthesizes CGM, insulin, and lifestyle data "
                "to create clear, actionable coaching plans for clinicians and patients."
            )
        },
        {
            "role": "user",
            "content": f"{instructions}\n\n{base_prompt}"
        }
    ]

    summary, error_message, _, notice = request_llm_text(
        openai_api_key,
        primary_model=model_name,
        messages=messages,
        max_tokens=1200,
        fallback_models=DEFAULT_FALLBACK_MODELS,
        missing_key_error="未提供 OpenAI API 金鑰，僅能提供基本指標彙整。請檢視各區塊資料後自行整合建議。",
        error_formatter=lambda model, exc: f"整合分析產生失敗 ({model}): {exc}",
        fallback_notice_formatter=_format_fallback_notice_en,
    )

    if error_message:
        cleaned_notice = notice.strip() if notice else None
        return error_message, cleaned_notice

    cleaned_notice = notice.strip() if notice else None
    normalized_summary = _strip_notice_prefix(summary, cleaned_notice)
    return normalized_summary, cleaned_notice


def perform_deep_analysis(
    cgm_df,
    insulin_data,
    meal_data,
    cgm_metrics,
    insulin_stats,
    agp_analysis,
    hypo_hyper_analysis,
    sd,
    cv,
    mage,
    gri_analysis,
    gri_gpt4_analysis,
    openai_api_key=None,
    model_name="o3",
    profile_config=None,
    agp_notice=None,
    gri_notice=None
):
    insulin_pharmacokinetics = analyze_insulin_pharmacokinetics(cgm_df, insulin_data)
    meal_impact = analyze_meal_impact(cgm_df, meal_data)

    insulin_glucose_analysis, insulin_notice = insulin_glucose_interaction(
        cgm_metrics,
        insulin_stats,
        profile_config,
        openai_api_key,
        model_name
    )

    cgm_summary = _summarize_cgm_metrics(cgm_metrics, profile_config)
    insulin_summary = _summarize_insulin_stats(insulin_stats)
    agp_summary = _summarize_agp_variability(agp_analysis, hypo_hyper_analysis, sd, cv, mage)
    gri_summary = _summarize_gri(gri_analysis, gri_gpt4_analysis)
    insulin_pk_summary = _summarize_insulin_pharmacokinetics(insulin_pharmacokinetics)
    meal_summary = _summarize_meal_impact(meal_impact)

    integrated_summary, integrated_notice = generate_integrated_summary(
        cgm_metrics=cgm_metrics,
        insulin_stats=insulin_stats,
        agp_analysis=agp_analysis,
        hypo_hyper_analysis=hypo_hyper_analysis,
        sd=sd,
        cv=cv,
        mage=mage,
        gri_analysis=gri_analysis,
        gri_gpt4_analysis=gri_gpt4_analysis,
        insulin_glucose_analysis=insulin_glucose_analysis,
        insulin_pharmacokinetics=insulin_pharmacokinetics,
        meal_impact=meal_impact,
        profile_config=profile_config,
        openai_api_key=openai_api_key,
        model_name=model_name,
        cgm_summary_text=cgm_summary,
        insulin_summary_text=insulin_summary,
        agp_summary_text=agp_summary,
        gri_summary_text=gri_summary,
        insulin_pk_summary_text=insulin_pk_summary,
        meal_summary_text=meal_summary
    )

    deep_analysis_result = {
        "整合總結": integrated_summary,
        "CGM 指標摘要": cgm_summary,
        "胰島素使用重點": insulin_summary,
        "AGP 與變異分析": agp_summary,
        "GRI 解析摘要": gri_summary,
        "胰島素與血糖互動解析": insulin_glucose_analysis,
        "胰島素藥代觀察": insulin_pk_summary,
        "餐食影響摘要": meal_summary,
    }

    if profile_config:
        deep_analysis_result['Patient Profile Guidance'] = _summarize_profile_guidance(profile_config)

    notice_groups: dict[str, list[str]] = {}

    def _append_notice(label: str, text: str | None):
        if not text:
            return
        cleaned = text.strip()
        if cleaned:
            notice_groups.setdefault(cleaned, []).append(label)

    _append_notice("AGP 變異分析", agp_notice)
    _append_notice("GRI RAG 解讀", gri_notice)
    _append_notice("胰島素與血糖互動解析", insulin_notice)
    _append_notice("整合總結", integrated_notice)

    if notice_groups:
        lines = []
        for message, labels in notice_groups.items():
            label_text = "、".join(labels)
            lines.append(f"- {label_text}：{message}")
        deep_analysis_result['LLM 調用提示'] = "\n".join(lines)

    return deep_analysis_result
