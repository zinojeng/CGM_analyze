import pandas as pd
import numpy as np
import openai  # 修改這行
from gri_rag import perform_gri_rag_analysis, ReferenceDatabase

def analyze_insulin_pharmacokinetics(cgm_df, insulin_data):
    if not isinstance(insulin_data, pd.DataFrame):
        if isinstance(insulin_data, list) and len(insulin_data) > 0:
            insulin_data = pd.DataFrame(insulin_data)
        else:
            return "無法進行胰島素藥代動力學分析，因為數據格式不正確。"

    if 'Insulin' not in insulin_data.columns:
        return "無法進行胰島素藥代動力學分析，因為數據中沒有胰島素注射記錄。"

    # 確保兩個數據框都有 Timestamp 列
    if 'Timestamp' not in cgm_df.columns:
        cgm_df['Timestamp'] = pd.to_datetime(cgm_df['Date'].astype(str) + ' ' + cgm_df['Time'].astype(str))
    if 'Timestamp' not in insulin_data.columns:
        insulin_data['Timestamp'] = pd.to_datetime(insulin_data['Date'].astype(str) + ' ' + insulin_data['Time'].astype(str))

    # 合併 CGM 和胰島素數據
    merged_data = pd.merge_asof(cgm_df.sort_values('Timestamp'), 
                                insulin_data.sort_values('Timestamp'), 
                                on='Timestamp', 
                                direction='nearest', 
                                tolerance=pd.Timedelta('1h'))
    
    # 計算胰島素作用時間、峰值時間和持續時間
    merged_data['Glucose_Change'] = merged_data['Sensor Glucose (mg/dL)'].diff()
    merged_data['Time_Since_Insulin'] = (merged_data['Timestamp'] - merged_data['Timestamp'].where(merged_data['Insulin'].notna()).ffill()).dt.total_seconds() / 3600
    
    # 準備分析結果
    def safe_min(x):
        return x.min() if not x.isna().all() else np.nan

    analysis_results = merged_data.groupby('Timestamp').apply(lambda x: pd.Series({
        'Action_Time': safe_min(x['Time_Since_Insulin']),
        'Peak_Time': x['Time_Since_Insulin'][x['Glucose_Change'].idxmin()] if not x['Glucose_Change'].isna().all() else np.nan,
        'Duration': x['Time_Since_Insulin'].max(),
        'Insulin_Sensitivity': x['Glucose_Change'].min() / x['Insulin'].max() if x['Insulin'].max() > 0 else np.nan
    }))

    # 計算平均值
    mean_results = analysis_results.mean()
    
    return mean_results

def analyze_meal_impact(cgm_df, meal_data):
    if not isinstance(meal_data, pd.DataFrame) or meal_data.empty:
        return "無法分析飲食對血糖的影響，因為沒有有效的飲食數據。"

    # 確保兩個數據框都有 Timestamp 列
    if 'Timestamp' not in cgm_df.columns:
        cgm_df['Timestamp'] = pd.to_datetime(cgm_df['Date'].astype(str) + ' ' + cgm_df['Time'].astype(str))
    if 'Timestamp' not in meal_data.columns:
        meal_data['Timestamp'] = pd.to_datetime(meal_data['Date'].astype(str) + ' ' + meal_data['Time'].astype(str))

    # 合併 CGM 和飲食數據
    merged_data = pd.merge_asof(cgm_df.sort_values('Timestamp'), 
                                meal_data.sort_values('Timestamp'), 
                                on='Timestamp', 
                                direction='nearest', 
                                tolerance=pd.Timedelta('1h'))
    
    # 計算飲食後的血糖變化
    merged_data['Glucose_Change'] = merged_data['Sensor Glucose (mg/dL)'].diff()
    merged_data['Time_Since_Meal'] = (merged_data['Timestamp'] - merged_data['Timestamp'].where(merged_data.index.isin(meal_data.index)).ffill()).dt.total_seconds() / 3600
    
    # 準備分析結果
    def safe_max(x):
        return x.max() if not x.isna().all() else np.nan

    analysis_results = merged_data.groupby('Timestamp').apply(lambda x: pd.Series({
        'Peak_Glucose_Time': x['Time_Since_Meal'][x['Glucose_Change'].idxmax()] if not x['Glucose_Change'].isna().all() else np.nan,
        'Peak_Glucose_Change': safe_max(x['Glucose_Change']),
        'Return_To_Baseline_Time': x['Time_Since_Meal'][x['Glucose_Change'].abs().idxmin()] if not x['Glucose_Change'].isna().all() else np.nan
    }))

    # 計算平均值
    mean_results = analysis_results.mean()

    return mean_results

def generate_gpt4_analysis(cgm_metrics, insulin_metrics, meal_impact, insulin_pharmacokinetics, api_key):
    # 設置 API 密鑰
    openai.api_key = api_key
    
    prompt = f"""
    請根據以下數據進行深入的血糖管理分析，請使用繁體中文 zh-tw 回答：

    血糖指標：
    {cgm_metrics}

    胰島素使用情況
    {insulin_metrics}

    飲食對血糖影響：
    {meal_impact}

    胰島素藥代動力學：
    {insulin_pharmacokinetics}

    請提供以下方面的分析：
    1. 整體血糖控制情況評估
    2. 胰島素使用效果分析
    3. 飲食對血糖的影響評估
    4. 改善建議

    請以專業醫療顧問的角度給出分析和建議。
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a endocrinologic professor and professional medical advisor specializing in diabetes management. give me clinically relevant insights and professional recommendations based on the following data as detal as you can:"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

def perform_deep_analysis(cgm_df, insulin_data, meal_data, cgm_metrics, insulin_stats, api_key):
    # 使用 RAG 系統進行 GRI 分析
    reference_db = ReferenceDatabase("references_articles/RAG")
    gri_analysis_result = perform_gri_rag_analysis(cgm_df, reference_db)
    
    insulin_pharmacokinetics = analyze_insulin_pharmacokinetics(cgm_df, insulin_data)
    meal_impact = analyze_meal_impact(cgm_df, meal_data)
    
    # 檢查 insulin_pharmacokinetics 的類型並相應處理
    if isinstance(insulin_pharmacokinetics, dict):
        insulin_pharmacokinetics_str = "\n".join([f"{k}: {v:.2f}" + (" hours" if k != 'Insulin_Sensitivity' else " mg/dL per unit") for k, v in insulin_pharmacokinetics.items()])
    elif isinstance(insulin_pharmacokinetics, str):
        insulin_pharmacokinetics_str = insulin_pharmacokinetics
    else:
        insulin_pharmacokinetics_str = "無法計算胰島素藥代動力學數據"

    # 檢查 meal_impact 的類型並相應處理
    if isinstance(meal_impact, dict):
        meal_impact_str = "\n".join([f"{k}: {v:.2f}" + (" hours" if "Time" in k else " mg/dL") for k, v in meal_impact.items()])
    elif isinstance(meal_impact, str):
        meal_impact_str = meal_impact
    else:
        meal_impact_str = "無法計算飲食對血糖的影響數據"

    gpt4_analysis = generate_gpt4_analysis(cgm_metrics, insulin_stats, meal_impact_str, insulin_pharmacokinetics_str, api_key)
    
    # 返回所有分析結果
    return {
        "gri_analysis": gri_analysis_result,
        "insulin_pharmacokinetics": insulin_pharmacokinetics_str,
        "meal_impact": meal_impact_str,
        "gpt4_analysis": gpt4_analysis
    }