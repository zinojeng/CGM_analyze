import pandas as pd
import numpy as np
from openai import OpenAI  # 修改這行
from gri_rag import GRIAnalyzer, perform_gri_rag_analysis, ReferenceDatabase
from gri_plotting import plot_gri  # 假設我們將獨立的繪圖函數放在 gri_plotting.py 中
import os

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
    
    # 計算胰島素作用時間、峰值間和持續時間
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

def generate_gpt4_analysis(cgm_metrics, insulin_stats, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    
    prompt = f"""
    基於以下血糖監測（CGM）和胰島素數據的分析結果，請提供深入的見解和建議：

    CGM 指標：
    {cgm_metrics}

    # 胰島素統計：
    # {insulin_stats}

    請提供以下方面的分析：
    1. 血糖控制的整體評估
    2. 胰島素使用的效果和建議

    請用中文（繁體）回答，並確保回答準確、專業且易於理解。
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a diabetes management expert providing analysis based on CGM and insulin data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7
    )

    return response.choices[0].message.content

def perform_deep_analysis(cgm_df, insulin_data, meal_data, cgm_metrics, insulin_stats, openai_api_key):
    # 使用相對路徑指向您的 PDF 文件目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_directory = os.path.join(current_dir, 'references_articles', 'RAG')
    
    # 創建 ReferenceDatabase 實例
    reference_db = ReferenceDatabase(pdf_directory)
    
    gri_analyzer = GRIAnalyzer(cgm_df, reference_db)
    gri_result = gri_analyzer.analyze()
    
    # 使用獨立的繪圖函數來生成 GRI 圖表
    gri_plot = plot_gri(cgm_df)
    
    # 執行 GRI RAG 分析
    gri_rag_analysis = perform_gri_rag_analysis(cgm_df, reference_db, openai_api_key)
    
    # 生成綜合 GPT-4 分析
    overall_gpt4_analysis = generate_gpt4_analysis(cgm_metrics, insulin_stats, openai_api_key)
    
    # 組合深度分析結果
    deep_analysis_result = {
        "GRI Analysis": gri_result,
        "GRI Plot": gri_plot,
        "GRI RAG Analysis": gri_rag_analysis,
        "Overall GPT-4 Analysis": overall_gpt4_analysis
    }
    
    return deep_analysis_result