import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from event_analysis import extract_event_data, analyze_meal
from glucose_analysis import calculate_metrics, create_agp, create_daily_clusters
from deep_analysis import perform_deep_analysis
from split_csv import split_csv
from insulin_input import get_insulin_info
from insulin_analysis import extract_insulin_data, analyze_insulin, plot_insulin_data, get_insulin_statistics
from gri_rag import GRIAnalyzer, ReferenceDatabase, perform_gri_rag_analysis
from gri_plotting import plot_gri
from agp_variability import agp_variability

# 設置中文顯示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 首選 Arial Unicode MS，備選 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解決坐標軸負號顯示問題

def read_cgm_file(file_path):
    if not os.path.exists(file_path):
        st.error(f"文件不存在：{file_path}")
        return pd.DataFrame()
    
    st.write(f"正在處理 CGM 數據文件：{os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"讀取文件時發生錯誤：{str(e)}")
        return pd.DataFrame()
    
    required_columns = ['Date', 'Time', 'Sensor Glucose (mg/dL)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"文件 {os.path.basename(file_path)} 中缺少以下必要的{', '.join(missing_columns)}")
        return pd.DataFrame()
    
    # 合併 Date 和 Time 列
    df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    
    df['Sensor Glucose (mg/dL)'] = pd.to_numeric(df['Sensor Glucose (mg/dL)'], errors='coerce')
    return df.sort_values('Timestamp')

def clean_value(value):
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if '%' in value:
            return float(value.strip('%')) / 100
        elif 'mg/dL' in value:
            return float(value.split()[0])
    try:
        return float(value)
    except ValueError:
        return value  # 如果無法轉換，則返回原始值

st.title("CGM 數據分析")

# 側邊欄設置保持不變
st.sidebar.title("設定")
openai_api_key = st.sidebar.text_input(
    label="請輸入您的 OpenAI API 金鑰：",
    type='password',
    placeholder="例：sk-2twmA88un4...",
    help="您可以從 https://platform.openai.com/account/api-keys/ 獲取您的 API 金鑰"
)

uploaded_file = st.file_uploader("請上傳 CGM 數據文件（CSV 或 Excel 格式）", type=["csv", "xlsx", "xls"])

if uploaded_file:
    st.success(f"文件 {uploaded_file.name} 已成功上傳")

    # 將胰島素資訊輸入移到這裡（主頁）
    st.subheader("胰島素資訊")
    insulin_info = get_insulin_info()

    # 新增"執行分析"按鈕
    if st.button("執行分析"):
        # 確保輸出目錄存在
        output_directory = "output_directory"
        os.makedirs(output_directory, exist_ok=True)

        # 使用 split_csv 函數分割上傳的文件
        event_file, sensor_glucose_file = split_csv(uploaded_file, output_directory)
        if event_file and sensor_glucose_file:
            st.success(f"文件 {uploaded_file.name} 已成功拆分")
            
            # 讀取拆分後的文件
            cgm_df = read_cgm_file(sensor_glucose_file)
            meal_data, insulin_data = extract_event_data([event_file])
            
            # 提取胰島素數據
            insulin_data = extract_insulin_data(event_file)
            
            if not cgm_df.empty:
                st.header("血糖數據分析")
                cgm_metrics = calculate_metrics(cgm_df)
                
                # 移除調試信息
                # st.write("Debug: CGM Metrics Keys:", list(cgm_metrics.keys()))
                # st.write("Debug: Mean Glucose Value:", cgm_metrics.get('Mean Glucose (mg/dL)', 'Not Found'))
                
                agp_plot = create_agp(cgm_df)
                daily_clusters_plot = create_daily_clusters(cgm_df)
                
                st.subheader("血糖數據指標")
                col1, col2, col3, col4 = st.columns(4)
                
                metrics_order = [
                    "VLow (<54 mg/dL)", "Low (54-<70 mg/dL)", "TIR (70-180 mg/dL)",
                    "High (>180-250 mg/dL)", "VHigh (>250 mg/dL)", "CV",
                    "Mean Glucose (mg/dL)", "GMI", "GRI"
                ]
                
                for i, metric in enumerate(metrics_order):
                    with [col1, col2, col3, col4][i % 4]:
                        value = cgm_metrics.get(metric, "N/A")
                        if isinstance(value, (float, np.float64)):
                            if metric == "Mean Glucose (mg/dL)":
                                st.metric(label=metric, value=f"{value:.1f}")
                            elif metric in ["VLow (<54 mg/dL)", "Low (54-<70 mg/dL)", "TIR (70-180 mg/dL)", "High (>180-250 mg/dL)", "VHigh (>250 mg/dL)"]:
                                st.metric(label=metric, value=f"{value:.2%}")
                            else:
                                st.metric(label=metric, value=f"{value:.2f}")
                        else:
                            st.metric(label=metric, value=value)
                
                # st.subheader("AGP 圖")
                # st.pyplot(agp_plot)
                
                # st.subheader("每日血糖聚類圖")
                # st.pyplot(daily_clusters_plot)
            
            if insulin_data is not None:
                st.header("胰島素數據分析")
                analyzed_insulin_data = analyze_insulin(insulin_data, insulin_info)
                
                # 繪製胰島素數據圖表
                fig = plot_insulin_data(analyzed_insulin_data)
                st.pyplot(fig)
                
                # 顯示統計信息
                st.subheader("胰島素注射統計")
                insulin_stats = get_insulin_statistics(analyzed_insulin_data)
                
                for category in ['長效胰島素', '速效胰島素', '預混胰島素', '未知']:
                    if category in insulin_stats:
                        st.write(f"**{category}**")
                        data = insulin_stats[category]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("平均劑量", f"{data['平均劑量']} 單位")
                        with col2:
                            st.metric("注射次數", data['注射次數'])
                        st.write(f"劑量範圍: {data['最小劑量']} - {data['最大劑量']} 單位")
                        
                        if category != '未知':
                            # 顯示常見注射時間和對應的平均劑量
                            if data['常見注射時間']:
                                st.write("常見注射時間和平均劑量：")
                                for time, avg_dose, count in data['常見注射時間']:
                                    st.write(f"  - {time:02d}:00 附近，平均劑量 {avg_dose} 單位 (共 {count} 次)")
                        else:
                            # 顯示未知類型的劑量分組
                            if '劑量分組' in data:
                                st.write("未分類胰島素注射分組：")
                                for avg_time, avg_dose, count in data['劑量分組']:
                                    hour = int(avg_time)
                                    minute = int((avg_time - hour) * 60)
                                    st.write(f"  - {hour:02d}:{minute:02d} 附近，平均劑量 {avg_dose} 單位 (共 {count} 次)")
                        
                        # 顯示每種胰島素的詳細信息
                        for insulin_name, insulin_data in data.items():
                            if isinstance(insulin_data, dict) and insulin_name not in ['常見注射時間', '劑量分組']:
                                st.write(f"  - {insulin_name}: 平均劑量 {insulin_data['平均劑量']} 單位, 注射次數 {insulin_data['注射次數']}")
                        
                        st.write("---")

                if '未知' in insulin_stats:
                    st.warning("有未分類的胰島素注射記錄。這些記錄已根據劑量和時間進行了分組。請檢查這些分組是否符合您的實際使用情況，並考慮更新您的胰島素輸入設置。")
            else:
                st.warning("無法提取有效的胰島素數據。")
            
            if not meal_data.empty:
                st.header("餐食數據分析")
                st.success(f"成功提取餐食數據！共 {len(meal_data)} 條記錄")
                
                st.subheader("餐食數據預覽")
                st.dataframe(meal_data.head())
                
                meal_metrics, meal_daily_stats = analyze_meal(meal_data)
                
                st.subheader("餐食分析結果")
                col1, col2 = st.columns(2)
                for i, (key, value) in enumerate(meal_metrics.items()):
                    if isinstance(value, (float, np.float64)):
                        value_float = value
                    elif isinstance(value, str):
                        value_float = float(value.strip('%')) / 100 if '%' in value else float(value)
                    else:
                        st.error(f"Unexpected value type: {type(value)}")
                        value_float = 0  # 或者其他適當的默認
                    
                    st.metric(label=key, value=f"{value_float:.2f}")
                
                st.subheader("每日餐食統計")
                st.dataframe(meal_daily_stats)
            else:
                st.warning("無法從文件中提取有效的餐食數據。")
            
            if not cgm_df.empty and insulin_data is not None:
                st.header("深度分析和總結")
                if openai_api_key:
                    with st.spinner("正在進行深度分析，請稍候..."):
                        deep_analysis_result = perform_deep_analysis(cgm_df, insulin_data, meal_data, cgm_metrics, insulin_stats, openai_api_key)
                        
                        
                        
                        # 新增: AGP 變異性分析
                        st.subheader("AGP 分析")
                        if openai_api_key:
                            agp_analysis, hypo_hyper_analysis, sd, cv, mage = agp_variability(cgm_df, openai_api_key)
                            
                            # 顯示 AGP 圖
                            st.pyplot(agp_plot)
                            st.subheader("每日血糖聚類圖")
                            st.pyplot(daily_clusters_plot)
                            
                            # 顯示 AGP 分析結果
                            st.write("AGP 分析:", agp_analysis)
                            st.write("低血糖和高血糖分析:", hypo_hyper_analysis)
                        else:
                            st.warning("請輸入 OpenAI API 金鑰以進行 AGP 變異性分析。")
                        
                        # 顯示 GRI 分析圖表
                        st.subheader("GRI 分析圖表")
                        gri_plot = plot_gri(cgm_df)
                        st.plotly_chart(gri_plot)
                        
                        # 顯示 GRI RAG 分析
                        st.subheader("GRI RAG 分析")
                        st.write(deep_analysis_result["GRI RAG Analysis"])

                        # 顯示綜合 GPT-4 分析
                        st.subheader("綜合 GPT-4 分析")
                        st.write(deep_analysis_result["Overall GPT-4 Analysis"])
                else:
                    st.warning("請輸入 OpenAI API 金鑰以進行深度分析。")
        else:
            st.error(f"文件 {uploaded_file.name} 拆分失敗")

        # 在這裡，您可以使用 insulin_info 來進行後續的分析
        # st.write("選擇的胰島素及劑量：", insulin_info)  # 移除或註釋掉這一行

else:
    st.info("請上傳 CGM 數據文件（CSV 或 Excel 格式）。")