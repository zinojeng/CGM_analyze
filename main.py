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

# 在側邊欄中置 API 金鑰輸入
st.sidebar.title("設定")
openai_api_key = st.sidebar.text_input(
    label="請輸入您的 OpenAI API 金鑰：",
    type='password',
    placeholder="例如：sk-2twmA88un4...",
    help="您可以從 https://platform.openai.com/account/api-keys/ 獲取您的 API 金鑰"
)

# 獲取胰島素信息
insulin_info = get_insulin_info()

uploaded_file = st.file_uploader("請上傳 CGM 數據文件（CSV 或 Excel 格式）", type=["csv", "xlsx", "xls"])

if uploaded_file:
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
            
            st.subheader("AGP 圖")
            st.pyplot(agp_plot)
            
            st.subheader("每日血糖聚類圖")
            st.pyplot(daily_clusters_plot)
        
        if insulin_data is not None:
            st.header("Insulin Data Analysis")  # 更改為英文標題
            
            if not insulin_info:
                st.warning("No insulin information provided. Please fill in the insulin information in the sidebar.")
            else:
                # 分析胰島素數據
                analyzed_insulin_data = analyze_insulin(insulin_data, insulin_info)
                
                # 繪製胰島素數據圖表
                fig = plot_insulin_data(analyzed_insulin_data, insulin_info)
                st.pyplot(fig)
                
                # 顯示統計信息
                st.subheader("Insulin Injection Statistics")  # 更改為英文副標題
                insulin_stats = get_insulin_statistics(analyzed_insulin_data)
                cols = st.columns(3)
                
                for i, (insulin_type, data) in enumerate(insulin_stats.items()):
                    with cols[i]:
                        st.write(insulin_type)
                        st.write(f"Average Dose: {data['平均劑量']:.2f} units")
                        st.write(f"Injection Count: {data['注射次數']}")

                # 使用 insulin_stats 替代 insulin_metrics
                deep_analysis_result = perform_deep_analysis(cgm_df, insulin_data, meal_data, cgm_metrics, insulin_stats, openai_api_key)
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
                    value_float = 0  # 或者其他適當的默認值
                
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
                st.markdown(deep_analysis_result)
            else:
                st.warning("請在側邊欄輸入您的 OpenAI API 金鑰以進行深度分析。")
    else:
        st.error(f"文件 {uploaded_file.name} 拆分失敗")
else:
    st.info("請上傳 CGM 數據文件（CSV 或 Excel 格式）。")