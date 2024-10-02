import streamlit as st
import pandas as pd
import os
from event_analysis import extract_event_data, analyze_insulin, analyze_meal
from glucose_analysis import calculate_metrics, create_agp, create_daily_clusters
from deep_analysis import perform_deep_analysis
from split_csv import split_csv
from insulin_input import get_insulin_info  # 導入 get_insulin_info 函數

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
        st.error(f"文件 {os.path.basename(file_path)} 中缺少以下必要的列：{', '.join(missing_columns)}")
        return pd.DataFrame()
    
    # 合併 Date 和 Time 列
    df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    
    df['Sensor Glucose (mg/dL)'] = pd.to_numeric(df['Sensor Glucose (mg/dL)'], errors='coerce')
    return df.sort_values('Timestamp')

st.title("CGM 數據分析")

# 在側邊欄中設置 API 金鑰輸入
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
        
        if not cgm_df.empty:
            st.header("血糖數據分析")
            st.success(f"成功讀取 CGM 數據！共 {len(cgm_df)} 條記錄")
            
            cgm_metrics = calculate_metrics(cgm_df)
            
            col1, col2, col3 = st.columns(3)
            for i, (key, value) in enumerate(cgm_metrics.items()):
                with [col1, col2, col3][i % 3]:
                    st.metric(label=key, value=value)
            
            st.subheader("血糖趨勢圖 (AGP)")
            agp_fig = create_agp(cgm_df)
            st.pyplot(agp_fig)
            
            st.subheader("每日血糖分布")
            clusters_fig = create_daily_clusters(cgm_df)
            st.pyplot(clusters_fig)
        else:
            st.warning("無法從文件中提取有效的 CGM 數據。")
        
        if not insulin_data.empty:
            st.header("胰島素數據分析")
            st.success(f"成功提取胰島素數據！共 {len(insulin_data)} 條記錄")
            
            st.subheader("胰島素數據預覽")
            st.dataframe(insulin_data.head())
            
            insulin_metrics, insulin_daily_stats = analyze_insulin(insulin_data)
            
            st.subheader("胰島素分析結果")
            col1, col2 = st.columns(2)
            for i, (key, value) in enumerate(insulin_metrics.items()):
                with col1 if i % 2 == 0 else col2:
                    st.metric(label=key, value=f"{value:.2f}")
            
            st.subheader("每日胰島素注射統計")
            st.dataframe(insulin_daily_stats)
        else:
            st.warning("無法從文件中提取有效的胰島素數據。")
        
        if not meal_data.empty:
            st.header("餐食數據分析")
            st.success(f"成功提取餐食數據！共 {len(meal_data)} 條記錄")
            
            st.subheader("餐食數據預覽")
            st.dataframe(meal_data.head())
            
            meal_metrics, meal_daily_stats = analyze_meal(meal_data)
            
            st.subheader("餐食分析結果")
            col1, col2 = st.columns(2)
            for i, (key, value) in enumerate(meal_metrics.items()):
                with col1 if i % 2 == 0 else col2:
                    st.metric(label=key, value=f"{value:.2f}")
            
            st.subheader("每日餐食統計")
            st.dataframe(meal_daily_stats)
        else:
            st.warning("無法從文件中提取有效的餐食數據。")
        
        if not cgm_df.empty and not insulin_data.empty:
            st.header("深度分析和總結")
            if openai_api_key:
                with st.spinner("正在進行深度分析，請稍候..."):
                    deep_analysis_result = perform_deep_analysis(cgm_df, insulin_data, meal_data, cgm_metrics, insulin_metrics, openai_api_key)
                st.markdown(deep_analysis_result)
            else:
                st.warning("請在側邊欄輸入您的 OpenAI API 金鑰以進行深度分析。")
    else:
        st.error(f"文件 {uploaded_file.name} 拆分失敗")
else:
    st.info("請上傳 CGM 數據文件（CSV 或 Excel 格式）。")