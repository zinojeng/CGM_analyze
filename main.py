import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_file(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        st.error("不支持的文件格式。請上傳 CSV 或 Excel 文件。")
        return None
    
    # 檢查必要的列是否存在
    required_columns = ['Date', 'Time', 'Sensor Glucose (mg/dL)']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"文件中缺少以下必要的列：{', '.join(missing_columns)}")
        st.write("可用的列：", ", ".join(df.columns))
        return None
    
    # 確保 Date 和 Time 列是字符串類型
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)
    
    # 合併 Date 和 Time 列來創建 Timestamp 列
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    return df

def calculate_metrics(df):
    glucose = df['Sensor Glucose (mg/dL)']
    
    vlow = (glucose < 54).mean() * 100
    low = ((glucose >= 54) & (glucose < 70)).mean() * 100
    tir = ((glucose >= 70) & (glucose <= 180)).mean() * 100
    high = ((glucose > 180) & (glucose <= 250)).mean() * 100
    vhigh = (glucose > 250).mean() * 100
    
    cv = glucose.std() / glucose.mean() * 100
    mg = glucose.mean()
    gmi = 3.31 + (0.02392 * mg)
    
    gri = (3.0 * vlow) + (2.4 * low) + (1.6 * vhigh) + (0.8 * high)
    
    return {
        "VLow (<54 mg/dL)": f"{vlow:.1f}%",
        "Low (54-<70 mg/dL)": f"{low:.1f}%",
        "TIR (70-180 mg/dL)": f"{tir:.1f}%",
        "High (>180-250 mg/dL)": f"{high:.1f}%",
        "VHigh (>250 mg/dL)": f"{vhigh:.1f}%",
        "CV": f"{cv:.1f}%",
        "Mean Glucose": f"{mg:.1f} mg/dL",
        "GMI": f"{gmi:.1f}%",
        "GRI": f"{gri:.1f}"
    }

def create_agp(df):
    # 提取小時和分鐘
    df['Time'] = df['Timestamp'].dt.strftime('%H:%M')
    
    # 按時間分組並計算百分位數
    grouped = df.groupby('Time')['Sensor Glucose (mg/dL)']
    percentiles = grouped.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    
    # 重塑數據以便繪圖
    percentiles = percentiles.unstack()
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 繪製填充區域
    ax.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95], alpha=0.2, color='blue')
    ax.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75], alpha=0.2, color='blue')
    
    # 繪製中位線
    ax.plot(percentiles.index, percentiles[0.5], color='blue', linewidth=2)
    
    # 設置 y 軸範圍
    ax.set_ylim(40, 400)
    
    # 添加標籤和標題
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.set_title('Ambulatory Glucose Profile')
    
    # 添加網格線
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 調整 x 軸刻度
    ax.set_xticks(ax.get_xticks()[::2])  # 每隔一個刻度顯示
    
    # 添加目標範圍
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=180, color='green', linestyle='--', alpha=0.7)
    
    return fig

def create_daily_clusters(df):
    df['Date'] = df['Timestamp'].dt.date
    daily_stats = df.groupby('Date').apply(lambda x: pd.Series({
        'Time > 250': (x['Sensor Glucose (mg/dL)'] > 250).mean() * 100,
        'TAR (181-250)': ((x['Sensor Glucose (mg/dL)'] > 180) & (x['Sensor Glucose (mg/dL)'] <= 250)).mean() * 100,
        'TIR (70-180)': ((x['Sensor Glucose (mg/dL)'] >= 70) & (x['Sensor Glucose (mg/dL)'] <= 180)).mean() * 100,
        'TBR (54-69)': ((x['Sensor Glucose (mg/dL)'] >= 54) & (x['Sensor Glucose (mg/dL)'] < 70)).mean() * 100,
        'Time < 50': (x['Sensor Glucose (mg/dL)'] < 50).mean() * 100
    }))
    
    # 保持順序不變
    daily_stats = daily_stats[['Time > 250', 'TAR (181-250)', 'TIR (70-180)', 'TBR (54-69)', 'Time < 50']]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 使用 bottom 參數來控制堆疊順序
    bottoms = np.zeros(len(daily_stats))
    colors = ['red', 'orange', 'green', 'yellow', 'blue']
    
    for column in reversed(daily_stats.columns):
        ax.bar(daily_stats.index, daily_stats[column], bottom=bottoms, label=column, color=colors[daily_stats.columns.get_loc(column)])
        bottoms += daily_stats[column]
    
    ax.set_title('Clinically Similar Clusters')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Date')
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 調整圖例順序並放置在右側
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    return fig

st.title("CGM 數據分析")

uploaded_file = st.file_uploader("請上傳您的 CSV 或 Excel 文件", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = read_file(uploaded_file)
    if df is not None:
        st.success("文件上傳成功！")
        
        metrics = calculate_metrics(df)
        
        st.header("分析結果")
        col1, col2, col3 = st.columns(3)
        for i, (key, value) in enumerate(metrics.items()):
            with [col1, col2, col3][i % 3]:
                st.metric(label=key, value=value)
        
        st.header("Ambulatory Glucose Profile")
        agp_fig = create_agp(df)
        st.pyplot(agp_fig)
        
        st.header("Clinically Similar Clusters")
        csc_fig = create_daily_clusters(df)
        st.pyplot(csc_fig)
        
        st.header("數據預覽")
        st.dataframe(df.head())

st.sidebar.header("使用說明")
st.sidebar.info(
    "1. 點擊 '瀏覽文件' 按鈕上傳您的 CSV 或 Excel 文件。\n"
    "2. 確保您的文件包含 'Date'、'Time' 和 'Sensor Glucose (mg/dL)' 列。\n"
    "3. 上傳後，您將看到分析結果、Ambulatory Glucose Profile 和 Clinically Similar Clusters 圖，以及數據預覽。"
)