import pandas as pd
import streamlit as st

def extract_insulin_data(file):
    try:
        df = pd.read_excel(file)
        st.write("成功讀取 Excel 文件")
    except Exception as e:
        st.error(f"無法讀取文件 {file.name}：{str(e)}")
        return None
    
    required_columns = ['Date', 'Time', 'Event Marker']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"文件中缺少以下必要的列：{', '.join(missing_columns)}")
        return None
    
    st.write("所有必要的列都存在")
    
    # 確保 Date 和 Time 列是字符串類型
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)
    
    # 合併 Date 和 Time 列來創建 Timestamp 列
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    st.write(f"創建 Timestamp 列，有效時間戳數量：{df['Timestamp'].notna().sum()}")
    
    # 提取胰島素劑量
    df['Insulin'] = df['Event Marker'].str.extract(r'Insulin: (\d+\.?\d*)')
    df['Insulin'] = pd.to_numeric(df['Insulin'], errors='coerce')
    st.write(f"提取胰島素劑量，有效劑量數量：{df['Insulin'].notna().sum()}")
    
    # 只保留有胰島素數據的行
    insulin_data = df[['Timestamp', 'Insulin']].dropna()
    
    if insulin_data.empty:
        st.warning(f"文件中沒有找到有效的胰島素數據。")
        return None
    
    st.write(f"最終提取到的有效胰島素數據行數：{len(insulin_data)}")
    return insulin_data

def analyze_insulin(insulin_data):
    # 確保 Timestamp 列是日期時間類型
    insulin_data['Timestamp'] = pd.to_datetime(insulin_data['Timestamp'])
    
    # 添加日期列
    insulin_data['Date'] = insulin_data['Timestamp'].dt.date
    
    # 按日期分組計算
    daily_stats = insulin_data.groupby('Date').agg({
        'Insulin': ['sum', 'mean', 'max', 'count']
    })
    
    daily_stats.columns = ['總劑量', '平均劑量', '最大劑量', '注射次數']
    
    # 計算整體統計
    total_days = len(daily_stats)
    avg_daily_total = daily_stats['總劑量'].mean()
    avg_daily_mean = daily_stats['平均劑量'].mean()
    avg_daily_max = daily_stats['最大劑量'].mean()
    avg_daily_count = daily_stats['注射次數'].mean()
    
    max_daily_total = daily_stats['總劑量'].max()
    max_daily_dose = daily_stats['最大劑量'].max()
    max_daily_count = daily_stats['注射次數'].max()
    
    return {
        "平均每日總劑量": f"{avg_daily_total:.2f} 單位",
        "平均每次劑量": f"{avg_daily_mean:.2f} 單位",
        "平均每日最大劑量": f"{avg_daily_max:.2f} 單位",
        "平均每日注射次數": f"{avg_daily_count:.1f} 次",
        "最大單日總劑量": f"{max_daily_total:.2f} 單位",
        "最大單次劑量": f"{max_daily_dose:.2f} 單位",
        "最大單日注射次數": f"{max_daily_count} 次",
        "分析天數": f"{total_days} 天"
    }, daily_stats