import pandas as pd
import streamlit as st

def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        st.error(f"不支持的文件格式：{file.name}。請上傳 CSV 或 Excel 文件。")
        return None

def extract_event_data(files):
    all_meal_data = []
    all_insulin_data = []
    
    for file in files:
        df = read_file(file)
        if df is None:
            continue
        
        st.write(f"正在處理文件：{file.name}")
        
        required_columns = ['Date', 'Time', 'Event Marker']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"文件 {file.name} 中缺少以下必要的列：{', '.join(missing_columns)}")
            continue
        
        # 確保 Date 和 Time 列是字符串類型
        df['Date'] = df['Date'].astype(str)
        df['Time'] = df['Time'].astype(str)
        
        # 合併 Date 和 Time 列來創建 Timestamp 列
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        
        # 提取飲食記錄
        meal_data = df[df['Event Marker'].str.contains('Meal', case=False, na=False)][['Timestamp']]
        all_meal_data.append(meal_data)
        
        # 提取胰島素記錄
        insulin_data = df[df['Event Marker'].str.contains('Insulin', case=False, na=False)][['Timestamp', 'Event Marker']]
        insulin_data['Insulin'] = insulin_data['Event Marker'].str.extract(r'Insulin: (\d+\.?\d*)')
        insulin_data['Insulin'] = pd.to_numeric(insulin_data['Insulin'], errors='coerce')
        insulin_data = insulin_data[['Timestamp', 'Insulin']]
        all_insulin_data.append(insulin_data)
    
    combined_meal_data = pd.concat(all_meal_data).sort_values('Timestamp') if all_meal_data else pd.DataFrame()
    combined_insulin_data = pd.concat(all_insulin_data).sort_values('Timestamp') if all_insulin_data else pd.DataFrame()
    
    st.write(f"提取到的飲食記錄總數：{len(combined_meal_data)}")
    st.write(f"提取到的胰島素記錄總數：{len(combined_insulin_data)}")
    
    return combined_meal_data, combined_insulin_data

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

# analyze_insulin 函數保持不變