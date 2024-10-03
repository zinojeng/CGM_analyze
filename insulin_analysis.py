import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_insulin_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            st.write("成功讀取 CSV 文件")
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, engine='openpyxl')  # 手動指定引擎
            st.write("成功讀取 Excel 文件")
        else:
            st.error("不支持的文件格式。請上傳 CSV 或 Excel 文件。")
            return None
    except Exception as e:
        st.error(f"無法讀取文件：{str(e)}")
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
    insulin_data = df[['Timestamp', 'Insulin', 'Event Marker']].dropna()
    
    if insulin_data.empty:
        st.warning(f"文件中沒有找到有效的胰島素數據。")
        return None
    
    st.write(f"最終提取到的有效胰島素數據行數：{len(insulin_data)}")
    return insulin_data

def classify_insulin(timestamp, dose, insulin_info):
    hour = timestamp.hour + timestamp.minute / 60
    
    # 首先嘗試根據用戶輸入進行匹配
    for insulin_category, insulins in insulin_info.items():
        if insulin_category in ['長效胰島素', '速效胰島素', '預混胰島素']:
            for insulin in insulins:
                if insulin in insulin_info:
                    for time, value in insulin_info[insulin].items():
                        if value > 0:  # 只考慮非零劑量
                            time_hour = {'morning': 6, 'noon': 12, 'evening': 18, 'bedtime': 22}[time]
                            # 允許3小時的時間誤差和30%的量誤差
                            if abs(hour - time_hour) <= 3 and 0.7 * value <= dose <= 1.3 * value:
                                return insulin_category, insulin
    
    # 如果無法根據用戶輸入匹配，則使用常見模式進行推斷
    if 5 <= hour <= 9 or 21 <= hour <= 23:  # 早晨或睡前
        if dose >= 10:  # 假設長效胰島素劑量通常較大
            return '長效胰島素', '未指定長效'
    
    if (6 <= hour <= 9) or (11 <= hour <= 14) or (17 <= hour <= 20):  # 餐前時間
        if dose < 10:  # 假設短效胰島素劑量通常較小
            return '速效胰島素', '未指定速效'
    
    return '未知', '未知'

def analyze_insulin(insulin_data, insulin_info):
    analyzed_data = []
    for _, row in insulin_data.iterrows():
        timestamp = row['Timestamp']
        dose = row['Insulin']
        insulin_category, insulin_name = classify_insulin(timestamp, dose, insulin_info)
        analyzed_data.append({
            'Timestamp': timestamp,
            'Hour': timestamp.hour + timestamp.minute / 60,
            'Dose': dose,
            'Category': insulin_category,
            'Name': insulin_name
        })
    return pd.DataFrame(analyzed_data)

def plot_insulin_data(analyzed_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = analyzed_data['Category'].unique()
    for category in categories:
        category_data = analyzed_data[analyzed_data['Category'] == category]
        ax.scatter(category_data['Hour'], category_data['Dose'], label=category)
    
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Dose (Units)')
    ax.set_title('Insulin Injection Time and Dose Distribution')
    ax.legend()
    
    return fig

def get_insulin_statistics(analyzed_data):
    stats = {}
    time_stats = defaultdict(lambda: defaultdict(list))
    
    for category in analyzed_data['Category'].unique():
        category_data = analyzed_data[analyzed_data['Category'] == category]
        stats[category] = {
            '平均劑量': round(category_data['Dose'].mean(), 2),
            '最小劑量': round(category_data['Dose'].min(), 2),
            '最大劑量': round(category_data['Dose'].max(), 2),
            '注射次數': len(category_data)
        }
        
        # 計算每個時間點的平均劑量
        for _, row in category_data.iterrows():
            hour = row['Hour']
            rounded_hour = round(hour)
            time_stats[category][rounded_hour].append(row['Dose'])
        
        # 合併2小時內的相近劑量
        merged_times = defaultdict(list)
        sorted_times = sorted(time_stats[category].items())
        for i, (time, doses) in enumerate(sorted_times):
            if i > 0 and time - sorted_times[i-1][0] <= 2:
                merged_times[sorted_times[i-1][0]].extend(doses)
            else:
                merged_times[time] = doses
        
        # 找出最常見的注射時間和對應的平均劑量
        common_times = sorted(merged_times.items(), key=lambda x: len(x[1]), reverse=True)
        stats[category]['常見注射時間'] = [
            (time, round(sum(doses) / len(doses), 2), len(doses))
            for time, doses in common_times
            if len(doses) > len(category_data) * 0.1  # 只考慮至少10%的注射發生在這個時間的情況
        ]
        
        # 對於未知類型，進行額外的劑量分組
        if category == '未知':
            dose_groups = defaultdict(list)
            for time, doses in merged_times.items():
                for dose in doses:
                    rounded_dose = round(dose / 5) * 5  # 將劑量四捨五入到最近的5的倍數
                    dose_groups[rounded_dose].append((time, dose))
            
            stats[category]['劑量分組'] = []
            for rounded_dose, time_doses in sorted(dose_groups.items()):
                avg_time = sum(t for t, _ in time_doses) / len(time_doses)
                avg_dose = sum(d for _, d in time_doses) / len(time_doses)
                stats[category]['劑量分組'].append((
                    round(avg_time, 1),
                    round(avg_dose, 2),
                    len(time_doses)
                ))
        
        # 添加每種胰島素的統計
        for insulin_name in category_data['Name'].unique():
            if insulin_name != '未知':
                insulin_data = category_data[category_data['Name'] == insulin_name]
                stats[category][insulin_name] = {
                    '平均劑量': round(insulin_data['Dose'].mean(), 2),
                    '注射次數': len(insulin_data)
                }
    
    return stats