import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

def classify_insulin(hour, dose, insulin_info):
    long_acting = insulin_info['長效胰島素']['劑量']
    short_acting = insulin_info['短效/速效胰島素']['劑量']
    
    if 6 <= hour < 10 and dose > 20:
        return '長效'
    elif dose <= 10:
        return '短效/速效'
    else:
        return '未知'

def analyze_insulin(insulin_data, insulin_info):
    analyzed_data = []
    for _, row in insulin_data.iterrows():
        timestamp = row['Timestamp']
        hour = timestamp.hour + timestamp.minute / 60
        dose = row['Insulin']
        insulin_type = classify_insulin(hour, dose, insulin_info)
        analyzed_data.append({
            'Timestamp': timestamp,
            'Hour': hour,
            'Dose': dose,
            'Type': insulin_type
        })
    return pd.DataFrame(analyzed_data)

def plot_insulin_data(analyzed_data, insulin_info):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    insulin_types = {
        '長效': 'Long-acting',
        '短效/速效': 'Short/Rapid-acting',
        '未知': 'Unknown'
    }
    
    for insulin_type, english_type in insulin_types.items():
        type_data = analyzed_data[analyzed_data['Type'] == insulin_type]
        ax.scatter(type_data['Hour'], type_data['Dose'], label=english_type)
    
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Dose (Units)')
    ax.set_title('Insulin Injection Time and Dose Distribution')
    ax.legend()
    
    return fig

def get_insulin_statistics(analyzed_data):
    stats = {}
    for insulin_type in ['長效', '短效/速效', '未知']:
        type_data = analyzed_data[analyzed_data['Type'] == insulin_type]
        if not type_data.empty:
            stats[insulin_type] = {
                '平均劑量': type_data['Dose'].mean(),
                '注射次數': len(type_data)
            }
    return stats