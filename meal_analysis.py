import pandas as pd
import streamlit as st

def extract_meal_data(file):
    try:
        df = pd.read_excel(file)
        st.write(f"成功讀取 Excel 文件：{file.name}")
    except Exception as e:
        st.error(f"無法讀取文件 {file.name}：{str(e)}")
        return None
    
    required_columns = ['Date', 'Time', 'Event Marker']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"文件中缺少以下必要的列：{', '.join(missing_columns)}")
        return None
    
    # 確保 Date 和 Time 列是字符串類型
    df['Date'] = df['Date'].astype(str)
    df['Time'] = df['Time'].astype(str)
    
    # 合併 Date 和 Time 列來創建 Timestamp 列
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    
    # 提取飲食記錄
    meal_data = df[df['Event Marker'].str.contains('Meal', case=False, na=False)][['Timestamp']]
    
    if meal_data.empty:
        st.warning(f"文件中沒有找到有效的飲食記錄。")
        return None
    
    st.write(f"提取到的飲食記錄數量：{len(meal_data)}")
    return meal_data.sort_values('Timestamp')