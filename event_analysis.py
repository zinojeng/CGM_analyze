import pandas as pd
from datetime import datetime, time
import re

def read_file(file_path):
    return pd.read_csv(file_path)

def extract_event_data(file_paths):
    all_insulin_data = []
    all_meal_data = []
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            
            if 'Event Marker' not in df.columns or 'Date' not in df.columns or 'Time' not in df.columns:
                print(f"警告：文件 {file_path} 中缺少必要的列")
                continue
            
            df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            
            # 提取胰島素數據
            insulin_data = df[df['Event Marker'].str.contains('Insulin', case=False, na=False)].copy()
            insulin_data['Insulin Value'] = insulin_data['Event Marker'].str.extract('(\d+\.?\d*)').astype(float)
            all_insulin_data.append(insulin_data[['Timestamp', 'Insulin Value']])
            
            # 提取餐食數據
            meal_data = df[df['Event Marker'].str.contains('Meal', case=False, na=False)].copy()
            meal_data['Meal Grams'] = meal_data['Event Marker'].str.extract('(\d+\.?\d*)').astype(float)
            all_meal_data.append(meal_data[['Timestamp', 'Meal Grams']])
            
        except Exception as e:
            print(f"處理文件 {file_path} 時發生錯誤：{str(e)}")
    
    combined_insulin_data = pd.concat(all_insulin_data, ignore_index=True) if all_insulin_data else pd.DataFrame()
    combined_meal_data = pd.concat(all_meal_data, ignore_index=True) if all_meal_data else pd.DataFrame()
    
    return combined_meal_data, combined_insulin_data

def analyze_insulin(insulin_data):
    if insulin_data.empty:
        return {}, pd.DataFrame()

    # 計算胰島素指標
    insulin_metrics = {
        '平均每日總劑量': insulin_data.groupby(insulin_data['Timestamp'].dt.date)['Insulin Value'].sum().mean(),
        '最大單次劑量': insulin_data['Insulin Value'].max(),
        '最小單次劑量': insulin_data['Insulin Value'].min(),
    }

    # 計算每日胰島素注射統計
    insulin_daily_stats = insulin_data.groupby(insulin_data['Timestamp'].dt.date).agg({
        'Insulin Value': ['count', 'sum', 'mean', 'min', 'max']
    }).reset_index()
    insulin_daily_stats.columns = ['日期', '注射次數', '總劑量', '平均劑量', '最小劑量', '最大劑量']

    return insulin_metrics, insulin_daily_stats

def analyze_meal(meal_data):
    if meal_data.empty:
        return {}, pd.DataFrame()

    # 計算餐食指標
    meal_metrics = {
        '平均每日總克數': meal_data.groupby(meal_data['Timestamp'].dt.date)['Meal Grams'].sum().mean(),
        '最大單次克數': meal_data['Meal Grams'].max(),
        '最小單次克數': meal_data['Meal Grams'].min(),
    }

    # 計算每日餐食統計
    meal_daily_stats = meal_data.groupby(meal_data['Timestamp'].dt.date).agg({
        'Meal Grams': ['count', 'sum', 'mean', 'min', 'max']
    }).reset_index()
    meal_daily_stats.columns = ['日期', '餐食次數', '總克數', '平均克數', '最小克數', '最大克數']

    return meal_metrics, meal_daily_stats

def analyze_events(event_data):
    # 實現事件分析邏輯
    # 這裡只是一個示例，您可能需要根據實際需求調整
    event_types = event_data['Event Marker'].unique()
    
    event_analysis = {}
    for event_type in event_types:
        type_data = event_data[event_data['Event Marker'] == event_type]
        event_analysis[event_type] = {
            "總次數": len(type_data),
            "首次記錄": type_data['Time'].min(),
            "最後記錄": type_data['Time'].max(),
            "平均每日次數": len(type_data) / ((type_data['Time'].max() - type_data['Time'].min()).days + 1)
        }
    
    return event_analysis

def classify_insulin(row, insulin_info):
    dose = row['Insulin']
    injection_time = datetime.strptime(row['Time'], '%H:%M:%S').time()

    for insulin_type, info in insulin_info.items():
        if insulin_type == '長效胰島素':
            if abs(dose - info['劑量']) <= 2:  # 允許2單位的誤差
                user_time = datetime.strptime(info['注射時間'], '%H:%M:%S').time()
                time_diff = (datetime.combine(datetime.today(), injection_time) - datetime.combine(datetime.today(), user_time)).total_seconds() / 3600
                if abs(time_diff) <= 1 or abs(time_diff) >= 23:  # 考慮跨日的情況
                    return '長效胰島素'
        elif insulin_type in ['短效/速效胰島素', '預混型胰島素']:
            for meal, meal_dose in info.items():
                if meal in ['早餐', '午餐', '晚餐']:
                    meal_time = get_meal_time(meal)
                    if meal_time[0] <= injection_time <= meal_time[1] and abs(dose - meal_dose) <= 2:
                        return f"{insulin_type} ({meal})"

    return '未分類胰島素'

def get_meal_time(meal):
    if meal == '早餐':
        return (time(6, 0), time(10, 0))
    elif meal == '午餐':
        return (time(11, 0), time(14, 0))
    elif meal == '晚餐':
        return (time(17, 0), time(21, 0))
    else:
        return (time(0, 0), time(23, 59))