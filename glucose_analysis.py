import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def calculate_metrics(cgm_df):
    metrics = {}
    glucose_values = cgm_df['Sensor Glucose (mg/dL)'].dropna()
    
    # Calculate Mean Glucose
    mean_glucose = glucose_values.mean()
    metrics['Mean Glucose (mg/dL)'] = mean_glucose

    # Calculate time in ranges
    total_readings = len(glucose_values)
    metrics['VLow (<54 mg/dL)'] = (glucose_values < 54).sum() / total_readings
    metrics['Low (54-<70 mg/dL)'] = ((glucose_values >= 54) & (glucose_values < 70)).sum() / total_readings
    metrics['TIR (70-180 mg/dL)'] = ((glucose_values >= 70) & (glucose_values <= 180)).sum() / total_readings
    metrics['High (>180-250 mg/dL)'] = ((glucose_values > 180) & (glucose_values <= 250)).sum() / total_readings
    metrics['VHigh (>250 mg/dL)'] = (glucose_values > 250).sum() / total_readings

    # Calculate CV
    cv = glucose_values.std() / mean_glucose
    metrics['CV'] = cv

    # Calculate GMI (Glucose Management Indicator)
    metrics['GMI'] = 3.31 + (0.02392 * mean_glucose)

    # Calculate GRI (Glycemic Risk Index)
    gri = np.log(glucose_values / 100) ** 2
    metrics['GRI'] = gri.mean() * 100

    # Calculate MAGE
    metrics['MAGE'] = calculate_mage(glucose_values)

    return metrics

def calculate_mage(glucose_values, threshold=1):
    differences = np.abs(np.diff(glucose_values))
    significant_changes = differences[differences > threshold * glucose_values.std()]
    return np.mean(significant_changes)

def create_agp(df):
    df['Time'] = df['Timestamp'].dt.strftime('%H:%M')
    grouped = df.groupby('Time')['Sensor Glucose (mg/dL)']
    percentiles = grouped.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    percentiles = percentiles.unstack()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95], alpha=0.2, color='blue')
    ax.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75], alpha=0.2, color='blue')
    ax.plot(percentiles.index, percentiles[0.5], color='blue', linewidth=2)
    ax.set_ylim(40, 400)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.set_title('Ambulatory Glucose Profile')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(ax.get_xticks()[::2])
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
        'Time < 54': (x['Sensor Glucose (mg/dL)'] < 54).mean() * 100
    }))
    
    daily_stats = daily_stats[['Time > 250', 'TAR (181-250)', 'TIR (70-180)', 'TBR (54-69)', 'Time < 54']]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros(len(daily_stats))
    
    # 更新顏色映射，使用較深的綠色
    colors = {
        'Time > 250': '#FF9900',    # 橙色
        'TAR (181-250)': '#FFFF00', # 黃色
        'TIR (70-180)': '#228B22',  # 較深的綠色
        'TBR (54-69)': '#FF0000',   # 紅色
        'Time < 54': '#990000'      # 深紅色
    }
    
    for column in reversed(daily_stats.columns):
        ax.bar(daily_stats.index, daily_stats[column], bottom=bottoms, label=column, color=colors[column])
        bottoms += daily_stats[column]
    
    ax.set_title('Daily Glucose Cluster Chart', fontsize=16)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5), title='Glucose Range')
    
    plt.tight_layout()
    return fig

def analyze_hypoglycemia(cgm_df):
    low_glucose = cgm_df[cgm_df['Sensor Glucose (mg/dL)'] < 70]
    very_low_glucose = cgm_df[cgm_df['Sensor Glucose (mg/dL)'] < 54]
    
    low_events = low_glucose.groupby(low_glucose['Timestamp'].dt.date).size()
    very_low_events = very_low_glucose.groupby(very_low_glucose['Timestamp'].dt.date).size()
    
    low_times = low_glucose.groupby(low_glucose['Timestamp'].dt.hour).size()
    very_low_times = very_low_glucose.groupby(very_low_glucose['Timestamp'].dt.hour).size()
    
    return {
        'low_events': low_events,
        'very_low_events': very_low_events,
        'low_times': low_times,
        'very_low_times': very_low_times
    }

def analyze_hyperglycemia(cgm_df):
    high_glucose = cgm_df[cgm_df['Sensor Glucose (mg/dL)'] > 180]
    very_high_glucose = cgm_df[cgm_df['Sensor Glucose (mg/dL)'] > 250]
    
    high_events = high_glucose.groupby(high_glucose['Timestamp'].dt.date).size()
    very_high_events = very_high_glucose.groupby(very_high_glucose['Timestamp'].dt.date).size()
    
    high_times = high_glucose.groupby(high_glucose['Timestamp'].dt.hour).size()
    very_high_times = very_high_glucose.groupby(very_high_glucose['Timestamp'].dt.hour).size()
    
    return {
        'high_events': high_events,
        'very_high_events': very_high_events,
        'high_times': high_times,
        'very_high_times': very_high_times
    }