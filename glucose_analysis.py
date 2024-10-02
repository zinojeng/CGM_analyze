import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
        'Time < 50': (x['Sensor Glucose (mg/dL)'] < 50).mean() * 100
    }))
    
    daily_stats = daily_stats[['Time > 250', 'TAR (181-250)', 'TIR (70-180)', 'TBR (54-69)', 'Time < 50']]
    
    fig, ax = plt.subplots(figsize=(12, 6))
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
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    return fig