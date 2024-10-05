import plotly.graph_objs as go
import pandas as pd
import numpy as np

def plot_gri(cgm_df, glucose_column='Sensor Glucose (mg/dL)'):
    # 提取所需的數據
    glucose_values = pd.to_numeric(cgm_df[glucose_column], errors='coerce')
    
    # 計算 Hypoglycemia 和 Hyperglycemia 組件
    hypoglycemia_component = (glucose_values < 70).sum() / len(glucose_values) * 100
    hyperglycemia_component = (glucose_values > 180).sum() / len(glucose_values) * 100

    # 創建圖形
    fig = go.Figure()

    # 創建漸層背景
    x = np.linspace(0, 100, 1000)
    y = np.linspace(0, 100, 1000)
    xx, yy = np.meshgrid(x, y)
    z = xx + yy

    fig.add_trace(go.Contour(
        z=z,
        x=x,
        y=y,
        colorscale=[
            [0, 'green'],
            [0.2, 'yellow'],
            [0.4, 'orange'],
            [0.6, 'red'],
            [0.8, 'darkred'],
            [1, 'darkred']
        ],
        showscale=False,
        contours=dict(start=0, end=200, size=40, coloring='heatmap'),
        line=dict(width=0),
        opacity=0.5
    ))

    # 添加區域標籤
    zones = ['A', 'B', 'C', 'D', 'E']
    positions = [(10, 10), (30, 30), (50, 50), (70, 70), (90, 90)]
    for zone, (x, y) in zip(zones, positions):
        fig.add_annotation(
            x=x,
            y=y,
            text=f"Zone {zone}",
            showarrow=False,
            font=dict(size=12, color="black")
        )

    # 添加網格線
    for i in range(0, 101, 10):
        fig.add_shape(type="line", x0=0, y0=i, x1=100, y1=i,
                      line=dict(color="white", width=1, dash="dash"))
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=100,
                      line=dict(color="white", width=1, dash="dash"))

    # 添加所有 sensor glucose 數據點
    for _, row in cgm_df.iterrows():
        glucose = pd.to_numeric(row[glucose_column], errors='coerce')
        if pd.notna(glucose):
            hypo = (glucose < 70) * 100
            hyper = (glucose > 180) * 100
            fig.add_trace(go.Scatter(x=[hypo], y=[hyper], mode='markers', 
                                     marker=dict(size=5, color='blue', opacity=0.5),
                                     showlegend=False))

    # 添加平均 GRI 點（黑色星號）
    fig.add_trace(go.Scatter(
        x=[hypoglycemia_component],
        y=[hyperglycemia_component],
        mode='markers',
        marker=dict(size=15, color='black', symbol='star'),
        showlegend=False
    ))

    # 設置圖形佈局
    fig.update_layout(
        title='Glycemia Risk Index (GRI) Components',
        xaxis_title='Hypoglycemia Component (%)',
        yaxis_title='Hyperglycemia Component (%)',
        showlegend=False,
        width=800,
        height=600
    )

    # 設置軸範圍
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])

    return fig