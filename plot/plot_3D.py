import plotly.graph_objects as go
import numpy as np
def draw_3D_trajetory(data,i):
    fig = go.Figure()

    impact_x,impact_y,impact_z=data['impact']
    label_x,label_y,label_z=data['label']
    x,y,z=data['points']


    fig.add_trace(go.Scatter3d(
        x=[0],  # X坐标固定为0
        y=[impact_y],
        z=[impact_z],
        mode='markers',
        name=f'predicted {i}',
        marker=dict(
            size=8,
            color='red',  # 预测点颜色
            symbol='circle'  # 圆形标记
        ),
        showlegend=True
    ))
    fig.add_trace(go.Scatter3d(
        x=[0],  # X坐标固定为0
        y=[label_y],
        z=[label_z],
        mode='markers',
        name=f'real {i}',
        marker=dict(
            size=8,
            color='green',  # 预测点颜色
            symbol='circle'  # 圆形标记
        ),
        showlegend=True
    ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        name=f'observation {i}',
        line=dict(width=2)
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[0,4],dtick=1),
            yaxis=dict(title='Y', range=[-1,1],dtick=1),
            zaxis=dict(title='Z', range=[0,3],dtick=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))  # 设定视角

        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.add_trace(go.Scatter3d(
        x=[0],  # X坐标固定为0
        y=[0],
        z=[0],
        mode='markers',
        name=f'camera',
        marker=dict(
            size=20,
            color='yellow',  # 预测点颜色
            symbol='circle'  # 圆形标记
        ),
        showlegend=True
    ))
    fig.show()