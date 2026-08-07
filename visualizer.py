import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

PLOTLY_TEMPLATE = "plotly_dark"

def plot_risk_gauge(risk_score: int, color_hex: str, risk_level: str) -> go.Figure:
    """
    Creates a semi-circle indicator Gauge chart for Safety Risk Index.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Safety Risk Index</b><br><span style='font-size:0.8em;color:{color_hex}'>{risk_level}</span>", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color_hex},
            'bgcolor': "#1E293B",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 90], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [90, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=260,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_sector_bar_chart(sectors: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Creates a bar chart comparing headcount across Zone A, B, C, D.
    """
    labels = list(sectors.keys())
    counts = [sectors[k]['count'] for k in labels]
    percentages = [sectors[k]['percentage'] for k in labels]

    df_sec = pd.DataFrame({
        'Zone': labels,
        'Headcount': counts,
        'Percentage': percentages
    })

    fig = px.bar(
        df_sec,
        x='Zone',
        y='Headcount',
        text='Headcount',
        color='Headcount',
        color_continuous_scale='Viridis',
        title='<b>Zone-Wise Crowd Distribution</b>',
        template=PLOTLY_TEMPLATE
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Crowd Sector / Zone",
        yaxis_title="Estimated People Count",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_crowd_timeline_chart(df_logs: pd.DataFrame) -> go.Figure:
    """
    Creates a time-series area chart tracking headcount over time.
    """
    if df_logs.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_logs['Timestamp'],
        y=df_logs['Estimated_Headcount'],
        mode='lines+markers',
        name='Headcount',
        line=dict(color='#6366F1', width=3),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.2)'
    ))

    fig.update_layout(
        title='<b>Crowd Headcount Timeline</b>',
        xaxis_title='Timestamp / Frame',
        yaxis_title='People Count',
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig
