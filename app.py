import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from datetime import datetime

# Import local modules
from sample_generator import create_sample_dataset_folder, generate_crowd_scene
from crowd_detector import CrowdDetector, generate_density_heatmap, draw_bounding_boxes, compute_sector_occupancy
from analytics_engine import evaluate_risk_level, create_crowd_log_record, compute_historical_trend_summary
from visualizer import plot_risk_gauge, plot_sector_bar_chart, plot_crowd_timeline_chart

# Top-level WSGI/ASGI Compatibility Exports for Cloud Platforms (Vercel, Render, Railway)
app = st
application = st
handler = st

# 1. Page Config
st.set_page_config(
    page_title="Crowd Density Estimation System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, sans-serif;
    }
    .header-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(239, 68, 68, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #F43F5E, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Session State for Audit Logs
if 'audit_logs' not in st.session_state:
    st.session_state.audit_logs = []

# Sidebar Setup & Controls
st.sidebar.markdown("## ⚙️ Dashboard Controls")

input_mode = st.sidebar.radio(
    "Select Input Source",
    ["📷 Upload Real Photo (JPG/PNG)", "🎬 Upload Real Video (MP4/AVI)", "🎲 Built-in Test Scenes"],
    index=0
)

# Detection Hyperparameters
capacity_threshold = st.sidebar.slider("Venue Max Capacity Threshold", min_value=10, max_value=500, value=50, step=5)
sigma_blur = st.sidebar.slider("Spatial Heatmap Blur Radius (Sigma)", min_value=10, max_value=60, value=35, step=5)
heatmap_alpha = st.sidebar.slider("Heatmap Overlay Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Initialize Crowd Detector
detector = CrowdDetector()

input_img_bgr = None
frame_label = "Real Source"

if input_mode == "📷 Upload Real Photo (JPG/PNG)":
    uploaded_file = st.sidebar.file_uploader("Upload Image File from your System", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_img_bgr = cv2.imdecode(file_bytes, 1)
        frame_label = uploaded_file.name
    else:
        st.info("👈 Please select and upload a Photo file from your computer using the sidebar button.")
        input_img_bgr = generate_crowd_scene(30, seed=12)
        frame_label = "Sample Image"

elif input_mode == "🎬 Upload Real Video (MP4/AVI)":
    uploaded_video = st.sidebar.file_uploader("Upload Video File from your System", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        temp_video_path = os.path.join("output_results", "temp_upload_video.mp4")
        os.makedirs("output_results", exist_ok=True)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
            
        cap = cv2.VideoCapture(temp_video_path)
        ret, input_img_bgr = cap.read()
        cap.release()
        frame_label = uploaded_video.name
    else:
        st.info("👈 Please select and upload a Video file from your computer using the sidebar button.")
        input_img_bgr = generate_crowd_scene(45, seed=77)
        frame_label = "Sample Video Frame"

else:
    sample_files = create_sample_dataset_folder('sample_scenes')
    scene_options = {
        "🟢 Low Density Plaza (12 People)": sample_files[0],
        "🟡 Moderate Density Event (28 People)": sample_files[1],
        "🟠 High Density Concert (55 People)": sample_files[2],
        "🔴 Critical Overcrowded Station (90 People)": sample_files[3]
    }
    selected_scene = st.sidebar.selectbox("Select Sample Scene", list(scene_options.keys()))
    input_img_bgr = cv2.imread(scene_options[selected_scene])
    frame_label = selected_scene

# Main Header
st.markdown("""
<div class="header-box">
    <h1 class="header-title">👥 Crowd Density Estimation & Safety Analytics System</h1>
    <p style="color: #94A3B8; margin-top: 6px;">AI-driven spatial crowd counting, sector occupancy analysis, and real-time density heatmaps.</p>
</div>
""", unsafe_allow_html=True)

if input_img_bgr is None:
    st.warning("⚠️ No data available matching the selected filters. Please adjust sidebar filter settings.")
    st.stop()

# Perform Detection & Analytics
boxes, centroids = detector.detect(input_img_bgr)
total_count = len(centroids)

# Density Heatmap & Bounding Boxes Overlay
heatmap_img_bgr = generate_density_heatmap(input_img_bgr, centroids, alpha=heatmap_alpha, sigma=sigma_blur)
bbox_img_bgr = draw_bounding_boxes(input_img_bgr, boxes)

# Risk Evaluation
risk_info = evaluate_risk_level(total_count, capacity_threshold=capacity_threshold)
sectors = compute_sector_occupancy(input_img_bgr, centroids)

# Log entry
log_record = create_crowd_log_record(frame_label, total_count, risk_info, sectors)
if not any(entry['Frame_Source'] == frame_label and entry['Estimated_Headcount'] == total_count for entry in st.session_state.audit_logs):
    st.session_state.audit_logs.append(log_record)

# Status Recommendation Alert Banner
st.markdown(f"""
<div style="background-color: {risk_info['color']}20; border: 2px solid {risk_info['color']}; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
    <h3 style="color: {risk_info['color']}; margin:0;">{risk_info['status_icon']} {risk_info['risk_level']} - {risk_info['occupancy_percentage']}% Occupancy</h3>
    <p style="color: #E2E8F0; margin-top: 6px; font-weight: 500;">{risk_info['recommendation']}</p>
</div>
""", unsafe_allow_html=True)

# KPI Cards Row
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Estimated Headcount</div>
        <div class="metric-val">👥 {total_count}</div>
        <div class="metric-sub" style="color:#38BDF8">People Detected</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Venue Occupancy Rate</div>
        <div class="metric-val">{risk_info['occupancy_percentage']}%</div>
        <div class="metric-sub" style="color:{risk_info['color']}">Capacity: {capacity_threshold}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Safety Risk Score</div>
        <div class="metric-val">{risk_info['risk_score']} / 100</div>
        <div class="metric-sub" style="color:{risk_info['color']}">{risk_info['risk_level']}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    max_zone = max(sectors.items(), key=lambda x: x[1]['count'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Highest Density Zone</div>
        <div class="metric-val">{max_zone[1]['count']}</div>
        <div class="metric-sub" style="color:#F59E0B">📍 {max_zone[0].split(' ')[0]}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dual View Display
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("📷 Detection & Bounding Boxes")
    bbox_rgb = cv2.cvtColor(bbox_img_bgr, cv2.COLOR_BGR2RGB)
    st.image(bbox_rgb, use_container_width=True, caption=f"Original Feed with Centroid Points ({total_count} Detected)")

with c_right:
    st.subheader("🔥 Spatial Density Heatmap (KDE)")
    heatmap_rgb = cv2.cvtColor(heatmap_img_bgr, cv2.COLOR_BGR2RGB)
    st.image(heatmap_rgb, use_container_width=True, caption="Spatial Density Heatmap Overlay (Jet Colormap)")

st.markdown("---")

# Navigation Tabs
t1, t2, t3 = st.tabs(["📊 Zone Distribution & Risk", "📈 Crowd Timeline Analytics", "📄 Audit Log & Export"])

with t1:
    col_g1, col_g2 = st.columns([5, 7])
    with col_g1:
        fig_gauge = plot_risk_gauge(risk_info['risk_score'], risk_info['color'], risk_info['risk_level'])
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_g2:
        fig_bar = plot_sector_bar_chart(sectors)
        st.plotly_chart(fig_bar, use_container_width=True)

with t2:
    st.subheader("📈 Historical Crowd Count Trend")
    df_history = compute_historical_trend_summary(st.session_state.audit_logs)
    if not df_history.empty:
        fig_timeline = plot_crowd_timeline_chart(df_history)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No timeline records available yet.")

with t3:
    st.subheader("📄 Crowd Density Audit Log")
    df_logs_export = compute_historical_trend_summary(st.session_state.audit_logs)
    st.dataframe(df_logs_export, use_container_width=True)
    
    if not df_logs_export.empty:
        csv_buffer = io.StringIO()
        df_logs_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Audit Log CSV",
            data=csv_buffer.getvalue(),
            file_name=f"crowd_density_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
