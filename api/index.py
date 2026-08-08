import os
import sys
import base64
import cv2
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse

# Add parent directory to sys.path to import local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crowd_detector import CrowdDetector, generate_density_heatmap, draw_bounding_boxes, compute_sector_occupancy
from analytics_engine import evaluate_risk_level

# Top-level 'app' export required by Vercel @vercel/python builder
app = FastAPI(
    title="Crowd Density Estimation System",
    description="Vercel Serverless AI Crowd Density Estimation & Analytics API",
    version="1.0.0"
)

detector = CrowdDetector()

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crowd Density Estimation & Analytics</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #0F172A;
                color: #F8FAFC;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                padding-bottom: 40px;
            }
            .header-box {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(239, 68, 68, 0.15) 100%);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 28px;
                margin-top: 24px;
                margin-bottom: 28px;
            }
            .header-title {
                font-size: 2.2rem;
                font-weight: 800;
                background: linear-gradient(90deg, #6366F1, #F43F5E, #38BDF8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .card-custom {
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .metric-val {
                font-size: 1.8rem;
                font-weight: 800;
                color: #F8FAFC;
            }
            .btn-gradient {
                background: linear-gradient(90deg, #6366F1, #3B82F6);
                border: none;
                color: white;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 8px;
            }
            .btn-gradient:hover {
                background: linear-gradient(90deg, #4F46E5, #2563EB);
                color: white;
            }
            .preview-img {
                width: 100%;
                border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.1);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-box">
                <h1 class="header-title">👥 Crowd Density Estimation System</h1>
                <p class="text-secondary mb-0">Vercel AI Serverless Analytics & Heatmap Studio</p>
            </div>

            <div class="row mb-4">
                <div class="col-md-5">
                    <div class="card-custom">
                        <h4 class="mb-3">📷 Upload Crowd Photo</h4>
                        <form id="uploadForm">
                            <div class="mb-3">
                                <label class="form-label text-secondary">Select Image File (JPG/PNG)</label>
                                <input type="file" class="form-control bg-dark text-white border-secondary" id="imageInput" accept="image/*" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-secondary">Venue Capacity Threshold: <span id="capVal">50</span></label>
                                <input type="range" class="form-range" id="capacityInput" min="10" max="500" value="50" oninput="document.getElementById('capVal').innerText = this.value">
                            </div>
                            <button type="submit" class="btn btn-gradient w-100">🚀 Analyze Crowd Density</button>
                        </form>
                    </div>
                </div>
                <div class="col-md-7">
                    <div class="card-custom" id="resultsBox" style="display:none;">
                        <h4 class="mb-3">📊 Analytics Summary</h4>
                        <div class="alert alert-info" id="statusAlert"></div>
                        <div class="row text-center mb-3">
                            <div class="col-4">
                                <small class="text-secondary">HEADCOUNT</small>
                                <div class="metric-val" id="headcountVal">0</div>
                            </div>
                            <div class="col-4">
                                <small class="text-secondary">OCCUPANCY</small>
                                <div class="metric-val" id="occupancyVal">0%</div>
                            </div>
                            <div class="col-4">
                                <small class="text-secondary">RISK SCORE</small>
                                <div class="metric-val" id="riskVal">0/100</div>
                            </div>
                        </div>
                        <h5 class="text-secondary mt-3">📍 Zonal Distribution</h5>
                        <ul class="list-group list-group-flush bg-transparent" id="zoneList"></ul>
                    </div>
                </div>
            </div>

            <div class="row" id="imagePreviewRow" style="display:none;">
                <div class="col-md-6 mb-3">
                    <div class="card-custom">
                        <h5>📷 Detections & Bounding Boxes</h5>
                        <img id="bboxImg" class="preview-img mt-2">
                    </div>
                </div>
                <div class="col-md-6 mb-3">
                    <div class="card-custom">
                        <h5>🔥 Spatial Density Heatmap (KDE)</h5>
                        <img id="heatmapImg" class="preview-img mt-2">
                    </div>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const fileInput = document.getElementById('imageInput');
                const capacity = document.getElementById('capacityInput').value;

                if (!fileInput.files[0]) return;

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('capacity', capacity);

                const res = await fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();
                if (data.error) {
                    alert(data.error);
                    return;
                }

                document.getElementById('resultsBox').style.display = 'block';
                document.getElementById('imagePreviewRow').style.display = 'flex';

                document.getElementById('headcountVal').innerText = data.total_count;
                document.getElementById('occupancyVal').innerText = data.risk_info.occupancy_percentage + '%';
                document.getElementById('riskVal').innerText = data.risk_info.risk_score + '/100';
                document.getElementById('statusAlert').innerText = data.risk_info.status_icon + ' ' + data.risk_info.risk_level + ': ' + data.risk_info.recommendation;

                let zoneHtml = '';
                for (let key in data.sectors) {
                    zoneHtml += `<li class="list-group-item bg-transparent text-white border-secondary d-flex justify-content-between">
                        <span>${key}</span>
                        <span class="badge bg-primary rounded-pill">${data.sectors[key].count} people (${data.sectors[key].percentage}%)</span>
                    </li>`;
                }
                document.getElementById('zoneList').innerHTML = zoneHtml;

                document.getElementById('bboxImg').src = 'data:image/jpeg;base64,' + data.bbox_b64;
                document.getElementById('heatmapImg').src = 'data:image/jpeg;base64,' + data.heatmap_b64;
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/api/detect")
async def detect_api(file: UploadFile = File(...), capacity: int = Form(50)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})

        # Run AI Detection
        boxes, centroids = detector.detect(img_bgr)
        total_count = len(centroids)

        # Risk and Sector Analytics
        risk_info = evaluate_risk_level(total_count, capacity_threshold=capacity)
        sectors = compute_sector_occupancy(img_bgr, centroids)

        # Overlays
        heatmap_bgr = generate_density_heatmap(img_bgr, centroids, alpha=0.55, sigma=35)
        bbox_bgr = draw_bounding_boxes(img_bgr, boxes)

        # Encode images to Base64 for Web Display
        _, bbox_encoded = cv2.imencode('.jpg', bbox_bgr)
        bbox_b64 = base64.b64encode(bbox_encoded).decode('utf-8')

        _, heatmap_encoded = cv2.imencode('.jpg', heatmap_bgr)
        heatmap_b64 = base64.b64encode(heatmap_encoded).decode('utf-8')

        return JSONResponse(content={
            "total_count": total_count,
            "risk_info": risk_info,
            "sectors": sectors,
            "bbox_b64": bbox_b64,
            "heatmap_b64": heatmap_b64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
