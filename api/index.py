import os
import sys
import base64
import cv2
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse

# Top-level 'app' export required by Vercel @vercel/python builder
app = FastAPI(
    title="Crowd Density Estimation System",
    description="Vercel Serverless AI Crowd Density Estimation & Analytics API",
    version="1.0.0"
)

# Lightweight Serverless Detector for Vercel (OpenCV DNN & Cascades)
class VercelCrowdDetector:
    def __init__(self):
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = self._load_cascade(os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'))
        self.profile_cascade = self._load_cascade(os.path.join(cascade_dir, 'haarcascade_profileface.xml'))
        self.upperbody_cascade = self._load_cascade(os.path.join(cascade_dir, 'haarcascade_upperbody.xml'))

    def _load_cascade(self, path: str):
        if os.path.exists(path) and hasattr(cv2, 'CascadeClassifier'):
            try:
                return cv2.CascadeClassifier(path)
            except Exception:
                return None
        return None

    def detect(self, image: np.ndarray):
        if image is None or image.size == 0:
            return [], []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        candidate_boxes = []
        candidate_scores = []

        cascades = [
            (self.face_cascade, 1.08, 3, 0.90),
            (self.profile_cascade, 1.10, 3, 0.80),
            (self.upperbody_cascade, 1.10, 2, 0.75)
        ]

        for cas, scale_f, min_n, conf in cascades:
            if cas is not None:
                try:
                    detected = cas.detectMultiScale(
                        gray_eq, scaleFactor=scale_f, minNeighbors=min_n, minSize=(20, 20)
                    )
                    for (x, y, bw, bh) in detected:
                        ebw, ebh = int(bw * 1.3), int(bh * 2.0)
                        ex = max(0, x - int(bw * 0.15))
                        ey = max(0, y - int(bh * 0.15))
                        candidate_boxes.append((ex, ey, ebw, ebh))
                        candidate_scores.append(conf)
                except Exception:
                    pass

        return self._apply_nms(candidate_boxes, candidate_scores, w, h)

    def _apply_nms(self, boxes, scores, img_w, img_h):
        if not boxes:
            return [], []
        boxes_np = np.array(boxes, dtype=np.int32)
        scores_np = np.array(scores, dtype=np.float32)

        indices = []
        try:
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_np.tolist(),
                scores=scores_np.tolist(),
                score_threshold=0.20,
                nms_threshold=0.35
            )
            if isinstance(indices, np.ndarray):
                indices = indices.flatten().tolist()
        except Exception:
            indices = list(range(len(boxes)))

        final_boxes, final_centroids = [], []
        for idx in indices:
            x, y, bw, bh = boxes_np[idx]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            bw = max(10, min(bw, img_w - x))
            bh = max(10, min(bh, img_h - y))
            cx, cy = int(x + bw // 2), int(y + bh // 2)
            final_boxes.append((int(x), int(y), int(bw), int(bh)))
            final_centroids.append((cx, cy))

        return final_boxes, final_centroids

detector = VercelCrowdDetector()

def evaluate_risk(total_count: int, capacity_threshold: int = 50):
    occupancy_pct = (total_count / float(capacity_threshold)) * 100 if capacity_threshold > 0 else 0.0
    if occupancy_pct <= 40.0:
        level, color, icon = "NORMAL", "#10B981", "🟢"
        recommendation = "Safe crowd flow conditions. Normal venue operations."
        risk_score = min(100, int(occupancy_pct * 0.7))
    elif occupancy_pct <= 70.0:
        level, color, icon = "MODERATE", "#F59E0B", "🟡"
        recommendation = "Crowd monitoring advised. Ensure clear exit pathways."
        risk_score = int(30 + (occupancy_pct - 40) * 1.0)
    elif occupancy_pct <= 95.0:
        level, color, icon = "HIGH DENSITY", "#F97316", "🟠"
        recommendation = "High density warning. Restrict new entries into the venue."
        risk_score = int(60 + (occupancy_pct - 70) * 1.2)
    else:
        level, color, icon = "CRITICAL OVERCROWDING", "#EF4444", "🔴"
        recommendation = "DANGER: Critical overcrowding detected! Dispatch security & initiate crowd dispersal protocols!"
        risk_score = min(100, int(90 + (occupancy_pct - 95) * 1.5))

    return {
        'total_count': total_count,
        'capacity_threshold': capacity_threshold,
        'occupancy_percentage': round(occupancy_pct, 1),
        'risk_level': level,
        'risk_score': risk_score,
        'color': color,
        'status_icon': icon,
        'recommendation': recommendation
    }

def generate_heatmap(image: np.ndarray, centroids, alpha: float = 0.5, sigma: int = 35):
    h, w = image.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    if not centroids:
        return image.copy()
    for (cx, cy) in centroids:
        if 0 <= cx < w and 0 <= cy < h:
            density_map[cy, cx] += 1.0
    kernel_size = sigma * 2 + 1
    density_map = cv2.GaussianBlur(density_map, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    norm_map = (density_map / density_map.max() * 255).astype(np.uint8) if density_map.max() > 0 else np.zeros((h, w), dtype=np.uint8)
    color_heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 1.0 - alpha, color_heatmap, alpha, 0)
    for (cx, cy) in centroids:
        cv2.circle(blended, (cx, cy), 6, (0, 255, 255), -1)
    return blended

def draw_boxes(image: np.ndarray, boxes):
    output = image.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, f"Person #{i}", (x + 3, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Crowd Density Estimation - Vercel Serverless Edition</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #0F172A; color: #F8FAFC; font-family: system-ui, sans-serif; padding-bottom: 40px; }
            .header-box { background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(239, 68, 68, 0.15)); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 24px; margin-top: 24px; margin-bottom: 24px; }
            .card-custom { background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 20px; }
            .metric-val { font-size: 1.8rem; font-weight: 800; }
            .preview-img { width: 100%; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-box">
                <h1 class="text-primary fw-bold">👥 Crowd Density Estimation System</h1>
                <p class="text-secondary mb-0">Vercel Serverless AI Cloud Edition</p>
            </div>
            <div class="row">
                <div class="col-md-5 mb-3">
                    <div class="card-custom">
                        <h4 class="mb-3">📷 Upload Crowd Photo</h4>
                        <form id="uploadForm">
                            <div class="mb-3">
                                <label class="form-label text-secondary">Select Image File</label>
                                <input type="file" class="form-control bg-dark text-white" id="imageInput" accept="image/*" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-secondary">Venue Max Capacity: <span id="capVal">50</span></label>
                                <input type="range" class="form-range" id="capacityInput" min="10" max="500" value="50" oninput="document.getElementById('capVal').innerText = this.value">
                            </div>
                            <button type="submit" class="btn btn-primary w-100 fw-bold">🚀 Analyze Crowd Density</button>
                        </form>
                    </div>
                </div>
                <div class="col-md-7 mb-3">
                    <div class="card-custom" id="resultsBox" style="display:none;">
                        <h4 class="mb-3">📊 Analytics Summary</h4>
                        <div class="alert alert-info" id="statusAlert"></div>
                        <div class="row text-center">
                            <div class="col-4"><small class="text-secondary">HEADCOUNT</small><div class="metric-val" id="headcountVal">0</div></div>
                            <div class="col-4"><small class="text-secondary">OCCUPANCY</small><div class="metric-val" id="occupancyVal">0%</div></div>
                            <div class="col-4"><small class="text-secondary">RISK SCORE</small><div class="metric-val" id="riskVal">0/100</div></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row" id="imagePreviewRow" style="display:none;">
                <div class="col-md-6 mb-3">
                    <div class="card-custom">
                        <h5>📷 Detections</h5>
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

                const res = await fetch('/api/detect', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.error) { alert(data.error); return; }

                document.getElementById('resultsBox').style.display = 'block';
                document.getElementById('imagePreviewRow').style.display = 'flex';
                document.getElementById('headcountVal').innerText = data.total_count;
                document.getElementById('occupancyVal').innerText = data.risk_info.occupancy_percentage + '%';
                document.getElementById('riskVal').innerText = data.risk_info.risk_score + '/100';
                document.getElementById('statusAlert').innerText = data.risk_info.status_icon + ' ' + data.risk_info.risk_level + ': ' + data.risk_info.recommendation;
                document.getElementById('bboxImg').src = 'data:image/jpeg;base64,' + data.bbox_b64;
                document.getElementById('heatmapImg').src = 'data:image/jpeg;base64,' + data.heatmap_b64;
            });
        </script>
    </body>
    </html>
    """

@app.post("/api/detect")
async def detect_api(file: UploadFile = File(...), capacity: int = Form(50)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})

        boxes, centroids = detector.detect(img_bgr)
        total_count = len(centroids)
        risk_info = evaluate_risk(total_count, capacity_threshold=capacity)

        heatmap_bgr = generate_heatmap(img_bgr, centroids, alpha=0.55, sigma=35)
        bbox_bgr = draw_boxes(img_bgr, boxes)

        _, bbox_enc = cv2.imencode('.jpg', bbox_bgr)
        bbox_b64 = base64.b64encode(bbox_enc).decode('utf-8')

        _, heatmap_enc = cv2.imencode('.jpg', heatmap_bgr)
        heatmap_b64 = base64.b64encode(heatmap_enc).decode('utf-8')

        return JSONResponse(content={
            "total_count": total_count,
            "risk_info": risk_info,
            "bbox_b64": bbox_b64,
            "heatmap_b64": heatmap_b64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
