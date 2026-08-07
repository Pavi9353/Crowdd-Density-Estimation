import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import cv2

from sample_generator import generate_crowd_scene
from crowd_detector import CrowdDetector, generate_density_heatmap, compute_sector_occupancy
from analytics_engine import evaluate_risk_level

def test_crowd_generation():
    img = generate_crowd_scene(num_people=20, width=400, height=300)
    assert img.shape == (300, 400, 3)

def test_crowd_detector():
    detector = CrowdDetector()
    img = generate_crowd_scene(num_people=15, width=400, height=300)
    boxes, centroids = detector.detect(img)
    
    assert isinstance(boxes, list)
    assert isinstance(centroids, list)

def test_heatmap_generation():
    img = generate_crowd_scene(num_people=10, width=400, height=300)
    centroids = [(100, 100), (200, 150), (300, 200)]
    heatmap = generate_density_heatmap(img, centroids)
    
    assert heatmap.shape == img.shape

def test_sector_occupancy():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    centroids = [(50, 50), (300, 50), (50, 300), (300, 300)]
    sectors = compute_sector_occupancy(img, centroids)
    
    assert sectors['Top-Left (Zone A)']['count'] == 1
    assert sectors['Top-Right (Zone B)']['count'] == 1
    assert sectors['Bottom-Left (Zone C)']['count'] == 1
    assert sectors['Bottom-Right (Zone D)']['count'] == 1

def test_risk_evaluation():
    risk_low = evaluate_risk_level(total_count=10, capacity_threshold=50)
    assert risk_low['risk_level'] == "NORMAL"
    
    risk_critical = evaluate_risk_level(total_count=60, capacity_threshold=50)
    assert risk_critical['risk_level'] == "CRITICAL OVERCROWDING"
