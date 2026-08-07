import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

def evaluate_risk_level(total_count: int, capacity_threshold: int = 50) -> Dict[str, Any]:
    """
    Evaluates crowd density risk score, level label, color code, and recommendation text.
    """
    occupancy_pct = (total_count / float(capacity_threshold)) * 100 if capacity_threshold > 0 else 0.0

    if occupancy_pct <= 40.0:
        level = "NORMAL"
        color = "#10B981"  # Emerald Green
        status_icon = "🟢"
        recommendation = "Safe crowd conditions. Normal venue operations."
        risk_score = min(100, int(occupancy_pct * 0.7))
        
    elif occupancy_pct <= 70.0:
        level = "MODERATE"
        color = "#F59E0B"  # Amber / Yellow
        status_icon = "🟡"
        recommendation = "Crowd monitoring advised. Ensure exit routes remain clear."
        risk_score = int(30 + (occupancy_pct - 40) * 1.0)
        
    elif occupancy_pct <= 95.0:
        level = "HIGH DENSITY"
        color = "#F97316"  # Orange Warning
        status_icon = "🟠"
        recommendation = "High density warning. Restrict new entries into the venue."
        risk_score = int(60 + (occupancy_pct - 70) * 1.2)
        
    else:
        level = "CRITICAL OVERCROWDING"
        color = "#EF4444"  # Red Alert
        status_icon = "🔴"
        recommendation = "DANGER: Critical overcrowding detected! Dispatch security & initiate crowd dispersal protocols!"
        risk_score = min(100, int(90 + (occupancy_pct - 95) * 1.5))

    return {
        'total_count': total_count,
        'capacity_threshold': capacity_threshold,
        'occupancy_percentage': round(occupancy_pct, 1),
        'risk_level': level,
        'risk_score': risk_score,
        'color': color,
        'status_icon': status_icon,
        'recommendation': recommendation
    }

def create_crowd_log_record(frame_name: str, total_count: int, risk_info: Dict[str, Any], sectors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a log entry row for audit logging and CSV export.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        'Timestamp': now,
        'Frame_Source': frame_name,
        'Estimated_Headcount': total_count,
        'Occupancy_Rate (%)': risk_info['occupancy_percentage'],
        'Risk_Level': risk_info['risk_level'],
        'Risk_Score (0-100)': risk_info['risk_score'],
        'Zone_A_Count': sectors.get('Top-Left (Zone A)', {}).get('count', 0),
        'Zone_B_Count': sectors.get('Top-Right (Zone B)', {}).get('count', 0),
        'Zone_C_Count': sectors.get('Bottom-Left (Zone C)', {}).get('count', 0),
        'Zone_D_Count': sectors.get('Bottom-Right (Zone D)', {}).get('count', 0)
    }
    return record

def compute_historical_trend_summary(logs_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts list of crowd log records into a pandas DataFrame.
    """
    if not logs_list:
        return pd.DataFrame(columns=[
            'Timestamp', 'Frame_Source', 'Estimated_Headcount', 
            'Occupancy_Rate (%)', 'Risk_Level', 'Risk_Score (0-100)'
        ])
    return pd.DataFrame(logs_list)
