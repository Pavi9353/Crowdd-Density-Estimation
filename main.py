import os
import sys
import cv2
import numpy as np
from datetime import datetime
import time

# Reconfigure stdout for UTF-8 compatibility on Windows Command Prompt if possible
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Optional Tkinter File Dialog for easy local file picking
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# Local imports
from crowd_detector import CrowdDetector, generate_density_heatmap, draw_bounding_boxes, compute_sector_occupancy
from analytics_engine import evaluate_risk_level, create_crowd_log_record

def print_banner():
    print("=" * 72)
    print("  👥 CROWD DENSITY ESTIMATION SYSTEM - LOCAL FILES & LIVE VIDEO")
    print("=" * 72)

def select_local_file(file_types: str = "images") -> str:
    """
    Opens native Windows file picker dialog to easily select photos or videos from local system.
    """
    if TK_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            if file_types == "images":
                title = "Select a Photo/Image from your Computer"
                ftypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
            else:
                title = "Select a Video file from your Computer"
                ftypes = [("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]

            file_path = filedialog.askopenfilename(title=title, filetypes=ftypes)
            root.destroy()
            if file_path:
                return file_path
        except Exception:
            pass

    # Fallback to manual typing
    prompt = "Enter full file path from your system (or press Enter to cancel): "
    path = input(prompt).strip().strip('"').strip("'")
    return path

def process_local_photo(image_path: str, capacity_threshold: int = 50):
    """
    Processes a real photo from the user's system and renders density heatmaps & stats.
    """
    if not os.path.exists(image_path):
        print(f"\n[!] Error: File '{image_path}' does not exist on your system.")
        return

    print(f"\n[*] Reading Photo: {os.path.basename(image_path)}")
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print("[!] Error: Could not load image file. Please check file format.")
        return

    detector = CrowdDetector()
    boxes, centroids = detector.detect(img_bgr)
    total_count = len(centroids)

    risk_info = evaluate_risk_level(total_count, capacity_threshold=capacity_threshold)
    sectors = compute_sector_occupancy(img_bgr, centroids)

    print("-" * 72)
    print(f" [+] PHOTO FILE             : {os.path.abspath(image_path)}")
    print(f" [+] REAL-TIME HEADCOUNT    : {total_count} People Detected")
    print(f" [+] VENUE MAX CAPACITY     : {capacity_threshold} People")
    print(f" [+] OCCUPANCY RATE (%)     : {risk_info['occupancy_percentage']}%")
    print(f" [!] SAFETY RISK LEVEL      : [{risk_info['risk_level']}] (Score: {risk_info['risk_score']}/100)")
    print(f" [*] STATUS ADVISORY        : {risk_info['recommendation']}")
    print("-" * 72)
    print(" [::] SECTOR OCCUPANCY BREAKDOWN:")
    for zone_name, info in sectors.items():
        print(f"      • {zone_name:<22} : {info['count']} people ({info['percentage']}%)")
    print("-" * 72)

    # Heatmap & Bounding Box overlays
    heatmap_overlay = generate_density_heatmap(img_bgr, centroids, alpha=0.55, sigma=35)
    bbox_img = draw_bounding_boxes(img_bgr, boxes)

    # Save output to local folder
    out_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, heatmap_overlay)
    print(f" [✓] Output Density Heatmap saved to: {out_path}")

    # Display side-by-side OpenCV window
    print("\n [🖥️] Displaying Window... Press ANY KEY (or ESC) on the image window to close.")
    
    # Scale for viewing
    h, w = img_bgr.shape[:2]
    target_w, target_h = 600, int(600 * (h / float(w)))
    
    left_view = cv2.resize(bbox_img, (target_w, target_h))
    right_view = cv2.resize(heatmap_overlay, (target_w, target_h))

    # Add HUD headers
    cv2.putText(left_view, f"Detected: {total_count} People", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(right_view, f"Risk: {risk_info['risk_level']}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    combined_view = np.hstack((left_view, right_view))
    cv2.imshow("Crowd Density Estimation - Photo Detection (Left) | Spatial Heatmap (Right)", combined_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video_source(source_input, capacity_threshold: int = 50, is_webcam: bool = False):
    """
    Processes real video files or live webcam feed frame-by-frame with real-time density estimation.
    """
    cap = cv2.VideoCapture(source_input)

    if not cap.isOpened():
        print(f"\n[!] Error: Could not open video source: {source_input}")
        return

    source_name = "Live Webcam Feed" if is_webcam else os.path.basename(str(source_input))
    print(f"\n[*] Starting Real-time Video Stream: {source_name}")
    print(" [ℹ] Press 'Q' or 'ESC' on the video window at any time to STOP processing.\n")

    detector = CrowdDetector()
    frame_count = 0
    fps_start_time = time.time()
    
    # Frame skip factor for faster real-time processing
    skip_frames = 2
    last_centroids = []
    last_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[✓] Video stream processing complete or end of file reached.")
            break

        frame_count += 1

        # Process detection on specified frames for speed
        if frame_count % skip_frames == 0 or not last_centroids:
            boxes, centroids = detector.detect(frame)
            last_boxes, last_centroids = boxes, centroids
        else:
            boxes, centroids = last_boxes, last_centroids

        total_count = len(centroids)
        risk_info = evaluate_risk_level(total_count, capacity_threshold=capacity_threshold)

        # Generate overlays
        heatmap_frame = generate_density_heatmap(frame, centroids, alpha=0.5, sigma=30)
        bbox_frame = draw_bounding_boxes(frame, boxes)

        # Build Real-time Heads Up Display (HUD) overlay
        h, w = frame.shape[:2]
        target_w, target_h = 640, int(640 * (h / float(w)))

        left_disp = cv2.resize(bbox_frame, (target_w, target_h))
        right_disp = cv2.resize(heatmap_frame, (target_w, target_h))

        # Real-time stats text overlay
        fps = frame_count / (time.time() - fps_start_time + 1e-5)
        
        cv2.rectangle(left_disp, (0, 0), (target_w, 45), (15, 23, 42), -1)
        cv2.putText(left_disp, f"FPS: {fps:.1f} | Frame: {frame_count} | Headcount: {total_count} People", 
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.rectangle(right_disp, (0, 0), (target_w, 45), (15, 23, 42), -1)
        cv2.putText(right_disp, f"Risk Level: {risk_info['risk_level']} ({risk_info['occupancy_percentage']}%)", 
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        combined = np.hstack((left_disp, right_disp))
        cv2.imshow(f"Real-Time Crowd Density - {source_name} (Press Q to quit)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("\n[!] User stopped video processing.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print_banner()

    while True:
        print("\nChoose how you want to test Crowd Density Estimation:")
        print(" [1] 🖼️  Select PHOTO File from your System (File Picker)")
        print(" [2] 🎬 Select VIDEO File from your System (File Picker)")
        print(" [3] 📹 Test Live System WEBCAM Feed")
        print(" [4] 🎲 Quick Synthetic Test Scene")
        print(" [5] ❌ Exit")

        choice = input("\nEnter choice [1-5]: ").strip()

        if choice == '1':
            print("\n[ Picker ] Opening Windows File Dialog... Please select an image file.")
            path = select_local_file(file_types="images")
            if path:
                process_local_photo(path)
            else:
                print("No photo selected.")

        elif choice == '2':
            print("\n[ Picker ] Opening Windows File Dialog... Please select a video file.")
            path = select_local_file(file_types="video")
            if path:
                process_video_source(path, is_webcam=False)
            else:
                print("No video selected.")

        elif choice == '3':
            print("\n[ Webcam ] Connecting to default system camera (Webcam index 0)...")
            process_video_source(0, is_webcam=True)

        elif choice == '4':
            from sample_generator import create_sample_dataset_folder
            files = create_sample_dataset_folder('sample_scenes')
            process_local_photo(files[2])

        elif choice == '5':
            print("\nExiting Crowd Density Estimation System. Goodbye!\n")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.")

if __name__ == '__main__':
    main()
