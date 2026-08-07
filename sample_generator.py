import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

def generate_crowd_scene(num_people: int = 25, width: int = 800, height: int = 600, seed: int = 42) -> np.ndarray:
    """
    Generates a synthetic crowd scene image with specified number of simulated human figures.
    """
    np.random.seed(seed)
    
    # Create background (plaza/stadium aesthetic grid floor)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient background (outdoor plaza tiles)
    for y in range(height):
        r = int(30 + (y / height) * 20)
        g = int(35 + (y / height) * 25)
        b = int(45 + (y / height) * 30)
        img[y, :] = [b, g, r]
        
    # Draw perspective floor grid lines
    for x in range(0, width, 80):
        cv2.line(img, (x, height // 3), (int(width / 2 + (x - width / 2) * 1.8), height), (60, 70, 85), 1)
    for y in range(height // 3, height, 40):
        cv2.line(img, (0, y), (width, y), (55, 65, 75), 1)

    # Draw simulated human figures (head + shoulders silhouette + bounding region)
    locations = []
    for i in range(num_people):
        # Scale figure based on y position (perspective depth)
        y = int(np.random.uniform(height // 3 + 20, height - 40))
        x = int(np.random.uniform(30, width - 30))

        scale = 0.5 + (y / height) * 0.7
        head_radius = int(12 * scale)
        body_width = int(24 * scale)
        body_height = int(45 * scale)

        # Clothes color variety
        color_b = int(np.random.randint(80, 240))
        color_g = int(np.random.randint(80, 240))
        color_r = int(np.random.randint(80, 240))
        shirt_color = (color_b, color_g, color_r)
        skin_color = (180, 210, 240)

        # Torso / body ellipse
        cv2.ellipse(img, (x, y + body_height // 2), (body_width // 2, body_height // 2), 0, 0, 360, shirt_color, -1)
        # Head circle
        cv2.circle(img, (x, y - head_radius // 2), head_radius, skin_color, -1)
        cv2.circle(img, (x, y - head_radius // 2), head_radius, (30, 30, 30), 1)

        locations.append((x, y, scale))

    return img

def create_sample_dataset_folder(output_dir: str = 'sample_scenes'):
    """
    Creates a set of sample crowd images (Low, Medium, High, Critical) in local folder.
    """
    os.makedirs(output_dir, exist_ok=True)

    samples = {
        'low_density_plaza.jpg': 12,
        'moderate_density_event.jpg': 28,
        'high_density_concert.jpg': 55,
        'critical_overcrowded_station.jpg': 90
    }

    generated_files = []
    for filename, count in samples.items():
        filepath = os.path.join(output_dir, filename)
        img = generate_crowd_scene(num_people=count, seed=count * 7)
        cv2.imwrite(filepath, img)
        generated_files.append(filepath)

    return generated_files

if __name__ == '__main__':
    files = create_sample_dataset_folder()
    print(f"Generated {len(files)} sample crowd scene images successfully!")
