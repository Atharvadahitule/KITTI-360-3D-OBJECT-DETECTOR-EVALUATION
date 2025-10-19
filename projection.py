import numpy as np
import os
import cv2
from ultralytics import YOLO

# ------------------- PATHS -------------------
Input_images = r"KITTI-360_sample\data_2d_raw\2013_05_28_drive_0000_sync\image_00\data_rect"
Output_folder = r"Projection_images"
Lidar_folder = r"KITTI-360_sample\data_3d_raw\2013_05_28_drive_0000_sync\velodyne_points\data"

# ------------------- CAMERA INTRINSICS -------------------
fx = 552.554261
fy = 552.554261
cx = 682.049453
cy = 238.769549
IMAGE_WIDTH = 1408
IMAGE_HEIGHT = 376

# ------------------- EXTRINSIC & RECTIFICATION -------------------
calib_cam_to_velo = np.array([
    [0.043071, -0.088293, 0.995163, 0.804391],
    [-0.999004, 0.007785, 0.043928, 0.299349],
    [-0.011625, -0.996064, -0.087870, -0.177023]
])
calib_cam_to_velo_4x4 = np.vstack([calib_cam_to_velo, [0, 0, 0, 1]])
T_velo_to_cam = np.linalg.inv(calib_cam_to_velo_4x4)
R_rect_00 = np.array([
    [0.999974, -0.007141, -0.000089],
    [0.007141, 0.999969, -0.003247],
    [0.000112, 0.003247, 0.999995]
])

# ------------------- YOLO MODEL -------------------
model = YOLO("yolo11x-seg.pt")

# ------------------- UTILS -------------------
def load_lidar_point_cloud_with_intensity(lidar_file):
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

def find_matching_lidar_file(image_filename, lidar_path):
    image_id = os.path.splitext(os.path.basename(image_filename))[0]
    lidar_files = sorted(os.listdir(lidar_path))
    closest = None
    min_diff = float("inf")
    for f in lidar_files:
        try:
            diff = abs(int(image_id) - int(os.path.splitext(f)[0]))
            if diff < min_diff:
                min_diff = diff
                closest = f
        except:
            continue
    return os.path.join(lidar_path, closest) if closest else None

def intensity_to_rainbow_color(intensities):
    norm = (intensities - intensities.min()) / (np.ptp(intensities) + 1e-8)
    norm_uint8 = np.clip(norm * 255, 0, 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_RAINBOW)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

def project_lidar_to_image(pc_intensity):
    points = pc_intensity[:, :3]
    intensities = pc_intensity[:, 3]
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (T_velo_to_cam @ points_hom.T).T[:, :3]
    points_rect = (R_rect_00 @ points_cam.T).T
    mask = points_rect[:, 2] > 0
    points_rect = points_rect[mask]
    intensities = intensities[mask]
    z = points_rect[:, 2]
    u = (fx * points_rect[:, 0] / z + cx).astype(int)
    v = (fy * points_rect[:, 1] / z + cy).astype(int)
    in_bounds = (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
    return u[in_bounds], v[in_bounds], intensities[in_bounds]

# ------------------- MAIN LOOP -------------------
if not os.path.exists(Output_folder):
    os.makedirs(Output_folder)

image_files = [os.path.join(Input_images, f) for f in sorted(os.listdir(Input_images)) if f.endswith(".png")]

for idx, image_path in enumerate(image_files):
    print(f"\nProcessing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        continue

    # --- Run YOLOv8 Segmentation ---
    results = model(image)[0]

    if results.masks is None:
        print("No segmentation masks found. Skipping this image.")
        continue

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names = model.names


    # Keep only masks for 'car' class
    car_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    for i, cls in enumerate(classes):
        if names[cls] == 'car':
            resized_mask = cv2.resize(masks[i].astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            car_mask |= resized_mask.astype(bool)

    lidar_file = find_matching_lidar_file(image_path, Lidar_folder)
    if lidar_file is None or not os.path.exists(lidar_file):
        print("LiDAR file not found.")
        continue

    pc = load_lidar_point_cloud_with_intensity(lidar_file)
    u, v, intensities = project_lidar_to_image(pc)
    if len(u) == 0:
        print("No valid LiDAR projections.")
        continue

    colors = intensity_to_rainbow_color(intensities)

    # ----------- TOP IMAGE (rainbow depth map) -----------
    top_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    for i in range(len(u)):
        color = tuple(map(int, colors[i].flatten()))
        cv2.circle(top_image, (u[i], v[i]), 1, color, -1)
    cv2.putText(top_image, "Projected Depth Map (Rainbow)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # ----------- BOTTOM IMAGE (projected car points only) -----------
    bottom_image = (image * 0.3).astype(np.uint8)  # dim the image
    for i in range(len(u)):
        if car_mask[v[i], u[i]]:
            color = tuple(map(int, colors[i].flatten()))
            cv2.circle(bottom_image, (u[i], v[i]), 1, color, -1)
    cv2.putText(bottom_image, "Car-only LiDAR Projection on RGB", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # ----------- COMBINE AND SAVE -----------
    combined = np.vstack([top_image, bottom_image])
    output_path = os.path.join(Output_folder, f"projection_{idx+1:03d}.png")
    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")
