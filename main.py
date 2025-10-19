import os
import numpy as np
import cv2
import json
import open3d as o3d
import pandas as pd
from ultralytics import YOLO

# --- Paths ---
Input_images = r"KITTI-360_sample\data_2d_raw\2013_05_28_drive_0000_sync\image_01\data_rect"
Output_folder = r"car_detections"
Lidar_folder = r"KITTI-360_sample\data_3d_raw\2013_05_28_drive_0000_sync\velodyne_points\data"
Bounding_Boxes = r"KITTI-360_sample\bboxes_3D_cam0"
Excel_output = r"car_stats.xlsx"
os.makedirs(Output_folder, exist_ok=True)

# --- Calibration parameters ---
fx = 552.554261
fy = 552.554261
cx = 682.049453
cy = 238.769549
IMAGE_WIDTH = 1408
IMAGE_HEIGHT = 376

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

# --- Helper Functions ---
def load_lidar_point_cloud(lidar_file):
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

def find_matching_lidar_file(image_filename, lidar_path):
    image_id = os.path.splitext(os.path.basename(image_filename))[0]
    lidar_files = sorted(os.listdir(lidar_path))
    min_diff = float('inf')
    closest_file = None
    for file in lidar_files:
        try:
            diff = abs(int(image_id) - int(os.path.splitext(file)[0]))
            if diff < min_diff:
                min_diff = diff
                closest_file = file
        except:
            continue
    return os.path.join(lidar_path, closest_file)

def load_3d_boxes(image_filename):
    base_name = os.path.splitext(os.path.basename(image_filename))[0].lstrip('0') or "0"
    json_path = os.path.join(Bounding_Boxes, f"BBoxes_{base_name}.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        return json.load(f)

def lidar_to_camera(points, T_velo_to_cam, R_rect_00):
    points_homog = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (T_velo_to_cam @ points_homog.T).T[:, :3]
    points_rect = (R_rect_00 @ points_cam.T).T
    return points_rect

def create_bbox_lineset(corners, color):
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set

def is_point_in_3dbox(points, box_corners):
    min_corner = np.min(box_corners, axis=0)
    max_corner = np.max(box_corners, axis=0)
    return np.all((points >= min_corner) & (points <= max_corner), axis=1)

def get_dominant_color(point_colors, color_palette, color_names):
    if len(point_colors) == 0:
        return "None"
    diffs = np.linalg.norm(point_colors[:, None, :] - color_palette[None, :, :], axis=2)
    closest_idx = np.argmin(diffs, axis=1)
    idx, counts = np.unique(closest_idx, return_counts=True)
    if len(counts) == 0:
        return "None"
    dominant_idx = idx[np.argmax(counts)]
    return color_names[dominant_idx]

# --- Custom Color Palette (highly distinguishable) ---
color_palette = np.array([
    [1, 0, 0],       # Red
    [0, 0, 1],       # Blue
    [0, 1, 0],       # Green
    [1, 1, 0],       # Yellow
    [1, 0, 1],       # Pink/Magenta
    [0, 1, 1],       # Cyan
    [1, 0.5, 0],     # Orange
    [0.5, 0, 0.5],   # Purple
    [0.6, 0.3, 0],   # Brown
    [0, 0, 0],       # Black
    [0.5, 0.5, 0.5], # Gray
    [0, 0.2, 0.4],   # Dark Blue
    [0, 0.3, 0],     # Dark Green
    [0.3, 0, 0],     # Dark Red
    [0.2, 0.2, 0.2], # Dark Gray
], dtype=float)

color_names = [
    "Red", "Blue", "Green", "Yellow", "Pink", "Cyan", "Orange", "Purple",
    "Brown", "Black", "Gray", "Dark Blue", "Dark Green", "Dark Red", "Dark Gray"
]

# --- Load YOLO segmentation model ---
model = YOLO("yolo11x-seg.pt")

load_image = sorted([os.path.join(Input_images, f) for f in os.listdir(Input_images) if f.endswith(".png")])
print(f"Found {len(load_image)} images to process.")

all_stats = []

for idx, image_path in enumerate(load_image):
    print(f"\nProcessing {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_path}: could not read image")
        continue

    lidar_file = find_matching_lidar_file(image_path, Lidar_folder)
    if lidar_file is None:
        print(f"Skipping {image_path}: could not find matching LiDAR file")
        continue

    point_cloud = load_lidar_point_cloud(lidar_file)
    points = point_cloud[:, :3]
    points_cam = lidar_to_camera(points, T_velo_to_cam, R_rect_00)

    # --- Project all LiDAR points to image ---
    u = (fx * points_cam[:, 0] / np.maximum(points_cam[:, 2], 1e-10) + cx).astype(int)
    v = (fy * points_cam[:, 1] / np.maximum(points_cam[:, 2], 1e-10) + cy).astype(int)
    valid = (points_cam[:, 2] > 0) & (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
    u_valid = u[valid]
    v_valid = v[valid]
    points_cam_valid = points_cam[valid]

    # --- Run YOLO segmentation ---
    res = model.predict(image, classes=[2], device="cpu") # class 2 is car in COCO
    if not res or res[0].masks is None:
        print("No masks detected")
        continue
    masks = res[0].masks.data.cpu().numpy()
    cls = res[0].boxes.cls.cpu().numpy().astype(int)
    car_ids = [i for i, c in enumerate(cls) if c == 2]

    # --- Color all points gray by default ---
    point_colors = np.full((len(points), 3), 0.7) # default gray for all points

    # --- Color points inside any car mask (inside or outside BB) ---
    for i in car_ids:
        m = masks[i]
        m_full = cv2.resize(m, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(bool)
        hits = np.zeros(len(points), dtype=bool)
        hits[valid] = m_full[v_valid, u_valid]
        point_colors[hits] = color_palette[i % len(color_palette)]

    # --- 3D BBs, sort by lowest Z to highest Z (arrow side = bottom) ---
    boxes = load_3d_boxes(image_path)
    if not boxes:
        print("No BBs found.")
        continue
    box_zs = [np.min(np.array(b["corners_cam0"])[:,2]) for b in boxes]
    sorted_indices = np.argsort(box_zs)
    boxes_sorted = [boxes[i] for i in sorted_indices]

    # --- Per-car (instance) statistics & Excel output ---
    print("\n--- Per-car (instance) statistics ---")
    total_inside = 0
    total_outside = 0
    in_any_mask = np.zeros(len(points_cam_valid), dtype=bool)
    valid_car_count = 0
    per_image_stats = []
    for car_num, bbox in enumerate(boxes_sorted, 1):
        color_idx = (car_num - 1) % len(color_palette)
        color_name = color_names[color_idx]
        corners = np.array(bbox["corners_cam0"])
        in_bb = is_point_in_3dbox(points_cam_valid, corners)
        detected = np.zeros(np.sum(in_bb), dtype=bool)
        if len(car_ids) > 0:
            for i in car_ids:
                m = masks[i]
                m_full = cv2.resize(m, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(bool)
                hits = m_full[v_valid[in_bb], u_valid[in_bb]]
                detected = detected | hits
        inside = np.sum(detected)
        outside = np.sum(in_bb) - inside

        if inside + outside == 0:
            continue

        valid_car_count += 1
        total_inside += inside
        total_outside += outside
        inside_indices = np.where(in_bb)[0][detected]
        dom_color = get_dominant_color(point_colors[valid][inside_indices], color_palette, color_names)
        if (inside + outside) > 0:
            precision = inside / (inside + outside)
            recall = inside / (inside + outside)
        else:
            precision = recall = 0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        print(f"Car {car_num} ({color_name} BB, pt:{dom_color}): {inside} points inside YOLO mask, {outside} points outside | Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        per_image_stats.append({
            "ImageNo": idx+1,
            "CarNo": car_num,
            "BBColor": color_name,
            "PointsInBB": inside + outside,
            "DominantPtColor": dom_color,
            "InsideMask": inside,
            "OutsideMask": outside,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })
        in_any_mask[in_bb] = in_any_mask[in_bb] | detected

    if valid_car_count == 0:
        print("No valid cars with points inside or outside.")
        continue

    if per_image_stats:
        df = pd.DataFrame(per_image_stats)
        if os.path.exists(Excel_output):
            df_old = pd.read_excel(Excel_output)
            df = pd.concat([df_old, df], ignore_index=True)
        df.to_excel(Excel_output, index=False)
        print(f"Saved current image stats to {Excel_output}")

    print(f"\nAggregate for all cars in image:")
    print(f"  Total points inside any car BB and YOLO mask: {total_inside}")
    print(f"  Total points inside any car BB but outside YOLO mask: {total_outside}")

    print("\n--- Image metrics ---")
    correct = total_inside
    in_mask = np.zeros(len(points_cam_valid), dtype=bool)
    for i in car_ids:
        m = masks[i]
        m_full = cv2.resize(m, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(bool)
        hits = m_full[v_valid, u_valid]
        in_mask = in_mask | hits
    false_positives = np.sum(in_mask & ~in_any_mask)
    false_negatives = total_outside
    background = len(points_cam_valid) - (correct + false_positives + false_negatives)

    if (correct + false_positives) > 0:
        precision = correct / (correct + false_positives)
    else:
        precision = 0
    if (correct + false_negatives) > 0:
        recall = correct / (correct + false_negatives)
    else:
        recall = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print(f"Correct (inside mask and box, GREEN): {correct}")
    print(f"False positives (inside mask, outside box, RED): {false_positives}")
    print(f"False negatives (inside box, outside mask, BLUE): {false_negatives}")
    print(f"Background (outside both, GRAY): {background}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")

    # --- Open Open3D window ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam)  # Use all points
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    geometries = [pcd]
    for box_idx, bbox in enumerate(boxes_sorted):
        corners = np.array(bbox["corners_cam0"])
        color = color_palette[box_idx % len(color_palette)]
        geometries.append(create_bbox_lineset(corners, color))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in geometries:
        vis.add_geometry(geom)
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.run()
    vis.destroy_window()

    # --- Save output image with colored points ---
    try:
        output_image = image.copy()
        for i in range(len(u_valid)):
            color_tuple = tuple(int(x) for x in (point_colors[valid][i].ravel() * 255))
            assert len(color_tuple) == 3, "color_tuple must have 3 elements"
            cv2.circle(output_image, (u_valid[i], v_valid[i]), 2, color_tuple, -1)

        output_path = os.path.join(Output_folder, f"output_image_{idx+1}.png")
        cv2.imwrite(output_path, output_image)
        print(f"Saved image to {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        continue
