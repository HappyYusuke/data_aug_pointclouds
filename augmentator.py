import os
import json
import numpy as np
import copy
import glob
from tqdm import tqdm
from shapely.geometry import Polygon

# ==========================================
# 設定パラメータ
# ==========================================
# 入力ディレクトリ
INPUT_POINT_DIR = '/path/to/your/lidar'
INPUT_LABEL_DIR = '/path/to/your/label'
OUTPUT_ROOT = './dataset_augmented'

OUTPUT_LIDAR_DIR = os.path.join(OUTPUT_ROOT, 'lidar')
OUTPUT_LABEL_DIR = os.path.join(OUTPUT_ROOT, 'label')

AUGMENT_MULTIPLIER = 23     # データ拡張の倍率 (元データが100で1000にしたい場合 => 10-1 = 9倍)
MIN_POINTS_THRESHOLD = 30  # bbox内の最小点群数

TARGET_CLASSES = ['Pedestrian']  # 拡張したいクラス (複数選択可)

# GT-Sampling設定
MAX_PASTE_PERSONS = 20   # 1シーンあたりに追加する人数の最大値 [人] (クラス合計)
PASTE_AREA_X = [-6, 10]  # X座標（前後）の範囲 [m] 
PASTE_AREA_Y = [-6, 6]   # Y座標（左右）の範囲 [m] 

# ローカル変換設定
LOCAL_ROT_RANGE = [-np.pi/20, np.pi/20]   # Z軸回転 (Yaw) [rad]
LOCAL_TILT_RANGE = [-np.pi/36, np.pi/36]  # X/Y軸回転 (Tilt) [rad] (あまり大きくするとBOXからはみ出るので注意: ±5度程度推奨)
LOCAL_SCALE_RANGE = [0.95, 1.05]          # 倍率
LOCAL_TRANS_STD = 0.1                     # 位置ズレ標準偏差 [m]
POINT_NOISE_STD = 0.01                    # 点群ノイズ標準偏差 [m]

# グローバル変換設定
GLOBAL_ROT_RANGE = [-np.pi/4, np.pi/4]  # 全体回転 [rad]
GLOBAL_SCALE_RANGE = [0.95, 1.05]       # 全体スケール
GLOBAL_FLIP_PROB = 0.5                  # 反転確率 (0.0〜1.0)

# 再配置のリトライ回数上限
MAX_POSITION_RETRIES = 10
# ==========================================

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

ensure_dir(OUTPUT_LIDAR_DIR)
ensure_dir(OUTPUT_LABEL_DIR)

# --- 入力データ読み込み関数 ---

def read_pcd(pcd_path):
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('DATA'):
            data_start = i + 1
            break
    points = []
    for line in lines[data_start:]:
        vals = line.strip().split()
        if len(vals) >= 3:
            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
            i_val = float(vals[3]) if len(vals) > 3 else 0.0
            points.append([x, y, z, i_val])
    return np.array(points, dtype=np.float32)

def read_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def read_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.bin':
        return read_bin(path)
    elif ext == '.pcd':
        return read_pcd(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

def parse_kitti_line(line):
    parts = line.strip().split()
    obj_type = parts[0]
    
    h = float(parts[8])
    # ★修正済: lとwの入れ替え(前回の修正を維持)
    l = float(parts[9])
    w = float(parts[10])
    
    x = float(parts[11])
    y = float(parts[12])
    z = float(parts[13]) 
    yaw = float(parts[14])

    z_center = z + (h / 2.0)

    return {
        'l': l, 'w': w, 'h': h,
        'x': x, 'y': y, 'z': z_center,
        'yaw': yaw,
        'cls': obj_type
    }

def read_label(path):
    ext = os.path.splitext(path)[1].lower()
    objects = []
    target_classes_lower = [c.lower() for c in TARGET_CLASSES]

    if ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            for obj in data.get('labels', []):
                if obj['category'].lower() not in target_classes_lower:
                    continue
                
                corrected_yaw = obj['box3d']['orientation']['rotationYaw']
                while corrected_yaw > np.pi: corrected_yaw -= 2 * np.pi
                while corrected_yaw < -np.pi: corrected_yaw += 2 * np.pi
                
                # ★修正済: lとwの入れ替え(前回の修正を維持)
                objects.append({
                    'l': obj['box3d']['dimension']['width'],
                    'w': obj['box3d']['dimension']['length'],
                    'h': obj['box3d']['dimension']['height'],
                    'x': obj['box3d']['location']['x'],
                    'y': obj['box3d']['location']['y'],
                    'z': obj['box3d']['location']['z'],
                    'yaw': corrected_yaw,
                    'cls': obj['category'].capitalize()
                })

    elif ext == '.txt':
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip(): continue
                parts = line.split()
                cls_name = parts[0]
                if cls_name.lower() not in target_classes_lower:
                    continue
                obj_dict = parse_kitti_line(line)
                objects.append(obj_dict)
    
    return objects

# --- 幾何処理関数 ---

def get_box_polygon(box):
    c, s = np.cos(box['yaw']), np.sin(box['yaw'])
    dx2, dy2 = box['l']/2, box['w']/2
    corners = np.array([[dx2, dy2], [dx2, -dy2], [-dx2, -dy2], [-dx2, dy2]])
    rot_mat = np.array([[c, -s], [s, c]])
    corners = np.dot(corners, rot_mat.T) + np.array([box['x'], box['y']])
    return Polygon(corners)

def check_collision(new_box, existing_boxes):
    poly_new = get_box_polygon(new_box)
    for box in existing_boxes:
        if poly_new.intersects(get_box_polygon(box)): return True
    return False

def crop_object_points(points, box):
    pts = points.copy()
    pts[:, 0] -= box['x']
    pts[:, 1] -= box['y']
    pts[:, 2] -= box['z']
    c, s = np.cos(-box['yaw']), np.sin(-box['yaw'])
    x_rot = pts[:, 0] * c - pts[:, 1] * s
    y_rot = pts[:, 0] * s + pts[:, 1] * c
    mask = (np.abs(x_rot) <= box['l']/2) & (np.abs(y_rot) <= box['w']/2) & (np.abs(pts[:, 2]) <= box['h']/2)
    pts_aligned = pts[mask].copy()
    pts_aligned[:, 0] = x_rot[mask]
    pts_aligned[:, 1] = y_rot[mask]
    return pts_aligned

def remove_points_in_box(points, box):
    pts = points.copy()
    pts[:, 0] -= box['x']
    pts[:, 1] -= box['y']
    pts[:, 2] -= box['z']
    c, s = np.cos(-box['yaw']), np.sin(-box['yaw'])
    x_rot = pts[:, 0] * c - pts[:, 1] * s
    y_rot = pts[:, 0] * s + pts[:, 1] * c
    margin = 0.05
    in_box_mask = (np.abs(x_rot) <= (box['l']/2 + margin)) & \
                  (np.abs(y_rot) <= (box['w']/2 + margin)) & \
                  (np.abs(pts[:, 2]) <= (box['h']/2 + margin))
    return ~in_box_mask

def simulate_sparsity(points, orig_dist, new_dist, min_points):
    if new_dist <= orig_dist:
        return points 
    keep_ratio = (orig_dist / new_dist) / 2
    num_points = len(points)
    num_keep = int(num_points * keep_ratio)
    if num_keep < min_points:
        return None 
    indices = np.random.choice(num_points, num_keep, replace=False)
    return points[indices]

# --- 変換ロジック ---

# ★追加: 3次元回転行列の生成関数
def get_rotation_matrix(roll, pitch, yaw):
    # Rz (Yaw)
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c_y, -s_y, 0],
                   [s_y,  c_y, 0],
                   [0,    0,   1]])
    # Ry (Pitch)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[c_p, 0, s_p],
                   [0,   1, 0],
                   [-s_p, 0, c_p]])
    # Rx (Roll)
    c_r, s_r = np.cos(roll), np.sin(roll)
    Rx = np.array([[1, 0,    0],
                   [0, c_r, -s_r],
                   [0, s_r,  c_r]])
    
    # R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def apply_local_transform(points_local, box_params):
    pts = points_local.copy()
    box = box_params.copy()
    
    # 1. Scaling
    scale = np.random.uniform(LOCAL_SCALE_RANGE[0], LOCAL_SCALE_RANGE[1])
    pts[:, :3] *= scale
    box['l'] *= scale; box['w'] *= scale; box['h'] *= scale

    # 2. Rotation (3D: Roll, Pitch, Yaw)
    # ★追加: X軸(Roll)・Y軸(Pitch)の微小回転
    roll = np.random.uniform(LOCAL_TILT_RANGE[0], LOCAL_TILT_RANGE[1])
    pitch = np.random.uniform(LOCAL_TILT_RANGE[0], LOCAL_TILT_RANGE[1])
    yaw_noise = np.random.uniform(LOCAL_ROT_RANGE[0], LOCAL_ROT_RANGE[1])
    
    # 回転行列の計算と適用
    R = get_rotation_matrix(roll, pitch, yaw_noise)
    pts[:, :3] = pts[:, :3] @ R.T  # 点群を回転
    
    # ★注意: KITTI/TAO形式のBoxはYawしか持てないため、Boxのパラメータ更新はYawのみ行う
    # (点群は傾いているが、Boxは直立している状態になる)
    box['yaw'] += yaw_noise

    # 3. Noise & Translation
    noise = np.random.normal(0, POINT_NOISE_STD, pts[:, :3].shape)
    pts[:, :3] += noise
    dx = np.random.normal(0, LOCAL_TRANS_STD)
    dy = np.random.normal(0, LOCAL_TRANS_STD)
    box['x'] += dx; box['y'] += dy
    
    return pts, box

def transform_points_to_world(points_local, box):
    pts = points_local.copy()
    c, s = np.cos(box['yaw']), np.sin(box['yaw'])
    x_rot = pts[:, 0] * c - pts[:, 1] * s
    y_rot = pts[:, 0] * s + pts[:, 1] * c
    pts[:, 0], pts[:, 1] = x_rot, y_rot
    pts[:, 0] += box['x']; pts[:, 1] += box['y']; pts[:, 2] += box['z']
    return pts

# --- メイン処理 ---

def main():
    np.random.seed(42)
    files_pcd = glob.glob(os.path.join(INPUT_POINT_DIR, '*.pcd'))
    files_bin = glob.glob(os.path.join(INPUT_POINT_DIR, '*.bin'))
    all_files = sorted(files_pcd + files_bin)
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
    
    print(f"Target Classes: {TARGET_CLASSES}")
    print(f"Processing {len(file_ids)} files. Output serial number...")

    # 1. DB構築
    print("Building Database...")
    person_db = []
    
    for file_id in tqdm(file_ids):
        pcd_path = os.path.join(INPUT_POINT_DIR, file_id + '.pcd')
        if not os.path.exists(pcd_path): pcd_path = os.path.join(INPUT_POINT_DIR, file_id + '.bin')
        label_path = os.path.join(INPUT_LABEL_DIR, file_id + '.json')
        if not os.path.exists(label_path): label_path = os.path.join(INPUT_LABEL_DIR, file_id + '.txt')

        if not os.path.exists(pcd_path) or not os.path.exists(label_path): continue

        points = read_point_cloud(pcd_path)
        objects = read_label(label_path)

        for box in objects:
            obj_points = crop_object_points(points, box)
            if len(obj_points) >= MIN_POINTS_THRESHOLD:
                dist = np.sqrt(box['x']**2 + box['y']**2)
                person_db.append({'points': obj_points, 'box': box, 'orig_dist': dist})
    
    if not person_db:
        print("No valid objects found in DB matching the target classes.")
        return
    else:
        print(f"DB Size: {len(person_db)} samples.")

    # 2. 拡張ループ
    print("Generating Augmented Dataset...")
    global_idx = 0

    for file_id in tqdm(file_ids):
        pcd_path = os.path.join(INPUT_POINT_DIR, file_id + '.pcd')
        if not os.path.exists(pcd_path): pcd_path = os.path.join(INPUT_POINT_DIR, file_id + '.bin')
        label_path = os.path.join(INPUT_LABEL_DIR, file_id + '.json')
        if not os.path.exists(label_path): label_path = os.path.join(INPUT_LABEL_DIR, file_id + '.txt')
        if not os.path.exists(pcd_path) or not os.path.exists(label_path): continue

        orig_points = read_point_cloud(pcd_path)
        orig_boxes = read_label(label_path)

        bg_mask = np.ones(len(orig_points), dtype=bool)
        orig_person_data = []
        for box in orig_boxes:
            keep_mask = remove_points_in_box(orig_points, box)
            bg_mask = bg_mask & keep_mask
            pts_local = crop_object_points(orig_points, box)
            orig_person_data.append({'points': pts_local, 'box': box})

        current_bg_points = orig_points[bg_mask]

        for aug_idx in range(AUGMENT_MULTIPLIER + 1):
            added_people_points_list = []
            added_people_boxes = []
            
            # A. 既存のオブジェクト (ローカル変換: ここでTiltも適用される)
            for p_data in orig_person_data:
                apt, abox = apply_local_transform(p_data['points'], p_data['box'])
                added_people_points_list.append(transform_points_to_world(apt, abox))
                added_people_boxes.append(abox)
            
            # B. GT-Sampling
            collision_boxes = copy.deepcopy(added_people_boxes)
            num_paste = np.random.randint(1, MAX_PASTE_PERSONS + 1)
            
            for _ in range(num_paste):
                sample = person_db[np.random.randint(len(person_db))]
                success_placement = False
                for retry in range(MAX_POSITION_RETRIES):
                    new_x = np.random.uniform(PASTE_AREA_X[0], PASTE_AREA_X[1])
                    new_y = np.random.uniform(PASTE_AREA_Y[0], PASTE_AREA_Y[1])
                    new_yaw = np.random.uniform(-np.pi, np.pi)
                    new_dist = np.sqrt(new_x**2 + new_y**2)
                    sparse_points = simulate_sparsity(sample['points'], sample['orig_dist'], new_dist, MIN_POINTS_THRESHOLD)
                    if sparse_points is None: continue 
                    
                    temp_box = sample['box'].copy()
                    temp_box['x'], temp_box['y'], temp_box['yaw'] = new_x, new_y, new_yaw
                    apt, abox = apply_local_transform(sparse_points, temp_box)
                    
                    if not check_collision(abox, collision_boxes):
                        added_people_points_list.append(transform_points_to_world(apt, abox))
                        added_people_boxes.append(abox)
                        collision_boxes.append(abox)
                        success_placement = True
                        break 
                
            final_bg_mask = np.ones(len(current_bg_points), dtype=bool)
            for box in added_people_boxes:
                keep_mask = remove_points_in_box(current_bg_points, box)
                final_bg_mask = final_bg_mask & keep_mask
            
            cleaned_bg_points = current_bg_points[final_bg_mask]
            
            if len(added_people_points_list) > 0:
                people_points = np.vstack(added_people_points_list)
                scene_points = np.vstack([cleaned_bg_points, people_points])
            else:
                scene_points = cleaned_bg_points

            # ==========================================
            # C. グローバル変換 (Flip追加)
            # ==========================================
            
            # 1. Flip X (左右反転のような効果: Y軸対称に反転)
            if np.random.rand() < GLOBAL_FLIP_PROB:
                scene_points[:, 0] *= -1
                for b in added_people_boxes:
                    b['x'] *= -1
                    # Yawの変換: Y軸対称反転なので (pi - yaw)
                    b['yaw'] = np.pi - b['yaw']

            # 2. Flip Y (前後反転のような効果: X軸対称に反転)
            if np.random.rand() < GLOBAL_FLIP_PROB:
                scene_points[:, 1] *= -1
                for b in added_people_boxes:
                    b['y'] *= -1
                    # Yawの変換: X軸対称反転なので (-yaw)
                    b['yaw'] = -b['yaw']

            # 3. Global Rotation & Scale
            g_rot = np.random.uniform(GLOBAL_ROT_RANGE[0], GLOBAL_ROT_RANGE[1])
            g_sc = np.random.uniform(GLOBAL_SCALE_RANGE[0], GLOBAL_SCALE_RANGE[1])
            
            c, s = np.cos(g_rot), np.sin(g_rot)
            gx = scene_points[:,0]*c - scene_points[:,1]*s
            gy = scene_points[:,0]*s + scene_points[:,1]*c
            scene_points[:,0], scene_points[:,1] = gx, gy
            scene_points[:,:3] *= g_sc
            
            for b in added_people_boxes:
                bx = b['x']*c - b['y']*s
                by = b['x']*s + b['y']*c
                b['x'], b['y'] = bx*g_sc, by*g_sc
                b['z'] *= g_sc; b['l'] *= g_sc; b['w'] *= g_sc; b['h'] *= g_sc
                b['yaw'] += g_rot

            # 保存 (LiDAR Coords 完全固定版)
            save_name = f"{global_idx:06d}"
            scene_points.astype(np.float32).tofile(os.path.join(OUTPUT_LIDAR_DIR, save_name + '.bin'))
            
            label_lines = []
            for b in added_people_boxes:
                out_yaw = b['yaw']
                while out_yaw > np.pi: out_yaw -= 2 * np.pi
                while out_yaw < -np.pi: out_yaw += 2 * np.pi

                out_x = b['x']
                out_y = b['y']
                out_z = b['z'] - (b['h'] / 2.0)

                line = f"{b['cls']} 0 0 0 0 0 0 0 {b['h']:.3f} {b['w']:.3f} {b['l']:.3f} {out_x:.3f} {out_y:.3f} {out_z:.3f} {out_yaw:.3f}"
                label_lines.append(line)

            with open(os.path.join(OUTPUT_LABEL_DIR, save_name + '.txt'), 'w') as f:
                f.write('\n'.join(label_lines))
            
            global_idx += 1

    print(f"Done. Files saved from 000000.bin to {global_idx-1:06d}.bin in '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    main()
