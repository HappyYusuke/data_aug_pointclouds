import os
import json
import numpy as np
import copy
import glob
from tqdm import tqdm
from shapely.geometry import Polygon

# ==========================================
# ■ 設定パラメータ
# ==========================================
# 入力ディレクトリ
INPUT_POINT_DIR = '/path/to/your/lidar'
INPUT_LABEL_DIR = '/path/to/your/label'
OUTPUT_ROOT = './dataset_augmented_v2'  # 出力先を変更

OUTPUT_LIDAR_DIR = os.path.join(OUTPUT_ROOT, 'lidar')
OUTPUT_LABEL_TAO_DIR = os.path.join(OUTPUT_ROOT, 'label')
OUTPUT_LABEL_VIS_DIR = os.path.join(OUTPUT_ROOT, 'label_lidar')

AUGMENT_MULTIPLIER = 15    # データ拡張の倍率 (元データが100で1000にしたい場合 => 10-1 = 9倍)
MIN_POINTS_THRESHOLD = 30  # bbox内の最小点群数

# 対象クラス
TARGET_CLASSES = ['Person'] 

# GT-Sampling設定
MAX_PASTE_PERSONS = 4   # 1シーンあたりに追加する人数の最大値 [人]
PASTE_AREA_X = [-6, 6]  # X座標（前後）の範囲 [m]
PASTE_AREA_Y = [-6, 6]  # # Y座標（左右）の範囲 [m]

# ローカル変換設定 (個別のオブジェクトに対する変換)
LOCAL_ROT_X_RANGE = [-np.pi/60, np.pi/60]  # Roll (±3度程度) [rad]
LOCAL_ROT_Y_RANGE = [-np.pi/60, np.pi/60]  # Pitch (±3度程度) [rad]
LOCAL_ROT_Z_RANGE = [-np.pi/20, np.pi/20]  # Yaw (Z軸回転) [rad]
LOCAL_SCALE_RANGE = [0.95, 1.05]  # 倍率
LOCAL_TRANS_STD = 0.1             # 位置ズレ標準偏差 [m]
POINT_NOISE_STD = 0.01            # 点群ノイズ標準偏差 [m]

# グローバル変換設定 (シーン全体に対する変換)
GLOBAL_ROT_RANGE = [-np.pi/4, np.pi/4]  # 全体回転 [rad]
GLOBAL_SCALE_RANGE = [0.95, 1.05]       # 全体スケールの倍率
GLOBAL_FLIP_PROB = 0.5                  # ランダム反転の確率

# 再配置のリトライ回数上限
MAX_POSITION_RETRIES = 10


def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

ensure_dir(OUTPUT_LIDAR_DIR)
ensure_dir(OUTPUT_LABEL_TAO_DIR)
ensure_dir(OUTPUT_LABEL_VIS_DIR)

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
    w = float(parts[9])
    l = float(parts[10])
    x_cam = float(parts[11])
    y_cam = float(parts[12])
    z_cam = float(parts[13])
    ry = float(parts[14])
    
    x_lidar = z_cam
    y_lidar = -x_cam
    z_lidar = -y_cam + (h / 2.0)
    yaw = -ry - (np.pi / 2.0)
    while yaw > np.pi: yaw -= 2 * np.pi
    while yaw < -np.pi: yaw += 2 * np.pi

    return {
        'l': l, 'w': w, 'h': h,
        'x': x_lidar, 'y': y_lidar, 'z': z_lidar,
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
                objects.append({
                    'l': obj['box3d']['dimension']['length'],
                    'w': obj['box3d']['dimension']['width'],
                    'h': obj['box3d']['dimension']['height'],
                    'x': obj['box3d']['location']['x'],
                    'y': obj['box3d']['location']['y'],
                    'z': obj['box3d']['location']['z'],
                    'yaw': obj['box3d']['orientation']['rotationYaw'],
                    'cls': obj['category'].capitalize()
                })
    elif ext == '.txt':
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip(): continue
                parts = line.split()
                if parts[0].lower() not in target_classes_lower: continue
                objects.append(parse_kitti_line(line))
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
    if new_dist <= orig_dist: return points 
    keep_ratio = (orig_dist / new_dist) / 2
    num_points = len(points)
    num_keep = int(num_points * keep_ratio)
    if num_keep < min_points: return None 
    indices = np.random.choice(num_points, num_keep, replace=False)
    return points[indices]

# --- ★追加: 3次元回転行列 ---
def rotate_points_3d(points, rx, ry, rz):
    """
    points: (N, 3+) array
    rx, ry, rz: radians
    """
    # Rotation matrix around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    # Rotation matrix around Y-axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    # Rotation matrix around Z-axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    
    points[:, :3] = points[:, :3] @ R.T
    return points

# --- 変換ロジック (更新版) ---

def apply_local_transform(points_local, box_params):
    pts = points_local.copy()
    box = box_params.copy()
    
    # 1. スケーリング
    scale = np.random.uniform(LOCAL_SCALE_RANGE[0], LOCAL_SCALE_RANGE[1])
    pts[:, :3] *= scale
    box['l'] *= scale; box['w'] *= scale; box['h'] *= scale
    
    # 2. 3次元回転 (X, Y, Z)
    # Z軸回転(Yaw)は箱の向きにも反映させる
    rot_z = np.random.uniform(LOCAL_ROT_Z_RANGE[0], LOCAL_ROT_Z_RANGE[1])
    # X, Y軸回転は点群のみに適用 (箱は直立のまま = KITTI仕様の制約)
    rot_x = np.random.uniform(LOCAL_ROT_X_RANGE[0], LOCAL_ROT_X_RANGE[1])
    rot_y = np.random.uniform(LOCAL_ROT_Y_RANGE[0], LOCAL_ROT_Y_RANGE[1])
    
    pts = rotate_points_3d(pts, rot_x, rot_y, rot_z)
    
    # 箱のYawのみ更新
    box['yaw'] += rot_z
    
    # 3. ノイズ付加
    noise = np.random.normal(0, POINT_NOISE_STD, pts[:, :3].shape)
    pts[:, :3] += noise
    
    # 4. 中心位置のズレ
    dx = np.random.normal(0, LOCAL_TRANS_STD)
    dy = np.random.normal(0, LOCAL_TRANS_STD)
    box['x'] += dx; box['y'] += dy
    
    return pts, box

def transform_points_to_world(points_local, box):
    pts = points_local.copy()
    # 既にローカルで回転済みなので、ワールド配置時は箱のYawだけ考慮して配置
    # (crop時に box['yaw'] で逆回転させているため、ここではその分を戻す必要があるが
    #  apply_local_transform でさらに回転させているため、単純な座標移動とYaw回転で配置する)
    
    # ここでは「配置」のための座標変換を行う
    # apply_local_transformで得られたptsは「Yaw回転後の箱」基準のローカル座標になっている
    # box['yaw'] は初期Yaw + 追加Yaw になっている
    
    # 単純化のため、ここでは box['yaw'] を使って回転させて配置する
    # ただし points_local は既に回転済みなので、ここでの回転は「箱の配置角度」への合わせ込み
    
    # 修正: crop_object_points は box['yaw'] をキャンセルして axis-aligned にしている。
    # apply_local_transform はそこで回転を加えている。
    # なので、ここでは「配置先の座標 (box['x'], box['y'])」へ移動させるだけでよいはずだが、
    # crop時の回転キャンセル分（元のbox['yaw']）を戻す必要があるか？
    # -> apply_local_transformで box['yaw'] を更新しているため、
    #    ここでは更新された box['yaw'] を使って回転させて配置するのが正しい。
    #    ただし、apply_local_transform 内の rotate_points_3d ですでに回転させてしまっている。
    #    これが二重回転にならないように注意が必要。
    
    # ロジック見直し: 
    # 1. crop: World -> Local (Yawキャンセル)
    # 2. apply_local: Local -> Local' (Scale, Noise, 3D Rotation)
    #    ここで pts は回転している。box['yaw'] も更新している。
    # 3. transform_to_world: Local' -> World
    #    apply_local で回転させた pts を、そのままワールドに平行移動させると、
    #    box['yaw'] の向きと合わなくなる可能性がある。
    
    # 正しい手順:
    # apply_local では「微小な回転ノイズ(x,y,z)」を点群に与えるだけにする。
    # box['yaw'] の更新は「配置時の回転」として扱う。
    
    # 今回の実装（簡易版）:
    # apply_local で rot_z を pts に適用済み。
    # transform_to_world では、pts を box['x'], box['y'] に平行移動する前に、
    # 「配置のための回転」が必要。
    # 元の box['yaw'] (配置角) + rot_z (微小回転) = 新しい box['yaw']
    # pts は既に rot_z 分だけ回っている。
    # なので、ここでは「元の box['yaw']」分だけ回す必要がある？
    # いや、crop 時に yaw をキャンセルして真っ直ぐにしている。
    # なので、ワールドに戻すときは「新しい box['yaw']」で回せばOK。
    # しかし apply_local で既に回してしまっている...
    
    # ★修正★ apply_local_transform の rotate_points_3d から Rz (Yaw) を外します。
    # Yaw回転は transform_points_to_world で一括で行うほうが安全です。
    pass 

# --- 再定義: ロジック整合性のため修正 ---

def apply_local_transform(points_local, box_params):
    pts = points_local.copy()
    box = box_params.copy()
    
    scale = np.random.uniform(LOCAL_SCALE_RANGE[0], LOCAL_SCALE_RANGE[1])
    pts[:, :3] *= scale
    box['l'] *= scale; box['w'] *= scale; box['h'] *= scale
    
    # X軸, Y軸の回転 (点群のみのノイズ)
    rot_x = np.random.uniform(LOCAL_ROT_X_RANGE[0], LOCAL_ROT_X_RANGE[1])
    rot_y = np.random.uniform(LOCAL_ROT_Y_RANGE[0], LOCAL_ROT_Y_RANGE[1])
    
    # Z軸回転 (Yaw) の決定
    rot_z = np.random.uniform(LOCAL_ROT_Z_RANGE[0], LOCAL_ROT_Z_RANGE[1])
    
    # 点群には X, Y 回転だけ適用する (Z回転はワールド配置時に行う)
    pts = rotate_points_3d(pts, rot_x, rot_y, 0) 
    
    # 箱のYaw情報を更新
    box['yaw'] += rot_z
    
    noise = np.random.normal(0, POINT_NOISE_STD, pts[:, :3].shape)
    pts[:, :3] += noise
    
    dx = np.random.normal(0, LOCAL_TRANS_STD)
    dy = np.random.normal(0, LOCAL_TRANS_STD)
    box['x'] += dx; box['y'] += dy
    
    return pts, box

def transform_points_to_world(points_local, box):
    pts = points_local.copy()
    # ここで Yaw (Z軸) 回転を適用してワールド座標系に戻す
    c, s = np.cos(box['yaw']), np.sin(box['yaw'])
    x_rot = pts[:, 0] * c - pts[:, 1] * s
    y_rot = pts[:, 0] * s + pts[:, 1] * c
    pts[:, 0], pts[:, 1] = x_rot, y_rot
    
    # 平行移動
    pts[:, 0] += box['x']
    pts[:, 1] += box['y']
    pts[:, 2] += box['z']
    return pts


# --- メイン処理 ---

def main():
    np.random.seed(42)
    
    files_pcd = glob.glob(os.path.join(INPUT_POINT_DIR, '*.pcd'))
    files_bin = glob.glob(os.path.join(INPUT_POINT_DIR, '*.bin'))
    all_files = sorted(files_pcd + files_bin)
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
    
    print(f"Target Classes: {TARGET_CLASSES}")
    print(f"Processing {len(file_ids)} files...")

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
        print("No valid objects found.")
        return

    # 2. 拡張ループ
    print("Generating Augmented Dataset...")
    global_idx = 0

    for file_id in tqdm(file_ids):
        pcd_path = os.path.join(INPUT_POINT_DIR, file_id + '.pcd')
        if not os.path.exists(pcd_path): pcd_path = os.path.join(INPUT_POINT_DIR, file_id + '.bin')
        label_path = os.path.join(INPUT_LABEL_DIR, file_id + '.json')
        if not os.path.exists(label_path): label_path = os.path.join(INPUT_LABEL_DIR, file_id + '.txt')
        
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
            
            # A. 既存オブジェクト配置
            for p_data in orig_person_data:
                apt, abox = apply_local_transform(p_data['points'], p_data['box'])
                added_people_points_list.append(transform_points_to_world(apt, abox))
                added_people_boxes.append(abox)
            
            # B. GT-Sampling
            collision_boxes = copy.deepcopy(added_people_boxes)
            num_paste = np.random.randint(1, MAX_PASTE_PERSONS + 1)
            
            for _ in range(num_paste):
                sample = person_db[np.random.randint(len(person_db))]
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
                        break
            
            # 背景削除 (追加した人の位置にある背景点を消す)
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

            # C. グローバル変換 (全体回転・スケール・★反転)
            g_rot = np.random.uniform(GLOBAL_ROT_RANGE[0], GLOBAL_ROT_RANGE[1])
            g_sc = np.random.uniform(GLOBAL_SCALE_RANGE[0], GLOBAL_SCALE_RANGE[1])
            
            # ★追加: ランダム反転 (Flip along X-axis = Y座標反転)
            # 確率的に反転フラグを立てる
            do_flip = np.random.random() < GLOBAL_FLIP_PROB

            # 回転行列
            c, s = np.cos(g_rot), np.sin(g_rot)
            
            # 点群への適用
            # 1. 回転
            gx = scene_points[:,0]*c - scene_points[:,1]*s
            gy = scene_points[:,0]*s + scene_points[:,1]*c
            scene_points[:,0], scene_points[:,1] = gx, gy
            
            # 2. スケール
            scene_points[:,:3] *= g_sc
            
            # 3. 反転 (Y座標を反転)
            if do_flip:
                scene_points[:, 1] *= -1

            # ボックスへの適用
            for b in added_people_boxes:
                # 1. 回転
                bx = b['x']*c - b['y']*s
                by = b['x']*s + b['y']*c
                
                # 2. スケール
                b['x'], b['y'] = bx*g_sc, by*g_sc
                b['z'] *= g_sc; b['l'] *= g_sc; b['w'] *= g_sc; b['h'] *= g_sc
                b['yaw'] += g_rot
                
                # 3. 反転 (Y座標反転 ＆ Yaw反転)
                if do_flip:
                    b['y'] *= -1
                    b['yaw'] = -b['yaw'] # Y軸反転ならYawも符号反転

            # 保存
            save_name = f"{global_idx:06d}"
            scene_points.astype(np.float32).tofile(os.path.join(OUTPUT_LIDAR_DIR, save_name + '.bin'))
            
            label_lines_tao = []
            for b in added_people_boxes:
                x_cam = -b['y']
                y_cam = -(b['z'] - b['h'] / 2.0)
                z_cam = b['x']
                ry = -b['yaw'] - (np.pi / 2)
                while ry > np.pi: ry -= 2*np.pi
                while ry < -np.pi: ry += 2*np.pi
                line = f"{b['cls']} 0.00 0 0.00 0.00 0.00 0.00 0.00 {b['h']:.3f} {b['w']:.3f} {b['l']:.3f} {x_cam:.3f} {y_cam:.3f} {z_cam:.3f} {ry:.3f}"
                label_lines_tao.append(line)
            with open(os.path.join(OUTPUT_LABEL_TAO_DIR, save_name + '.txt'), 'w') as f: f.write('\n'.join(label_lines_tao))
            
            global_idx += 1

    print(f"Done. Files saved to '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    main()
