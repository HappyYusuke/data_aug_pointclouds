import os
import json
import numpy as np
import copy
import argparse
from tqdm import tqdm
from shapely.geometry import Polygon

# ==========================================
# ■ 設定パラメータ
# ==========================================
INPUT_PCD_DIR = './pcd_json/pointclouds'
INPUT_JSON_DIR = './pcd_json/annotations'
OUTPUT_ROOT = './dataset_augmented'

OUTPUT_LIDAR_DIR = os.path.join(OUTPUT_ROOT, 'lidar')
OUTPUT_LABEL_TAO_DIR = os.path.join(OUTPUT_ROOT, 'label')
OUTPUT_LABEL_VIS_DIR = os.path.join(OUTPUT_ROOT, 'label_lidar')

AUGMENT_MULTIPLIER = 9
MIN_POINTS_THRESHOLD = 20

# GT-Sampling設定
MAX_PASTE_PERSONS = 10    # 1シーンあたりに追加する人数の最大値 単位: [人]
PASTE_AREA_X = [-6, 6]  # 原点から見た、人を配置するX座標（前後）の範囲 [m] 
PASTE_AREA_Y = [-6, 6]  # 原点から見た、人を配置するY座標（左右）の範囲 [m]

# ローカル変換設定
LOCAL_ROT_RANGE = [-np.pi/20, np.pi/20]  # 個別の人間を回転させる角度の範囲 [rad]
LOCAL_SCALE_RANGE = [0.95, 1.05]         # サイズを変更する倍率（[0.95, 1.05] は 「95% 〜 105%」）
LOCAL_TRANS_STD = 0.1                    # 人間の配置位置を微小にずらす際の標準偏差 [m]
POINT_NOISE_STD = 0.01                   # 個々の点群の位置をノイズでずらす際の標準偏差 [m]

# グローバル変換設定
GLOBAL_ROT_RANGE = [-np.pi/4, np.pi/4]   # シーン全体を回転させる角度の範囲 [rad]
GLOBAL_SCALE_RANGE = [0.95, 1.05]        # シーン全体のスケールを変更する倍率

# 再配置のリトライ回数上限
MAX_POSITION_RETRIES = 10
# ==========================================


def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

ensure_dir(OUTPUT_LIDAR_DIR)
ensure_dir(OUTPUT_LABEL_TAO_DIR)
ensure_dir(OUTPUT_LABEL_VIS_DIR)

# --- 共通関数 ---

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

# --- ★修正された間引き関数 ---
def simulate_sparsity(points, orig_dist, new_dist, min_points):
    """
    点数が閾値を下回る場合は None を返す
    """
    if new_dist <= orig_dist:
        return points 
    
    keep_ratio = (orig_dist / new_dist) / 2
    num_points = len(points)
    num_keep = int(num_points * keep_ratio)
    
    # ★ ここでチェック: 点が少なすぎるなら失敗とする
    if num_keep < min_points:
        return None 
    
    indices = np.random.choice(num_points, num_keep, replace=False)
    return points[indices]

# --- 変換ロジック ---

def apply_local_transform(points_local, box_params):
    pts = points_local.copy()
    box = box_params.copy()
    scale = np.random.uniform(LOCAL_SCALE_RANGE[0], LOCAL_SCALE_RANGE[1])
    pts[:, :3] *= scale
    box['l'] *= scale; box['w'] *= scale; box['h'] *= scale
    rot_angle = np.random.uniform(LOCAL_ROT_RANGE[0], LOCAL_ROT_RANGE[1])
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    x_new = pts[:, 0] * c - pts[:, 1] * s
    y_new = pts[:, 0] * s + pts[:, 1] * c
    pts[:, 0], pts[:, 1] = x_new, y_new
    box['yaw'] += rot_angle
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
    files = sorted([os.path.splitext(f)[0] for f in os.listdir(INPUT_JSON_DIR) if f.endswith('.json')])
    print(f"Processing {len(files)} files. Output serial number...")

    # 1. DB構築
    print("Building Database...")
    person_db = []
    for file_id in tqdm(files):
        pcd_path = os.path.join(INPUT_PCD_DIR, file_id + '.pcd')
        json_path = os.path.join(INPUT_JSON_DIR, file_id + '.json')
        if not os.path.exists(pcd_path): continue
        points = read_pcd(pcd_path)
        with open(json_path, 'r') as f: data = json.load(f)
        for obj in data['labels']:
            if obj['category'] != 'person': continue
            box = {
                'l': obj['box3d']['dimension']['length'],
                'w': obj['box3d']['dimension']['width'],
                'h': obj['box3d']['dimension']['height'],
                'x': obj['box3d']['location']['x'],
                'y': obj['box3d']['location']['y'],
                'z': obj['box3d']['location']['z'],
                'yaw': obj['box3d']['orientation']['rotationYaw'],
                'cls': 'Pedestrian'
            }
            obj_points = crop_object_points(points, box)
            if len(obj_points) >= MIN_POINTS_THRESHOLD:
                dist = np.sqrt(box['x']**2 + box['y']**2)
                person_db.append({'points': obj_points, 'box': box, 'orig_dist': dist})
    
    if not person_db:
        print("No valid persons found in DB.")
        return

    # 2. 拡張ループ
    print("Generating Augmented Dataset...")
    global_idx = 0

    for file_id in tqdm(files):
        pcd_path = os.path.join(INPUT_PCD_DIR, file_id + '.pcd')
        json_path = os.path.join(INPUT_JSON_DIR, file_id + '.json')
        if not os.path.exists(pcd_path): continue
        
        orig_points = read_pcd(pcd_path)
        with open(json_path, 'r') as f: orig_json = json.load(f)
        
        orig_boxes = []
        for obj in orig_json['labels']:
            if obj['category'] == 'person':
                orig_boxes.append({
                    'l': obj['box3d']['dimension']['length'],
                    'w': obj['box3d']['dimension']['width'],
                    'h': obj['box3d']['dimension']['height'],
                    'x': obj['box3d']['location']['x'],
                    'y': obj['box3d']['location']['y'],
                    'z': obj['box3d']['location']['z'],
                    'yaw': obj['box3d']['orientation']['rotationYaw'],
                    'cls': 'Pedestrian'
                })

        # 背景分離
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
            
            # A. 既存の人
            for p_data in orig_person_data:
                apt, abox = apply_local_transform(p_data['points'], p_data['box'])
                added_people_points_list.append(transform_points_to_world(apt, abox))
                added_people_boxes.append(abox)
            
            # B. GT-Sampling (リトライ機能付き)
            collision_boxes = copy.deepcopy(added_people_boxes)
            num_paste = np.random.randint(1, MAX_PASTE_PERSONS + 1)
            
            for _ in range(num_paste):
                sample = person_db[np.random.randint(len(person_db))]
                
                # ★リトライループ: 有効な位置が見つかるまで試行する
                success_placement = False
                for retry in range(MAX_POSITION_RETRIES):
                    # 1. 座標決定
                    new_x = np.random.uniform(PASTE_AREA_X[0], PASTE_AREA_X[1])
                    new_y = np.random.uniform(PASTE_AREA_Y[0], PASTE_AREA_Y[1])
                    new_yaw = np.random.uniform(-np.pi, np.pi)
                    
                    new_dist = np.sqrt(new_x**2 + new_y**2)
                    
                    # 2. 疎密チェック (Noneが返ってきたら点が少なすぎる -> やり直し)
                    sparse_points = simulate_sparsity(sample['points'], sample['orig_dist'], new_dist, MIN_POINTS_THRESHOLD)
                    
                    if sparse_points is None:
                        continue # 点が少なすぎるので、別の場所(恐らくもっと近い場所)を抽選しなおす
                    
                    # 3. 衝突チェック
                    temp_box = sample['box'].copy()
                    temp_box['x'], temp_box['y'], temp_box['yaw'] = new_x, new_y, new_yaw
                    apt, abox = apply_local_transform(sparse_points, temp_box)
                    
                    if not check_collision(abox, collision_boxes):
                        # 成功！
                        added_people_points_list.append(transform_points_to_world(apt, abox))
                        added_people_boxes.append(abox)
                        collision_boxes.append(abox)
                        success_placement = True
                        break # リトライループを抜ける
                
                # リトライしてもダメだった場合は、この人の配置をスキップして次へ

            # 背景削除
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

            # C. グローバル変換
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

            # 保存 (連番)
            save_name = f"{global_idx:06d}"
            
            scene_points.astype(np.float32).tofile(os.path.join(OUTPUT_LIDAR_DIR, save_name + '.bin'))
            
            # TAO Label
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

            # VIS Label
            label_lines_vis = []
            for b in added_people_boxes:
                z_bottom = b['z'] - (b['h'] / 2.0)
                line = f"{b['cls']} 0 0 0 0 0 0 0 {b['h']:.3f} {b['w']:.3f} {b['l']:.3f} {b['x']:.3f} {b['y']:.3f} {z_bottom:.3f} {b['yaw']:.3f}"
                label_lines_vis.append(line)
            with open(os.path.join(OUTPUT_LABEL_VIS_DIR, save_name + '.txt'), 'w') as f: f.write('\n'.join(label_lines_vis))
            
            global_idx += 1

    print(f"Done. Files saved from 000000.bin to {global_idx-1:06d}.bin in '{OUTPUT_ROOT}'")

if __name__ == "__main__":
    main()
