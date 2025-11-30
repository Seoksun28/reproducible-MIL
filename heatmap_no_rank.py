import os
import argparse
import openslide
import h5py
import numpy as np
import cv2
from scipy.stats import rankdata
from PIL import Image

# =============================================================================
# [Configuration] 미만성(Diffuse) 질환 & 논문용 최적화 설정
# =============================================================================
CONF = {
    'PATCH_SIZE': 256,       # 학습 시 Patch Size (256 or 512)
    'VIS_LEVEL': -1,         # -1: 자동 (메모리 아끼면서 적절한 해상도 찾음)
    'ALPHA': 0.4,            # 히트맵 투명도 (0.4 ~ 0.5 추천)
    'USE_RANK': False,       # [중요] False로 설정 (Diffuse 병변은 Rank 쓰면 안됨)
    'BLUR_FACTOR': 4.0,      # [중요] 흐림 강도 (기본 2.0 -> 4.0으로 UP for Diffuse look)
    'TEXT_INFO': True,       # 이미지 구석에 Max Score 적을지 여부 (Control 방어용)
    'SCORE_THRES': 0.0,      # (선택) 이 점수 이하는 아예 투명하게 날림 (노이즈 제거용, 예: 0.1)
}

# =============================================================================
# [Utils]
# =============================================================================
def get_file_map(folder, exts):
    """폴더 내 파일 매핑 (stem -> path)"""
    fmap = {}
    if exts: exts = tuple(e.lower() for e in exts)
    for root, _, files in os.walk(folder):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            if exts and ext.lower() not in exts: continue
            
            # h5 파일명 정규화 (_patches 제거)
            if stem.endswith('_patches'): stem = stem[:-8]
            fmap[stem] = os.path.join(root, fname)
    return fmap

def infer_patch_size(coords, default_size):
    """좌표 간격으로 실제 Patch Size 추론"""
    if len(coords) < 2: return default_size
    xs = np.sort(np.unique(coords[:, 0]))
    ys = np.sort(np.unique(coords[:, 1]))
    dx = np.min(np.diff(xs)) if len(xs) > 1 else default_size
    dy = np.min(np.diff(ys)) if len(ys) > 1 else default_size
    inferred = float(max(dx, dy))
    
    # 터무니없는 값이면 default 사용
    if inferred < default_size * 0.5 or inferred > default_size * 2.0:
        return default_size
    return inferred

def normalize_scores(scores):
    """
    [핵심 로직]
    Diffuse 병변을 위해 Min-Max Scaling 사용.
    Control 군의 낮은 점수를 그대로 반영하기 위해 Rank는 끔.
    """
    raw_min, raw_max = scores.min(), scores.max()
    
    if CONF['USE_RANK']:
        # 기존 방식 (Focal 병변용)
        norm_scores = rankdata(scores) / len(scores)
    else:
        # 개선 방식 (Diffuse 병변용)
        # 점수 차이가 거의 없으면(0으로 나눔 방지) 그냥 0 처리
        if (raw_max - raw_min) < 1e-9:
            norm_scores = np.zeros_like(scores)
        else:
            norm_scores = (scores - raw_min) / (raw_max - raw_min)
            
        # (선택) 노이즈 제거: 너무 낮은 점수는 아예 0으로
        if CONF['SCORE_THRES'] > 0:
            norm_scores[norm_scores < CONF['SCORE_THRES']] = 0

    return norm_scores, raw_max

# =============================================================================
# [Core Visualization]
# =============================================================================
def draw_heatmap(slide_path, coords_path, scores_path, out_path):
    print(f"\nProcessing: {os.path.basename(slide_path)}")
    
    # 1. Load Data
    try:
        slide = openslide.OpenSlide(slide_path)
        with h5py.File(coords_path, 'r') as f:
            coords = f['coords'][:]
        scores = np.load(scores_path)
        if scores.ndim > 1: scores = scores.flatten()
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return

    # 2. Normalize Scores (Min-Max)
    norm_scores, raw_max_score = normalize_scores(scores)
    
    # 3. Determine Vis Level
    vis_level = CONF['VIS_LEVEL']
    if vis_level < 0:
        vis_level = slide.level_count - 1
        for i in range(slide.level_count):
            # 너무 작지 않은 적당한 해상도(width < 5000) 선택
            if slide.level_dimensions[i][0] < 5000:
                vis_level = i
                break
    
    w_vis, h_vis = slide.level_dimensions[vis_level]
    ds = slide.level_downsamples[vis_level]
    print(f"  - Level: {vis_level} ({w_vis}x{h_vis}), Max Score: {raw_max_score:.4f}")

    # 4. Infer Patch Size & Scaling
    patch_size_l0 = infer_patch_size(coords, CONF['PATCH_SIZE'])
    scaled_patch_size = max(1, int(patch_size_l0 / ds))
    
    # 5. Draw Masks
    heatmap = np.zeros((h_vis, w_vis), dtype=np.float32)
    count_map = np.zeros((h_vis, w_vis), dtype=np.float32)
    
    coords_vis = (coords / ds).astype(np.int32)
    
    # Vectorized operation is hard for overlay, using loop (safe & clear)
    for (x, y), score in zip(coords_vis, norm_scores):
        if x >= w_vis or y >= h_vis: continue
        xe, ye = min(x + scaled_patch_size, w_vis), min(y + scaled_patch_size, h_vis)
        heatmap[y:ye, x:xe] += score
        count_map[y:ye, x:xe] += 1

    # Average overlapping patches
    mask = count_map > 0
    heatmap[mask] /= count_map[mask]
    
    # 6. Apply Blur (Diffuse Style)
    # 팩터를 4.0으로 키워서 '구름'처럼 뭉개버림
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    k_size = int(scaled_patch_size * CONF['BLUR_FACTOR']) | 1 
    sigma = k_size // 3
    heatmap_u8 = cv2.GaussianBlur(heatmap_u8, (k_size, k_size), sigma)
    
    # 7. Apply ColorMap (Jet is standard)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    
    # 8. Overlay
    region = slide.read_region((0,0), vis_level, (w_vis, h_vis)).convert("RGB")
    original = np.array(region)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR) # PIL(RGB) -> cv2(BGR)
    
    # Tissue Mask가 있는 곳만 색칠 (배경 흰색 유지)
    # count_map이 있는 곳이 곧 tissue (patch가 추출된 곳)
    tissue_mask_vis = count_map > 0
    
    overlay = original.copy()
    blended = cv2.addWeighted(original, 1 - CONF['ALPHA'], heatmap_color, CONF['ALPHA'], 0)
    overlay[tissue_mask_vis] = blended[tissue_mask_vis]

    # 9. [Optional] Add Score Text (논문 방어용)
    if CONF['TEXT_INFO']:
        text = f"Max Attention: {raw_max_score:.4f}"
        # 점수가 낮으면(Control) 초록색 글씨, 높으면 빨간 글씨
        txt_color = (0, 200, 0) if raw_max_score < 0.1 else (0, 0, 255)
        cv2.putText(overlay, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, txt_color, 3, cv2.LINE_AA)

    # 10. Save
    cv2.imwrite(out_path, overlay)
    print(f"  - Saved: {out_path}")

# =============================================================================
# [Main Executor]
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_case', type=str, required=True)
    parser.add_argument('--wsi_control', type=str, required=True)
    parser.add_argument('--h5_case', type=str, required=True)
    parser.add_argument('--h5_control', type=str, required=True)
    parser.add_argument('--npy_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    # 1. Map Files
    npy_map = get_file_map(args.npy_dir, ['.npy'])
    
    groups = [
        ('case', args.wsi_case, args.h5_case),
        ('control', args.wsi_control, args.h5_control)
    ]

    for grp_name, wsi_d, h5_d in groups:
        wsi_map = get_file_map(wsi_d, ['.svs', '.tif', '.ndpi', '.mrxs', '.bif'])
        h5_map = get_file_map(h5_d, ['.h5'])
        
        # 교집합 찾기
        common = sorted(set(wsi_map.keys()) & set(h5_map.keys()) & set(npy_map.keys()))
        print(f"\n[Group: {grp_name}] Found {len(common)} slides.")
        
        out_grp = os.path.join(args.out_dir, grp_name)
        os.makedirs(out_grp, exist_ok=True)
        
        for stem in common:
            draw_heatmap(wsi_map[stem], h5_map[stem], npy_map[stem], 
                         os.path.join(out_grp, f"{stem}.jpg"))
