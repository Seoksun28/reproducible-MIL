import os
import argparse
import openslide
import h5py
import numpy as np
import cv2
from PIL import Image

# =============================================================================
# [Configuration] 연구 목적에 맞춘 하이퍼파라미터
# =============================================================================
CONF = {
    'PATCH_SIZE': 256,       # 학습 시 Patch Size (Target Magnification 기준)
    'VIS_LEVEL': -1,         # -1: 자동 (메모리 효율 고려하여 적절한 해상도 선택)
    'ALPHA': 0.4,            # 히트맵 투명도 (0.4 ~ 0.5 추천)
    
    # [중요] Diffuse 질환 시각화 핵심 설정
    'BLUR_FACTOR': 4.0,      # 높을수록 구름처럼 몽글몽글해짐 (기본 2.0 -> 4.0 상향)
    'CLIP_PCT': (2, 98),     # Logit의 이상치(Outlier) 제거 범위 (하위 2%, 상위 98%)
    'BG_THRESH': 0.1,        # 정규화된 점수가 이 값 미만이면 투명하게 날림 (노이즈 제거)
    
    # [중요] 논문 방어용 텍스트
    'TEXT_INFO': True,       # 이미지 구석에 Raw Logit 통계(μ, σ) 표기 여부
}

# =============================================================================
# [Helper Functions]
# =============================================================================
def get_file_map(folder, exts):
    """폴더 내 파일 재귀 탐색 및 매핑 (Stem -> Path)"""
    fmap = {}
    if exts: exts = tuple(e.lower() for e in exts)
    for root, _, files in os.walk(folder):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            if exts and ext.lower() not in exts:
                continue
            if stem.endswith('_patches'):
                stem = stem[:-8]  # h5 이름 정규화
            fmap[stem] = os.path.join(root, fname)
    return fmap

def infer_patch_size(coords, default_size):
    """좌표 간격으로 실제 Patch Size 추론"""
    if len(coords) < 2:
        return default_size
    xs = np.sort(np.unique(coords[:, 0]))
    ys = np.sort(np.unique(coords[:, 1]))
    dx = np.min(np.diff(xs)) if len(xs) > 1 else default_size
    dy = np.min(np.diff(ys)) if len(ys) > 1 else default_size
    inferred = float(max(dx, dy))
    if inferred < default_size * 0.5 or inferred > default_size * 2.0:
        return default_size
    return inferred

def normalize_logits(logits):
    """
    [핵심 알고리즘] Robust Min-Max Scaling for Logits
    - Softmax를 쓰지 않은 Logit은 범위가 제멋대로임 (-inf ~ inf)
    - 그냥 Min-Max를 하면 튀는 값 하나 때문에 전체가 흐려짐
    - 따라서 Percentile Clipping 후 Min-Max를 적용함
    """
    logits = np.asarray(logits, dtype=np.float32)

    # 통계값은 클리핑 이전의 raw logit 기준으로 계산
    raw_max = float(np.max(logits))
    raw_min = float(np.min(logits))
    raw_mean = float(np.mean(logits))
    raw_std = float(np.std(logits))

    # 1. 이상치(Outlier) Clipping
    lower = np.percentile(logits, CONF['CLIP_PCT'][0])
    upper = np.percentile(logits, CONF['CLIP_PCT'][1])
    clipped = np.clip(logits, lower, upper)
    
    # 2. Min-Max Scaling
    denom = upper - lower
    if denom < 1e-9:
        norm = np.zeros_like(logits)
    else:
        norm = (clipped - lower) / denom
        
    # 3. Background Noise Cutoff
    norm[norm < CONF['BG_THRESH']] = 0.0
    
    return norm, raw_max, raw_min, raw_mean, raw_std

# =============================================================================
# [Core Visualization Logic]
# =============================================================================
def draw_heatmap(slide_path, coords_path, scores_path, out_path):
    print(f"\nProcessing: {os.path.basename(slide_path)}")
    
    # 1. Load Data
    try:
        slide = openslide.OpenSlide(slide_path)
        with h5py.File(coords_path, 'r') as f:
            coords = f['coords'][:]
        scores = np.load(scores_path)
        if scores.ndim > 1:
            scores = scores.flatten()
    except Exception as e:
        print(f"  [Error] Load failed: {e}")
        return

    # 2. Normalize (Logit Optimized)
    norm_scores, raw_max, raw_min, raw_mean, raw_std = normalize_logits(scores)
    
    # 3. Determine Visualization Level
    vis_level = CONF['VIS_LEVEL']
    if vis_level < 0:
        vis_level = slide.level_count - 1
        for i in range(slide.level_count):
            if slide.level_dimensions[i][0] < 5000:  # 적절한 해상도 제한
                vis_level = i
                break
    
    w_vis, h_vis = slide.level_dimensions[vis_level]
    ds = slide.level_downsamples[vis_level]
    
    # 4. Patch Size Calculation
    patch_size_l0 = infer_patch_size(coords, CONF['PATCH_SIZE'])
    scaled_patch_size = max(1, int(patch_size_l0 / ds))
    
    print(
        f"  - Level: {vis_level} ({w_vis}x{h_vis}), Patch: {scaled_patch_size}px\n"
        f"  - Logit Range: {raw_min:.4f} ~ {raw_max:.4f}, "
        f"mean={raw_mean:.4f}, std={raw_std:.4f}"
    )

    # 5. Create Masks
    heatmap = np.zeros((h_vis, w_vis), dtype=np.float32)
    count_map = np.zeros((h_vis, w_vis), dtype=np.float32)
    coords_vis = (coords / ds).astype(np.int32)
    
    # 6. Fill Patches
    for (x, y), score in zip(coords_vis, norm_scores):
        if x >= w_vis or y >= h_vis:
            continue
        xe, ye = min(x + scaled_patch_size, w_vis), min(y + scaled_patch_size, h_vis)
        
        heatmap[y:ye, x:xe] += score
        count_map[y:ye, x:xe] += 1

    # 7. Average Overlaps
    mask = count_map > 0
    heatmap[mask] /= count_map[mask]
    
    # 8. Apply Strong Blur (Cloud Effect for Diffuse Lesion)
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    k_size = int(scaled_patch_size * CONF['BLUR_FACTOR']) | 1 
    sigma = k_size // 3  # Sigma를 크게 주어 부드럽게 퍼지게 함
    heatmap_u8 = cv2.GaussianBlur(heatmap_u8, (k_size, k_size), sigma)
    
    # 9. Apply ColorMap (Jet)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    
    # 10. Overlay on WSI
    # RGBA로 읽어서 흰 배경 처리
    region = slide.read_region((0, 0), vis_level, (w_vis, h_vis))
    if region.mode == 'RGBA':
        bg = Image.new("RGBA", region.size, (255, 255, 255, 255))
        region = Image.alpha_composite(bg, region)
    original = cv2.cvtColor(np.array(region.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    # 조직이 있는 부분(mask)에만 히트맵 합성
    overlay = original.copy()
    
    # 여기서는 count_map이 0보다 큰 곳(조직)만 정확히 칠함
    tissue_mask_vis = count_map > 0
    
    blended = cv2.addWeighted(original, 1 - CONF['ALPHA'], heatmap_color, CONF['ALPHA'], 0)
    overlay[tissue_mask_vis] = blended[tissue_mask_vis]

    # 11. Add Raw Logit Stats Text (논문 방어용)
    if CONF['TEXT_INFO']:
        # 예: "Logit μ=0.85, σ=0.32"
        text = f"Logit μ={raw_mean:.2f}, σ={raw_std:.2f}"
        # mean 기준으로 색 구분 (control/low: 초록, case/high: 빨강 느낌)
        color = (0, 180, 0) if raw_mean < 0.0 else (0, 0, 255)
        
        cv2.putText(
            overlay, text, (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (255, 255, 255), 5, cv2.LINE_AA  # 흰 테두리
        )
        cv2.putText(
            overlay, text, (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            color, 2, cv2.LINE_AA            # 실제 글자색
        )

    # 12. Save
    cv2.imwrite(out_path, overlay)
    print(f"  - Saved: {out_path}")

# =============================================================================
# [Main Executor]
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diffuse Disease Heatmap Generator (Logit Optimized)")
    parser.add_argument('--wsi_case', type=str, required=True, help="Case WSI Folder")
    parser.add_argument('--wsi_control', type=str, required=True, help="Control WSI Folder")
    parser.add_argument('--h5_case', type=str, required=True, help="Case Coords Folder")
    parser.add_argument('--h5_control', type=str, required=True, help="Control Coords Folder")
    parser.add_argument('--npy_dir', type=str, required=True, help="Attention Scores (.npy) Folder")
    parser.add_argument('--out_dir', type=str, required=True, help="Output Folder")
    args = parser.parse_args()

    # 1. Map NPY Files (Common)
    npy_map = get_file_map(args.npy_dir, ['.npy'])
    print(f"[Init] Found {len(npy_map)} attention score files.")

    # 2. Process Groups
    groups = [
        ('case', args.wsi_case, args.h5_case),
        ('control', args.wsi_control, args.h5_control)
    ]

    for grp_name, wsi_d, h5_d in groups:
        wsi_map = get_file_map(wsi_d, ['.svs', '.tif', '.ndpi', '.mrxs', '.bif'])
        h5_map = get_file_map(h5_d, ['.h5'])
        
        # 교집합 파일만 처리
        common_stems = sorted(set(wsi_map.keys()) & set(h5_map.keys()) & set(npy_map.keys()))
        print(f"\n[Group: {grp_name.upper()}] Matched {len(common_stems)} slides.")
        
        out_grp = os.path.join(args.out_dir, grp_name)
        os.makedirs(out_grp, exist_ok=True)
        
        for stem in common_stems:
            try:
                draw_heatmap(
                    slide_path=wsi_map[stem],
                    coords_path=h5_map[stem],
                    scores_path=npy_map[stem],
                    out_path=os.path.join(out_grp, f"{stem}.jpg")
                )
            except Exception as e:
                print(f"[Error] Failed on {stem}: {e}")
