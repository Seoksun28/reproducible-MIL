import os
import argparse
import openslide
import h5py
import numpy as np
import cv2
from PIL import Image


# =============================================================================
# CONFIG – Tone-mapping 기반 파라미터
# =============================================================================
CONF = {
    'PATCH_SIZE': 256,
    'VIS_LEVEL': -1,
    'ALPHA': 0.40,        # Overlay 투명도
    'BLUR_FACTOR': 4.0,   # Blur 크기
    'MEAN_INTENSITY': 0.30,   # 평균(z=0)이 찍히는 intensity 값
    'Z_CLIP': (-2, 4),        # z-score 클리핑 범위
    'BG_THRESH': 0.10,        # 평균 이하 일부 제거
    'TEXT_INFO': True
}


# =============================================================================
# Tone-Mapped Normalization (핵심)
# =============================================================================
def normalize_attention_tonemapped(scores):
    """
    Softmax attention → Z-score Normalization → Tone-Mapping(mean=0.3)
    논문 문장의 의도:
      - "정규분포 normalize"
      - "표준편차 단위 설명력 확보"
      - "평균점을 heatmap intensity 0.3 근처에 배치"
    를 모두 반영한 함수.
    """

    scores = scores.astype(np.float32)
    raw_mean = float(scores.mean())
    raw_std = float(scores.std() + 1e-8)

    # 1) Z-score
    z = (scores - raw_mean) / raw_std

    # 2) Clip z-range (diffuse disease 안정화)
    z_min, z_max = CONF['Z_CLIP']
    z = np.clip(z, z_min, z_max)   # [-2, 4] 기본

    # 3) Normalize to [0,1] first
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)

    # 4) Tone-mapping:
    #    mean(z)=0 → intensity ≈ 0.3
    intensity = CONF['MEAN_INTENSITY'] + z_norm * (1.0 - CONF['MEAN_INTENSITY'])

    # 5) Background threshold (noise 제거)
    intensity[intensity < CONF['BG_THRESH']] = 0.0

    return intensity, raw_mean, raw_std



# =============================================================================
# Utilities
# =============================================================================
def get_file_map(folder, exts):
    fmap = {}
    if exts: exts = tuple(e.lower() for e in exts)
    for root, _, files in os.walk(folder):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            if exts and ext.lower() not in exts:
                continue
            if stem.endswith('_patches'):
                stem = stem[:-8]
            fmap[stem] = os.path.join(root, fname)
    return fmap


def infer_patch_size(coords, default_size):
    if len(coords) < 2:
        return default_size
    xs = np.sort(np.unique(coords[:, 0]))
    ys = np.sort(np.unique(coords[:, 1]))
    dx = np.min(np.diff(xs)) if len(xs) > 1 else default_size
    dy = np.min(np.diff(ys)) if len(ys) > 1 else default_size
    inferred = float(max(dx, dy))
    if inferred < default_size * 0.5 or inferred > default_size * 2:
        return default_size
    return inferred



# =============================================================================
# Heatmap Draw Core
# =============================================================================
def draw_heatmap(slide_path, coords_path, scores_path, out_path):
    print(f"\n[Processing] {os.path.basename(slide_path)}")

    # 1) Load slide, coords, attention scores
    slide = openslide.OpenSlide(slide_path)
    with h5py.File(coords_path, "r") as f:
        coords = f["coords"][:]

    scores = np.load(scores_path).astype(np.float32)
    if scores.ndim > 1:
        scores = scores.flatten()

    # 2) Tone-mapped normalization
    norm_scores, raw_mean, raw_std = normalize_attention_tonemapped(scores)

    # 3) Choose Visualization Level
    vis_level = CONF['VIS_LEVEL']
    if vis_level < 0:
        vis_level = slide.level_count - 1
        for i in range(slide.level_count):
            if slide.level_dimensions[i][0] < 5000:
                vis_level = i
                break

    w_vis, h_vis = slide.level_dimensions[vis_level]
    ds = slide.level_downsamples[vis_level]

    patch_size_l0 = infer_patch_size(coords, CONF['PATCH_SIZE'])
    scaled_patch_size = max(1, int(patch_size_l0 / ds))

    coords_vis = (coords / ds).astype(np.int32)

    # 4) Heatmap Memory
    heatmap = np.zeros((h_vis, w_vis), dtype=np.float32)
    count_map = np.zeros((h_vis, w_vis), dtype=np.float32)

    # 5) Fill
    for (x, y), score in zip(coords_vis, norm_scores):
        if x >= w_vis or y >= h_vis:
            continue
        xe, ye = min(x + scaled_patch_size, w_vis), min(y + scaled_patch_size, h_vis)
        heatmap[y:ye, x:xe] += score
        count_map[y:ye, x:xe] += 1

    mask = count_map > 0
    heatmap[mask] /= count_map[mask]

    # 6) Blur for diffusion
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    k = int(scaled_patch_size * CONF['BLUR_FACTOR']) | 1
    sigma = k // 3
    heatmap_u8 = cv2.GaussianBlur(heatmap_u8, (k, k), sigma)

    # 7) Colormap
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    # 8) Overlay
    region = slide.read_region((0, 0), vis_level, (w_vis, h_vis))
    if region.mode == "RGBA":
        bg = Image.new("RGBA", region.size, (255,255,255,255))
        region = Image.alpha_composite(bg, region)

    base = cv2.cvtColor(np.array(region.convert("RGB")), cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(base, 1 - CONF['ALPHA'], heatmap_color, CONF['ALPHA'], 0)

    overlay = base.copy()
    overlay[mask] = blended[mask]

    # 9) Text Info (논문용)
    if CONF['TEXT_INFO']:
        text = f"μ={raw_mean:.3f}, σ={raw_std:.3f}"
        cv2.putText(overlay, text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255,255,255), 4, cv2.LINE_AA)
        cv2.putText(overlay, text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0,0,255), 2, cv2.LINE_AA)

    # 10) Save
    cv2.imwrite(out_path, overlay)
    print(f"  - Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_case', type=str, required=True)
    parser.add_argument('--wsi_control', type=str, required=True)
    parser.add_argument('--h5_case', type=str, required=True)
    parser.add_argument('--h5_control', type=str, required=True)
    parser.add_argument('--npy_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    npy_map = get_file_map(args.npy_dir, ['.npy'])

    for grp_name, wsi_dir, h5_dir in [
        ('case', args.wsi_case, args.h5_case),
        ('control', args.wsi_control, args.h5_control)
    ]:
        wsi_map = get_file_map(wsi_dir, ['.svs', '.tif', '.ndpi', '.mrxs', '.bif'])
        h5_map = get_file_map(h5_dir, ['.h5'])

        common = sorted(set(wsi_map.keys()) & set(h5_map.keys()) & set(npy_map.keys()))
        print(f"\n[Group: {grp_name.upper()}] {len(common)} slides matched.")

        out_grp = os.path.join(args.out_dir, grp_name)
        os.makedirs(out_grp, exist_ok=True)

        for stem in common:
            draw_heatmap(
                slide_path=wsi_map[stem],
                coords_path=h5_map[stem],
                scores_path=npy_map[stem],
                out_path=os.path.join(out_grp, f"{stem}.jpg")
            )
