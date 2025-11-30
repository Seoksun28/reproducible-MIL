import os
import openslide
import h5py
import numpy as np
import cv2
from scipy.stats import rankdata


# ==========================
# --- 공통 설정 ---
# ==========================
PATCH_SIZE = 256
VIS_LEVEL = 2
ALPHA = 0.4
BLUR = True
USE_PERCENTILES = True


# ==========================
# --- 유틸 ---
# ==========================
def infer_patch_size_from_coords(coords_l0: np.ndarray,
                                 default_patch_size_l0: float) -> float:
    xs = np.sort(np.unique(coords_l0[:, 0]))
    ys = np.sort(np.unique(coords_l0[:, 1]))

    if len(xs) > 1:
        dx = np.min(np.diff(xs))
    else:
        dx = default_patch_size_l0

    if len(ys) > 1:
        dy = np.min(np.diff(ys))
    else:
        dy = default_patch_size_l0

    inferred = float(max(dx, dy))
    if inferred < default_patch_size_l0 * 0.5 or inferred > default_patch_size_l0 * 2.0:
        print(f"[WARN] inferred {inferred}, fallback to {default_patch_size_l0}")
        return default_patch_size_l0

    print(f"[DEBUG] inferred patch_size_l0={inferred}")
    return inferred


def get_stem_to_file_map(folder: str, exts=None):
    """재귀적 파일 탐색"""
    stem_map = {}
    if exts is not None:
        exts = tuple(e.lower() for e in exts)

    for root, _, files in os.walk(folder):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            ext = ext.lower()

            if exts is not None and ext not in exts:
                continue

            fpath = os.path.join(root, fname)
            stem_map[stem] = fpath

    return stem_map


# ==========================
# --- 핵심: CLAM Heatmap ---
# ==========================
def get_clam_style_heatmap(slide_path: str,
                           coords_path: str,
                           scores_path: str,
                           output_path: str):

    print(f"\n=== Processing: {os.path.basename(slide_path)} ===")
    slide = openslide.OpenSlide(slide_path)

    with h5py.File(coords_path, 'r') as f:
        coords = f['coords'][:]

    scores = np.load(scores_path).astype(np.float32)
    scores = scores.flatten()

    assert len(coords) == len(scores), f"coords({len(coords)}) != scores({len(scores)})"

    # 1) Score Normalize
    if USE_PERCENTILES:
        scores = rankdata(scores) / len(scores)
    else:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # 2) Visualization level 선택
    if VIS_LEVEL < 0:
        best = slide.level_count - 1
        for i in range(slide.level_count):
            if slide.level_dimensions[i][0] < 5000:
                best = i
                break
        vis_level = best
    else:
        vis_level = VIS_LEVEL

    w_vis, h_vis = slide.level_dimensions[vis_level]
    ds = float(slide.level_downsamples[vis_level])

    coords_l0 = coords.astype(np.float32)

    # 3) patch size 추정
    patch_size_l0 = infer_patch_size_from_coords(coords_l0, PATCH_SIZE)
    scaled_patch_size = max(1, int(round(patch_size_l0 / ds)))

    # 4) 빈 mask 생성
    heatmap = np.zeros((h_vis, w_vis), dtype=np.float32)
    count_mask = np.zeros((h_vis, w_vis), dtype=np.float32)

    # 5) pixel 채우기
    for (x0, y0), score in zip(coords_l0, scores):
        x = int(x0 / ds)
        y = int(y0 / ds)

        if x >= w_vis or y >= h_vis:
            continue

        x2 = min(x + scaled_patch_size, w_vis)
        y2 = min(y + scaled_patch_size, h_vis)

        heatmap[y:y2, x:x2] += float(score)
        count_mask[y:y2, x:x2] += 1.0

    # 6) 평균화
    nonzero = count_mask > 0
    heatmap[nonzero] /= count_mask[nonzero]

    # 7) 0~255 scaling
    heatmap = (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)

    # 8) Gaussian blur
    if BLUR:
        k = max(3, (scaled_patch_size * 2) | 1)
        heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 9) WSI load
    original = slide.read_region((0, 0), vis_level, (w_vis, h_vis)).convert("RGB")
    original = np.array(original)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    tissue_mask = (count_mask > 0).astype(np.uint8)
    overlay = cv2.addWeighted(original, 1 - ALPHA, heatmap_color, ALPHA, 0)

    idx = np.where(tissue_mask == 0)
    overlay[idx] = original[idx]

    cv2.imwrite(output_path, overlay)
    print(f"Saved: {output_path}")


# ==========================
# --- Case / Control 처리 ---
# ==========================
def generate_group_heatmaps(wsi_dir, h5_dir, npy_dir, out_dir_group):
    os.makedirs(out_dir_group, exist_ok=True)

    wsi_exts = ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.svslide', '.bif']

    print(f"\n[Group: {os.path.basename(out_dir_group)}] Mapping files...")

    wsi_map = get_stem_to_file_map(wsi_dir, wsi_exts)
    h5_map = get_stem_to_file_map(h5_dir, ['.h5', '.hdf5'])
    npy_map = get_stem_to_file_map(npy_dir, ['.npy'])

    common = sorted(set(wsi_map.keys()) & set(h5_map.keys()) & set(npy_map.keys()))
    print(f"  공통 stem 수: {len(common)}")

    for i, stem in enumerate(common, 1):
        print(f" [{i}/{len(common)}] {stem}")

        out_path = os.path.join(out_dir_group, f"{stem}_heatmap.jpg")

        get_clam_style_heatmap(
            slide_path=wsi_map[stem],
            coords_path=h5_map[stem],
            scores_path=npy_map[stem],
            output_path=out_path
        )


# ==========================
# --- Main ---
# ==========================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CLAM Heatmap Generator (Case/Control 분리)")

    parser.add_argument('--wsi_case_dir', type=str, required=True)
    parser.add_argument('--wsi_control_dir', type=str, required=True)

    parser.add_argument('--h5_case_dir', type=str, required=True)
    parser.add_argument('--h5_control_dir', type=str, required=True)

    parser.add_argument('--npy_case_dir', type=str, required=True)
    parser.add_argument('--npy_control_dir', type=str, required=True)

    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()

    # CASE
    generate_group_heatmaps(
        wsi_dir=args.wsi_case_dir,
        h5_dir=args.h5_case_dir,
        npy_dir=args.npy_case_dir,
        out_dir_group=os.path.join(args.out_dir, "case")
    )

    # CONTROL
    generate_group_heatmaps(
        wsi_dir=args.wsi_control_dir,
        h5_dir=args.h5_control_dir,
        npy_dir=args.npy_control_dir,
        out_dir_group=os.path.join(args.out_dir, "control")
    )
