import os
import openslide
import h5py
import numpy as np
import cv2
from scipy.stats import rankdata

# ==========================
# --- 공통 설정 ---
# ==========================
PATCH_SIZE = 256       # 모델 학습에 쓴 patch 크기 (target mag 기준)
VIS_LEVEL = 2          # 0=원본, 1/2=downsample, -1이면 자동 선택
ALPHA = 0.4            # 히트맵 투명도
BLUR = True            # CLAM 스타일 Gaussian blur
USE_PERCENTILES = True # score를 percentile(0~1)로 변환할지 여부


# ==========================
# --- 유틸 함수 ---
# ==========================
def infer_patch_size_from_coords(coords_l0: np.ndarray,
                                 default_patch_size_l0: float) -> float:
    """
    coords_l0: level-0 기준 좌표 (N, 2)
    default_patch_size_l0: 기본 patch size (예: 256)

    coords 간 최소 간격(dx, dy)을 이용해 실제 patch 한 변 길이를 추정.
    """
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

    inferred = float(max(dx, dy))  # x, y 중 큰 쪽 사용

    if inferred < default_patch_size_l0 * 0.5 or inferred > default_patch_size_l0 * 2.0:
        print(f"[WARN] inferred patch_size_l0={inferred}, "
              f"fallback to default={default_patch_size_l0}")
        return default_patch_size_l0

    print(f"[DEBUG] inferred patch_size_l0={inferred}")
    return inferred


def get_stem_to_file_map(folder: str, exts=None):
    """
    folder 아래를 재귀적으로 돌면서
    stem(확장자 뺀 파일명) -> 전체 경로 매핑 생성.
    (WSI, npy용)
    """
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


def get_h5_stem_to_file_map(folder: str, exts=None):
    """
    h5 파일용: stem 끝의 '_patches'를 슬라이드명으로 정규화.
    예) 'P001_patches.h5' -> key: 'P001'
    """
    stem_map = {}
    if exts is not None:
        exts = tuple(e.lower() for e in exts)

    for root, _, files in os.walk(folder):
        for fname in files:
            stem, ext = os.path.splitext(fname)
            ext = ext.lower()
            if exts is not None and ext not in exts:
                continue

            if stem.endswith('_patches'):
                key = stem[:-8]  # len('_patches') == 8
            else:
                key = stem

            fpath = os.path.join(root, fname)
            stem_map[key] = fpath

    return stem_map


# ==========================
# --- CLAM 스타일 히트맵 (단일 슬라이드 로직 이식) ---
# ==========================
def get_clam_style_heatmap(slide_path: str,
                           coords_path: str,
                           scores_path: str,
                           output_path: str):
    print(f"\n=== Processing: {os.path.basename(slide_path)} ===")
    print("Loading slide and coordinates...")

    slide = openslide.OpenSlide(slide_path)

    with h5py.File(coords_path, 'r') as f:
        coords = f['coords'][:]   # (N, 2)

    scores = np.load(scores_path)
    if scores.ndim > 1:
        scores = scores.flatten()

    assert len(coords) == len(scores), f"Coords({len(coords)})와 Scores({len(scores)}) 개수가 안 맞음!"

    # 1) 점수 정규화 (CLAM 스타일)
    if USE_PERCENTILES:
        print("Converting scores to percentiles (CLAM Style)...")
        scores = rankdata(scores) / len(scores)  # 0~1로 스케일
    else:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # 2) 시각화 level 결정
    if VIS_LEVEL < 0:
        best_level = slide.level_count - 1
        for i in range(slide.level_count):
            if slide.level_dimensions[i][0] < 5000:
                best_level = i
                break
        vis_level = best_level
    else:
        vis_level = VIS_LEVEL

    w_vis, h_vis = slide.level_dimensions[vis_level]
    ds_vis = float(slide.level_downsamples[vis_level])
    print(f"Visualization Level: {vis_level} (Size: {w_vis}x{h_vis}, DS: {ds_vis})")

    # 3) Trident coords는 이미 Level 0 기준 → scale = 1.0
    coord_to_level0 = 1.0
    coords_l0 = coords.astype(np.float32) * coord_to_level0

    # 4) Level 0에서 patch 크기 추정
    patch_size_l0 = PATCH_SIZE * coord_to_level0
    patch_size_l0 = infer_patch_size_from_coords(coords_l0, patch_size_l0)

    # vis_level 기준으로 스케일링
    scaled_patch_size = max(1, int(round(patch_size_l0 / ds_vis)))
    print(f"[DEBUG] patch_size_l0={patch_size_l0}, scaled_patch_size={scaled_patch_size}")

    # 5) 히트맵 캔버스 준비
    print("Drawing heatmap mask...")
    heatmap_mask = np.zeros((h_vis, w_vis), dtype=np.float32)
    count_mask = np.zeros((h_vis, w_vis), dtype=np.float32)

    # 6) 패치마다 score 채우기
    for (x0, y0), score in zip(coords_l0, scores):
        x = int(x0 / ds_vis)
        y = int(y0 / ds_vis)

        if x >= w_vis or y >= h_vis:
            continue

        x_end = min(x + scaled_patch_size, w_vis)
        y_end = min(y + scaled_patch_size, h_vis)

        heatmap_mask[y:y_end, x:x_end] += float(score)
        count_mask[y:y_end, x:x_end] += 1.0

    # 7) overlap 있으면 평균 내기
    nonzero = count_mask > 0
    heatmap_mask[nonzero] /= count_mask[nonzero]
    heatmap_mask[~nonzero] = 0.0

    # 8) 0~255로 정규화
    if heatmap_mask.max() > 0:
        heatmap_norm = (heatmap_mask / heatmap_mask.max()) * 255.0
    else:
        heatmap_norm = heatmap_mask * 0
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # 9) 블러 처리 (CLAM 스타일)
    if BLUR:
        k_size = max(3, (scaled_patch_size * 2) | 1)
        print(f"[DEBUG] GaussianBlur ksize={k_size}")
        heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8, (k_size, k_size), 0)

    # 10) 컬러맵 입히기
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 11) 조직이 있는 위치만 마스킹 (coords 있었던 곳)
    tissue_mask = (count_mask > 0).astype(np.uint8)

    # 12) 원본 이미지 읽기
    print("Overlaying on WSI...")
    original_img = slide.read_region((0, 0), vis_level, (w_vis, h_vis)).convert("RGB")
    original_img = np.array(original_img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(original_img, 1 - ALPHA, heatmap_color, ALPHA, 0)

    # 조직 없는 부분은 원본만 남기기
    idx_bg = np.where(tissue_mask == 0)
    overlay[idx_bg] = original_img[idx_bg]

    cv2.imwrite(output_path, overlay)
    print(f"Saved heatmap to: {output_path}")


# ==========================
# --- Case / Control 그룹 처리 ---
# ==========================
def generate_group_heatmaps(wsi_dir: str,
                            h5_dir: str,
                            npy_map: dict,
                            out_dir_group: str):
    """
    한 그룹(case 또는 control)에 대해:
    - wsi_dir: 해당 그룹 WSI 루트
    - h5_dir: 해당 그룹 h5(coords) 루트
    - npy_map: (전체 공통) stem -> npy 경로 dict
    - out_dir_group: 출력 폴더 (예: out/case)
    """
    os.makedirs(out_dir_group, exist_ok=True)

    wsi_exts = ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.svslide', '.bif']

    print(f"\n[Group: {os.path.basename(out_dir_group)}] 파일 매핑 중...")

    wsi_map = get_stem_to_file_map(wsi_dir, wsi_exts)
    h5_map = get_h5_stem_to_file_map(h5_dir, ['.h5', '.hdf5'])

    common_stems = sorted(set(wsi_map.keys()) & set(h5_map.keys()) & set(npy_map.keys()))
    print(f"  공통 stem 개수: {len(common_stems)}")

    if not common_stems:
        print("  ⚠ 공통 stem이 없습니다. (이름 규칙/경로 확인 필요)")
        return

    for idx, stem in enumerate(common_stems, 1):
        print(f"  [{idx}/{len(common_stems)}] {stem}")

        slide_path = wsi_map[stem]
        coords_path = h5_map[stem]
        scores_path = npy_map[stem]

        out_path = os.path.join(out_dir_group, f"{stem}_heatmap.jpg")

        try:
            get_clam_style_heatmap(
                slide_path=slide_path,
                coords_path=coords_path,
                scores_path=scores_path,
                output_path=out_path
            )
        except Exception as e:
            print(f"  [ERROR] {stem} 처리 중 오류: {e}")
            continue


# ==========================
# --- Main ---
# ==========================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="CLAM-style Heatmap Generator (WSI/h5는 case/control 분리, npy는 공통 폴더, h5에 _patches suffix)"
    )

    parser.add_argument('--wsi_case_dir', type=str, required=True,
                        help='case WSI 루트 폴더')
    parser.add_argument('--wsi_control_dir', type=str, required=True,
                        help='control WSI 루트 폴더')

    parser.add_argument('--h5_case_dir', type=str, required=True,
                        help='case h5(coords) 루트 폴더')
    parser.add_argument('--h5_control_dir', type=str, required=True,
                        help='control h5(coords) 루트 폴더')

    parser.add_argument('--npy_dir', type=str, required=True,
                        help='모든 attention npy가 들어있는 루트 폴더')

    parser.add_argument('--out_dir', type=str, required=True,
                        help='heatmap 출력 루트 폴더')

    args = parser.parse_args()

    # 1) npy는 한 번만 전체 매핑
    npy_map = get_stem_to_file_map(args.npy_dir, ['.npy'])
    print(f"[INFO] npy 파일 수: {len(npy_map)}")

    # 2) CASE 그룹
    generate_group_heatmaps(
        wsi_dir=args.wsi_case_dir,
        h5_dir=args.h5_case_dir,
        npy_map=npy_map,
        out_dir_group=os.path.join(args.out_dir, "case")
    )

    # 3) CONTROL 그룹
    generate_group_heatmaps(
        wsi_dir=args.wsi_control_dir,
        h5_dir=args.h5_control_dir,
        npy_map=npy_map,
        out_dir_group=os.path.join(args.out_dir, "control")
    )
