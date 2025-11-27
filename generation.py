import openslide
import h5py
import numpy as np
import cv2
from scipy.stats import rankdata

# --- 설정 구간 (여기만 수정하면 됨) ---
SLIDE_PATH = 'path/to/slide.svs'      # WSI 파일 경로
COORDS_PATH = 'path/to/patches.h5'    # Patch 좌표 h5 파일 (npz에서 쓰던 coords랑 같은 레벨이어야 함)
SCORES_PATH = 'path/to/scores.npy'    # Attention score npy 파일
OUTPUT_PATH = 'clam_style_heatmap.jpg'

PATCH_SIZE = 256        # "학습에 쓰인 patch 이미지 크기" (예: 256x256)
VIS_LEVEL = 2           # 시각화할 배율 (1~2 추천, -1이면 자동 계산)
ALPHA = 0.4             # 투명도 (0.0 ~ 1.0, 0.4가 CLAM 기본값)
BLUR = True             # True면 부드럽게 블러 처리 (CLAM 스타일), False면 격자무늬
USE_PERCENTILES = True  # True면 점수를 등수로 변환 (CLAM 기본값 - 색대비가 뚜렷해짐)
# ------------------------------------


def infer_coord_scale(slide, coords, patch_size):
    """
    coords가 level-0 기준인지, 이미 downsample된 좌표인지 대략 추정.
    - 반환값: coord_to_level0_scale (coords * scale -> level0 좌표)
    """
    w0, h0 = slide.level_dimensions[0]  # level 0 크기
    max_x = coords[:, 0].max() + patch_size
    max_y = coords[:, 1].max() + patch_size

    # 좌표가 level0에 거의 꽉 차 있으면 scale ~ 1
    sx = w0 / max_x
    sy = h0 / max_y
    est = (sx + sy) / 2.0

    # 보통 WSI는 40x/20x/10x 관계 → 0.5, 1, 2, 4 정도에서 골라주면 됨
    candidates = np.array([0.5, 1.0, 2.0, 4.0])
    best = candidates[np.argmin(np.abs(candidates - est))]
    print(f"[DEBUG] inferred coord_to_level0_scale ≈ {best:.2f} (raw est={est:.2f})")
    return best


def get_clam_style_heatmap():
    print(f"Loading data...")
    slide = openslide.OpenSlide(SLIDE_PATH)

    with h5py.File(COORDS_PATH, 'r') as f:
        coords = f['coords'][:]   # (N, 2)

    scores = np.load(SCORES_PATH)
    if scores.ndim > 1:
        scores = scores.flatten()

    assert len(coords) == len(scores), f"Coords({len(coords)})와 Scores({len(scores)}) 개수가 안 맞음!"

    # 1) 점수 정규화 (CLAM 스타일)
    if USE_PERCENTILES:
        print("Converting scores to percentiles (CLAM Style)...")
        scores = rankdata(scores) / len(scores)  # 0~1
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

    # 3) coords가 어떤 스케일인지 추정해서 level-0 좌표로 보정
    ### >>> 여기 추가: coords 기준 스케일 추정
    coord_to_level0 = infer_coord_scale(slide, coords, PATCH_SIZE)
    coords_l0 = coords.astype(np.float32) * coord_to_level0

    # 4) level-0 패치 크기 계산 후 vis_level 기준으로 스케일
    patch_size_l0 = PATCH_SIZE * coord_to_level0
    scaled_patch_size = max(1, int(round(patch_size_l0 / ds_vis)))
    print(f"[DEBUG] patch_size_l0={patch_size_l0}, scaled_patch_size={scaled_patch_size}")

    # 5) 히트맵 캔버스
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

    # 7) overlap 있으면 평균
    nonzero = count_mask > 0
    heatmap_mask[nonzero] /= count_mask[nonzero]
    heatmap_mask[~nonzero] = 0.0

    # 8) 0~255로 변환
    if heatmap_mask.max() > 0:
        heatmap_norm = (heatmap_mask / heatmap_mask.max()) * 255.0
    else:
        heatmap_norm = heatmap_mask * 0
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # 9) 블러 (CLAM 스타일)
    if BLUR:
        k_size = max(3, (scaled_patch_size * 2) | 1)  # 홀수
        print(f"[DEBUG] GaussianBlur ksize={k_size}")
        heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8, (k_size, k_size), 0)

    # 10) 컬러맵
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 11) 조직이 있는 위치만 남기기
    tissue_mask = (count_mask > 0).astype(np.uint8)
    # 필요하면 erode/dilate 해도 됨

    # 12) 원본 이미지 읽기
    print("Overlaying on WSI...")
    original_img = slide.read_region((0, 0), vis_level, (w_vis, h_vis)).convert("RGB")
    original_img = np.array(original_img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(original_img, 1 - ALPHA, heatmap_color, ALPHA, 0)

    # 조직 없는 부분은 원본 유지
    idx_bg = np.where(tissue_mask == 0)
    overlay[idx_bg] = original_img[idx_bg]

    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"Saved heatmap to: {OUTPUT_PATH}")


if __name__ == '__main__':
    get_clam_style_heatmap()
