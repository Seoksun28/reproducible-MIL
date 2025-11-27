import openslide
import h5py
import numpy as np
import cv2
from scipy.stats import rankdata

# --- 설정 구간 (여기만 수정하면 됨) ---
SLIDE_PATH = 'path/to/slide.svs'      # WSI 파일 경로
COORDS_PATH = 'path/to/patches.h5'    # Trident coords h5 (coords: Level 0 기준)
SCORES_PATH = 'path/to/scores.npy'    # Attention score npy 파일
OUTPUT_PATH = 'clam_style_heatmap.jpg'

PATCH_SIZE = 256        # "모델 학습에 쓴 patch 크기" (예: 256x256 at target_mag)
VIS_LEVEL = 2           # 시각화할 level (0 = 원본, 1/2 추천, -1이면 자동 선택)
ALPHA = 0.4             # 히트맵 투명도 (0.0 ~ 1.0, CLAM은 0.4 근처)
BLUR = True             # True면 부드럽게 Gaussian blur (CLAM 스타일)
USE_PERCENTILES = True  # True면 rank -> 0~1 percentile로 변환 (CLAM 스타일)
# ------------------------------------


def infer_patch_size_from_coords(coords_l0: np.ndarray,
                                 default_patch_size_l0: float) -> float:
    """
    coords_l0: level-0 기준 좌표 (N, 2)
    default_patch_size_l0: PATCH_SIZE * coord_to_level0 로 계산한 대략 값

    → coords 간 최소 간격(dx, dy)을 이용해 실제 patch 한 변 길이를 추정.
      (Trident 20x / patch_size 256 @40x 원본이면 보통 512로 나옴)
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

    inferred = float(max(dx, dy))  # x, y 중 큰 쪽을 patch 크기로 간주

    # 너무 말이 안 되면 기본값으로 롤백
    if inferred < default_patch_size_l0 * 0.5 or inferred > default_patch_size_l0 * 2.0:
        print(
            f"[WARN] inferred patch_size_l0={inferred}이(default={default_patch_size_l0})와 너무 달라 "
            f"기본값({default_patch_size_l0}) 사용"
        )
        return default_patch_size_l0

    print(f"[DEBUG] inferred patch_size_l0 from coords = {inferred}")
    return inferred


def get_clam_style_heatmap():
    print("Loading slide and coordinates...")
    slide = openslide.OpenSlide(SLIDE_PATH)

    with h5py.File(COORDS_PATH, 'r') as f:
        # Trident coords: 보통 h5 안에 'coords' (N, 2) 저장돼 있음
        coords = f['coords'][:]   # (N, 2)

    scores = np.load(SCORES_PATH)
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
        # 자동 선택: 가로 길이가 5000 이하가 되는 가장 높은 해상도 level
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
    # 1차 기본값: PATCH_SIZE (target_mag에서의 patch 크기) * scale
    patch_size_l0 = PATCH_SIZE * coord_to_level0

    # 2차 보정: coords 간 간격으로 실제 patch footprint 추론 (Trident 20x@40x면 512일 것)
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
        # patch 두 칸 정도 커버하는 커널, 홀수로 맞추기
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

    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"Saved heatmap to: {OUTPUT_PATH}")


if __name__ == '__main__':
    get_clam_style_heatmap()
