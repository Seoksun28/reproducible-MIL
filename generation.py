import openslide
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# --- 설정 구간 (여기만 수정하면 됨) ---
SLIDE_PATH = 'path/to/slide.svs'      # WSI 파일 경로
COORDS_PATH = 'path/to/patches.h5'    # Patch 좌표 h5 파일
SCORES_PATH = 'path/to/scores.npy'    # Attention score npy 파일
OUTPUT_PATH = 'clam_style_heatmap.jpg'

PATCH_SIZE = 256        # 학습할 때 쓴 패치 크기 (보통 256)
VIS_LEVEL = 2           # 시각화할 배율 (1~2 추천, -1이면 자동 계산)
ALPHA = 0.4             # 투명도 (0.0 ~ 1.0, 0.4가 CLAM 기본값)
BLUR = True             # True면 부드럽게 블러 처리 (CLAM 스타일), False면 격자무늬
USE_PERCENTILES = True  # True면 점수를 등수로 변환 (CLAM 기본값 - 색대비가 뚜렷해짐)
# ------------------------------------

def get_clam_style_heatmap():
    # 1. 데이터 로드
    print(f"Loading data...")
    slide = openslide.OpenSlide(SLIDE_PATH)
    
    with h5py.File(COORDS_PATH, 'r') as f:
        coords = f['coords'][:]
    
    scores = np.load(SCORES_PATH)
    if scores.ndim > 1: scores = scores.flatten()

    # 좌표와 점수 개수 맞는지 확인
    assert len(coords) == len(scores), f"Coords({len(coords)})와 Scores({len(scores)}) 개수가 안 맞음!"

    # 2. Score 처리 (CLAM 스타일의 핵심: Percentile)
    if USE_PERCENTILES:
        print("Converting scores to percentiles (CLAM Style)...")
        # 점수를 0~100 등급(백분위)으로 변환 -> 색깔 구분이 확연해짐
        scores = rankdata(scores) / len(scores) * 100
        scores = scores / 100.0 # 0.0 ~ 1.0 정규화
    else:
        # 그냥 Min-Max 정규화
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    # 3. 캔버스 준비
    # VIS_LEVEL이 -1이면 적당한 크기(가로 3000px 근처) 찾기
    if VIS_LEVEL < 0:
        best_level = slide.level_count - 1
        for i in range(slide.level_count):
            if slide.level_dimensions[i][0] < 5000:
                best_level = i
                break
        vis_level = best_level
    else:
        vis_level = VIS_LEVEL

    w, h = slide.level_dimensions[vis_level]
    downsample = slide.level_downsamples[vis_level]
    print(f"Visualization Level: {vis_level} (Size: {w}x{h}, DS: {downsample})")

    # 4. Heatmap 마스크 그리기
    print("Drawing heatmap mask...")
    heatmap_mask = np.zeros((h, w), dtype=np.float32)
    
    # 겹침 방지 및 평균 계산을 위한 카운트 마스크 (Overlap 처리용)
    count_mask = np.zeros((h, w), dtype=np.float32)

    scaled_patch_size = int(PATCH_SIZE / downsample)
    
    for coord, score in zip(coords, scores):
        x, y = int(coord[0] / downsample), int(coord[1] / downsample)
        
        # 범위 체크
        y_end = min(y + scaled_patch_size, h)
        x_end = min(x + scaled_patch_size, w)
        
        if y < h and x < w:
            heatmap_mask[y:y_end, x:x_end] += score
            count_mask[y:y_end, x:x_end] += 1

    # 중첩된 부분 평균 내기 (Overlap이 있었다면)
    count_mask[count_mask == 0] = 1 # 0으로 나누기 방지
    heatmap_mask = heatmap_mask / count_mask

    # 5. 스무딩 & 컬러 입히기
    # 0~255로 변환
    heatmap_uint8 = (heatmap_mask * 255).astype(np.uint8)

    if BLUR:
        # CLAM은 격자 무늬를 없애기 위해 Gaussian Blur를 씀
        k_size = int(scaled_patch_size * 2) | 1 # 홀수여야 함
        heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8, (k_size, k_size), 0)

    # JET Colormap 적용 (파랑:낮음 -> 빨강:높음)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 배경(조직 없는 부분) 처리: count_mask가 0인 부분은 검은색(또는 흰색)이 아니라 투명해야 함
    # 하지만 overlay를 위해 일단 검은색으로 둠. 나중에 원본이랑 섞을 때 처리.
    # 마스크가 0인 부분(조직 없음)을 검출
    tissue_mask = (count_mask > 1.0/255.0).astype(np.uint8) # 조금이라도 값이 있으면 조직

    # 6. 원본 이미지와 합치기
    print("Overlaying on WSI...")
    original_img = slide.read_region((0, 0), vis_level, (w, h)).convert("RGB")
    original_img = np.array(original_img)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    # cv2.addWeighted 사용
    # heatmap_color가 있는 부분만 블렌딩하고, 없는 부분은 원본 그대로 둠
    
    # 1) 전체를 블렌딩
    overlay = cv2.addWeighted(original_img, 1 - ALPHA, heatmap_color, ALPHA, 0)
    
    # 2) 조직이 없는 부분(tissue_mask == 0)은 원본 이미지로 덮어쓰기
    # (이렇게 해야 배경이 파랗게 안 뜸)
    non_tissue_indices = np.where(tissue_mask == 0)
    overlay[non_tissue_indices] = original_img[non_tissue_indices]

    # 7. 저장
    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"Saved heatmap to: {OUTPUT_PATH}")

if __name__ == '__main__':
    get_clam_style_heatmap()
