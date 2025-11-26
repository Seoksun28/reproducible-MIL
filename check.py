import openslide
import h5py
import numpy as np
import cv2

WSI_PATH = "슬라이드.svs 경로"
H5_PATH  = "문제의 coords.h5 경로"

# patch 3개 정도만 샘플링해서 실제 픽셀 크기 측정
N = 3

with h5py.File(H5_PATH, "r") as f:
    coords = f["coords"][:]

slide = openslide.OpenSlide(WSI_PATH)

for i in np.random.choice(len(coords), N, replace=False):
    x, y = coords[i]
    patch = slide.read_region((int(x), int(y)), 0, (1024,1024))   # 1024까지 크게 읽기
    patch = np.array(patch)[..., :3]

    # 연속된 조직/비조직 변화가 생기는 지점 탐지 → patch_size 추정
    edges = cv2.Canny(patch, 50,150)
    ys, xs = np.where(edges > 0)

    px_width = xs.max() - xs.min()
    py_height = ys.max() - ys.min()

    print(f"coords[{i}] → estimated patch_size ~ {px_width} x {py_height}")
