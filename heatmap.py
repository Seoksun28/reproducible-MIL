import os
import numpy as np
import h5py
from trident import WSI
from trident import visualize_heatmap  # trident 버전에 따라 visualizaton 모듈일 수도 있음

wsi_path = "path/to/slide.svs"
h5_path = "path/to/slide.h5"
attn_path = "path/to/slide_attention.npy"

# 1) WSI
wsi = WSI.from_file(wsi_path)

# 2) h5에서 coords 꺼내기
with h5py.File(h5_path, "r") as f:
    coords = f["coords"][:]   # [M, 2] 이길 기대

print("coords.shape:", coords.shape)

# 필요시 transpose
# if coords.shape[0] == 2:
#     coords = coords.T

# 3) attention 로드 & shape 정리
scores = np.load(attn_path)   # (1, 9098) 같은 형태
scores = scores.reshape(-1)   # (9098,)

print("scores.shape:", scores.shape)
assert coords.shape[0] == scores.shape[0]

# 4) patch size 직접 설정
patch_size_level0 = 256  # Trident patch_size랑 동일하게

# 5) heatmap 생성
out_dir = "heatmap_outputs"
os.makedirs(out_dir, exist_ok=True)

out_path = visualize_heatmap(
    wsi=wsi,
    scores=scores,
    coords=coords,
    patch_size_level0=patch_size_level0,
    vis_level=2,
    cmap="coolwarm",
    normalize=True,
    num_top_patches_to_save=-1,
    output_dir=out_dir,
    overlay_only=True,
    filename="slide_heatmap.png",
)

print("saved heatmap:", out_path)
