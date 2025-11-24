import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from openslide import OpenSlide

# ------------------------------------------
# 설정
# ------------------------------------------
wsi_path = "path/to/slide.svs"
h5_path = "path/to/slide.h5"
attn_path = "path/to/slide_attention.npy"

patch_size = 256        # Level0 patch size (Trident patch size)
vis_level = 2           # Thumbnail level
sigma = 20              # Gaussian blur
alpha = 0.6             # Overlay transparency
output_path = "heatmap.png"
# ------------------------------------------

# 1) Openslide로 슬라이드 로드
slide = OpenSlide(wsi_path)

# 2) Thumbnail 추출
thumb = slide.read_region(
    (0, 0),
    vis_level,
    slide.level_dimensions[vis_level]
).convert("RGB")
thumb = np.array(thumb)

# 3) coords & attention score 로드
with h5py.File(h5_path, 'r') as f:
    coords = f['coords'][:]   # shape (N, 2)

scores = np.load(attn_path).reshape(-1)
assert coords.shape[0] == scores.shape[0]

# 4) Level 변환
down = slide.level_downsamples[vis_level]
coords_lv = (coords / down).astype(int)

# 5) Heatmap canvas 생성
H, W = thumb.shape[:2]
heat = np.zeros((H, W), dtype=float)

# 6) Patch 단위로 score 반영
patch_w = int(patch_size / down)
patch_h = int(patch_size / down)

for (x, y), s in zip(coords_lv, scores):
    x2 = min(W, x + patch_w)
    y2 = min(H, y + patch_h)
    heat[y:y2, x:x2] += s

# 7) Gaussian blur
heat_blur = gaussian_filter(heat, sigma=sigma)

# 8) Normalization
norm = Normalize(
    vmin=np.percentile(heat_blur, 1),
    vmax=np.percentile(heat_blur, 99)
)
heat_norm = norm(heat_blur)

# 9) PNG로 저장
plt.figure(figsize=(10, 10))
plt.imshow(thumb)
plt.imshow(heat_norm, cmap="coolwarm", alpha=alpha)
plt.axis("off")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"저장 완료 → {output_path}")
