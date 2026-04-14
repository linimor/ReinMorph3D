import cv2
import numpy as np
import os


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def edge_preserving_smooth(gray, d=9, sigma_color=30, sigma_space=30):
    """
    双边滤波:
    平滑区域内部，同时尽量保留边缘
    d 越大平滑越强
    sigma_color 越大，对灰度差异更不敏感
    sigma_space 越大，空间范围更大
    """
    return cv2.bilateralFilter(gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def enhance_structure(gray):
    """
    轻度结构增强，不追求细节锐化
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def mild_unsharp(gray, sigma=1.2, amount=0.6):
    """
    轻微软锐化:
    只把边界拉回来一点，避免整体发糊
    """
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def soft_edge_map(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)
    mag = cv2.GaussianBlur(mag, (0, 0), 0.8)
    return mag


def fuse_gray_and_edges(gray, edges, edge_weight=0.12):
    fused = gray.astype(np.float32) * (1.0 - edge_weight) + edges.astype(np.float32) * edge_weight
    return np.clip(fused, 0, 255).astype(np.uint8)


def process_for_image23d(image_path, output_dir="output_image23d"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = to_gray(img)

    # 1. 保边平滑，去掉部分细节但不把轮廓糊掉
    smooth = edge_preserving_smooth(gray, d=9, sigma_color=35, sigma_space=35)

    # 2. 增强结构层次
    structure = enhance_structure(smooth)

    # 3. 轻微软锐化，把边界感拉回来一点
    sharpened = mild_unsharp(structure, sigma=1.0, amount=0.45)

    # 4. 弱边缘提示
    edges = soft_edge_map(sharpened)

    # 5. 轻量融合
    final = fuse_gray_and_edges(sharpened, edges, edge_weight=0.10)

    cv2.imwrite(os.path.join(output_dir, "01_gray.png"), gray)
    cv2.imwrite(os.path.join(output_dir, "02_smooth_edge_preserving.png"), smooth)
    cv2.imwrite(os.path.join(output_dir, "03_structure_enhanced.png"), structure)
    cv2.imwrite(os.path.join(output_dir, "04_sharpened_structure.png"), sharpened)
    cv2.imwrite(os.path.join(output_dir, "05_soft_edges.png"), edges)
    cv2.imwrite(os.path.join(output_dir, "06_final_for_image23d.png"), final)

    print("Saved to:", output_dir)


if __name__ == "__main__":
    process_for_image23d("/root/autodl-tmp/MorphAny3D/output_image23d/0001.png")