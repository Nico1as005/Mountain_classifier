
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max


def standardize_image(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    # Стандартизация изображения
    h, w = img.shape[:2]
    scale = target_size / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    cropped = resized[top:top + target_size, left:left + target_size]
    return cropped


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _water_and_land_mask(img: np.ndarray):
    # Отделение воды от суши
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    water_mask = (s_ch < 100) & (v_ch > 170)
    out_of_bounds_mask = (h_ch > 100) & (h_ch < 140) & (s_ch > 100) & (v_ch < 100)
    land_mask = ~(water_mask | out_of_bounds_mask)
    return water_mask, out_of_bounds_mask, land_mask


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_density(gray: np.ndarray, low=50, high=150) -> float:
    edges = cv2.Canny(gray, low, high)
    return float(np.mean(edges > 0))


def local_std_mean(gray: np.ndarray, ksize: int = 15) -> float:
    # рассчет среднего отклонения локальной яркости
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f ** 2, (ksize, ksize))
    local_var = np.clip(sq_mean - mean ** 2, 0, None)
    local_std = np.sqrt(local_var)
    return float(np.mean(local_std))


def glcm_features(gray: np.ndarray, distances=(3,), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)) -> dict:
    gray_q = (gray.astype(np.float32) / 255 * 31).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=list(distances), angles=list(angles),
                         levels=32, symmetric=True, normed=True)
    out = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        out[f"glcm_{prop}"] = float(np.mean(graycoprops(glcm, prop)))
    return out


def bimodality_score(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    p10 = np.percentile(gray, 10)
    p90 = np.percentile(gray, 90)
    dark_frac = float(np.mean(gray <= p10))
    bright_frac = float(np.mean(gray >= p90))
    spread = float(p90 - p10)
    return dark_frac + bright_frac, spread


def bright_blob_features(gray: np.ndarray, land_mask: np.ndarray, quantile: float = 0.85) -> dict:
    # Проверка наличия "плато"
    empty_result = {
        "blob_max_area_frac": 0.0,
        "blob_max_compactness": 0.0,
        "blob_count_significant": 0,
        "blob_solidity": 1.0,
        "blob_defect_count": 0,
    }

    if land_mask.sum() < 0.05 * gray.size:
        return empty_result

    thresh_val = np.quantile(gray[land_mask], quantile)
    mask = ((gray >= thresh_val) & land_mask).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n_labels <= 1:
        return empty_result

    areas = stats[1:, cv2.CC_STAT_AREA]
    total_px = gray.shape[0] * gray.shape[1]
    max_area_idx = int(np.argmax(areas)) + 1
    max_area = float(areas[max_area_idx - 1])

    blob_mask = (labels == max_area_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    compactness = 0.0
    solidity = 1.0
    defect_count = 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        compactness = (perimeter ** 2) / (max_area + 1e-9)

        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = cv2.contourArea(largest) / hull_area

        hull_idx = cv2.convexHull(largest, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            try:
                hull_idx = np.sort(hull_idx, axis=0)[::-1]
                defects = cv2.convexityDefects(largest, hull_idx)
                if defects is not None and defects.ndim == 3 and defects.shape[1:] == (1, 4):
                    diag = (gray.shape[0] ** 2 + gray.shape[1] ** 2) ** 0.5
                    depths = defects[:, 0, 3] / 256.0
                    defect_count = int(np.sum(depths > 0.01 * diag))
            except Exception:
                defect_count = 0

    significant = int(np.sum(areas >= 0.002 * total_px))

    return {
        "blob_max_area_frac": max_area / total_px,
        "blob_max_compactness": compactness,
        "blob_count_significant": significant,
        "blob_solidity": solidity,
        "blob_defect_count": defect_count,
    }


def ridge_density(gray: np.ndarray, sigmas=(1, 2, 4)) -> float:
    # Обнаружение горных хребтов
    from skimage.filters import frangi
    gray_f = gray.astype(np.float64) / 255.0
    response = frangi(gray_f, sigmas=sigmas, black_ridges=False)
    return float(np.mean(response))


def terrain_shape_features(gray: np.ndarray, water_mask: np.ndarray, out_of_bounds_mask: np.ndarray,
                            land_mask: np.ndarray) -> dict:
    total_px = gray.shape[0] * gray.shape[1]
    land_px = int(land_mask.sum())

    empty = {
        "elevated_area_frac": 0.0,
        "elevated_blob_count_significant": 0,
        "elevated_concentration": 0.0,
        "elevated_largest_solidity": 1.0,
        "elevated_largest_compactness": 0.0,
        "water_area_frac": float(water_mask.mean()),
        "out_of_bounds_area_frac": float(out_of_bounds_mask.mean()),
    }

    if land_px < 0.05 * total_px:
        return empty

    land_values = gray[land_mask]
    try:
        thresh_val = threshold_otsu(land_values)
    except ValueError:
        return empty

    elevated_mask = (land_mask & (gray >= thresh_val)).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(elevated_mask, connectivity=8)

    if n_labels <= 1:
        return empty

    areas = stats[1:, cv2.CC_STAT_AREA]
    total_elevated = float(areas.sum())
    if total_elevated <= 0:
        return empty

    n_significant = int(np.sum(areas >= 0.01 * total_px))
    max_idx = int(np.argmax(areas)) + 1
    max_area = float(areas[max_idx - 1])
    concentration = max_area / total_elevated

    blob_mask = (labels == max_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 1.0
    compactness = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        compactness = (perimeter ** 2) / (area + 1e-9)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area

    return {
        "elevated_area_frac": total_elevated / total_px,
        "elevated_blob_count_significant": n_significant,
        "elevated_concentration": concentration,
        "elevated_largest_solidity": solidity,
        "elevated_largest_compactness": compactness,
        "water_area_frac": float(water_mask.mean()),
        "out_of_bounds_area_frac": float(out_of_bounds_mask.mean()),
    }


def peak_count_features(gray: np.ndarray, land_mask: np.ndarray,
                         sigma: float = 5, min_distance: int = 20, percentile: float = 50) -> dict:
    # Производит подсчет локальных максимумов по яркости
    if land_mask.sum() == 0:
        return {"n_peaks_significant": 0}

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    masked = blurred.copy()
    masked[~land_mask] = 0

    threshold_abs = float(np.percentile(gray[land_mask], percentile))
    coords = peak_local_max(masked, min_distance=min_distance, threshold_abs=threshold_abs)

    return {"n_peaks_significant": int(len(coords))}


def _neutralize_non_land(gray: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    # Функция замены нетеррейна на среднее по округе
    if land_mask.sum() == 0:
        return gray
    land_mean = float(gray[land_mask].mean())
    result = gray.copy()
    result[~land_mask] = np.uint8(round(land_mean))
    return result


def extract_features(img: np.ndarray, standardize: bool = True, target_size: int = 512) -> dict:
    if standardize:
        img = standardize_image(img, target_size=target_size)

    gray = _to_gray(img)
    water_mask, out_of_bounds_mask, land_mask = _water_and_land_mask(img)
    gray_texture = _neutralize_non_land(gray, land_mask)

    feats = {
        "laplacian_var": laplacian_variance(gray_texture),
        "edge_density": edge_density(gray_texture),
        "local_std_mean": local_std_mean(gray_texture),
    }
    feats.update(glcm_features(gray_texture))

    bimodal, spread = bimodality_score(gray_texture)
    feats["bimodality"] = bimodal
    feats["brightness_spread"] = spread

    feats.update(bright_blob_features(gray, land_mask))
    feats["ridge_density"] = ridge_density(gray)
    feats.update(terrain_shape_features(gray, water_mask, out_of_bounds_mask, land_mask))
    feats.update(peak_count_features(gray, land_mask))

    return feats


def extract_features_from_path(path: str) -> dict:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return extract_features(img)


FEATURE_NAMES = [
    "laplacian_var", "edge_density", "local_std_mean",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "bimodality", "brightness_spread",
    "blob_max_area_frac", "blob_max_compactness", "blob_count_significant",
    "blob_solidity", "blob_defect_count",
    "ridge_density",
    "elevated_area_frac", "elevated_blob_count_significant",
    "elevated_concentration", "elevated_largest_solidity",
    "elevated_largest_compactness", "water_area_frac", "out_of_bounds_area_frac",
    "n_peaks_significant",
]