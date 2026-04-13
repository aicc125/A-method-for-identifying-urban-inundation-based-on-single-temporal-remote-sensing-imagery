import os
import numpy as np
import rasterio
from skimage.filters import threshold_otsu
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import (
    closing, opening, disk, remove_small_objects, dilation,
    skeletonize, thin
)
from scipy.ndimage import distance_transform_edt, label as ndi_label, generic_filter
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import warnings
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')


# ========== User-adjustable Parameters ==========
INPUT_TIF = r"data/input.tif"
OUTPUT_DIR = r"output/flood"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Water extraction parameters
WATER_CLOSING_RADIUS = 2
WATER_BRIGHTNESS_FALLBACK = 0.3    # Fallback value when dynamic brightness calculation fails

# River and lake exclusion parameters
WATER_MAX_SIZE = 50000
LAKE_MIN_AREA = 5000
# COMPACTNESS_THRESHOLD: dynamically computed, see Step 6
COMPACTNESS_AUTO = True         # True = automatically determine compactness threshold
# Dynamic river scoring parameters
RIVER_SCORE_AUTO_THRESHOLD = True   # True = data-driven auto threshold; False = use manual value below
RIVER_SCORE_THRESHOLD_MANUAL = 0.55 # Only used when AUTO=False
RIVER_MIN_AREA = 200                # Minimum area for river score calculation
RIVER_SMALL_SAMPLE_N = 10           # Small sample threshold: use small-sample branch when n < this value

# Building extraction parameters (impervious surface)
BUILDING_CLOSING_RADIUS = 3
BUILDING_MIN_SIZE = 50

# Flood determination parameters
BUILDING_WATER_DISTANCE = 3
FLOOD_MIN_SIZE = 10

# Texture feature parameters
USE_TEXTURE_FEATURES = True
TEXTURE_WINDOW_SIZE = 5
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Bimodality test parameters
BIMODAL_D_THRESHOLD = 2.0          # Ashman's D > 2 indicates significant bimodality
BIMODAL_FALLBACK_THRESHOLD = 0.1   # Fallback MNDWI threshold for non-bimodal cases (>=0.1 to exclude shadow interference)

try:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

BAND_MAP = {
    'aerosol': 1, 'blue': 2, 'green': 3, 'red': 4,
    'nir': 8, 'swir1': 11, 'swir2': 12, 'scl': 15
}

def save_tif(data, profile, out_path, dtype=rasterio.float32, nodata=np.nan):
    p = profile.copy()
    p.update(count=1, dtype=dtype, nodata=nodata)
    arr = np.nan_to_num(data, nan=nodata).astype(dtype)
    with rasterio.open(out_path, 'w', **p) as dst:
        dst.write(arr, 1)

def normalize_for_vis(arr, percentile_clip=True, gamma=1.0):
    arr = np.nan_to_num(arr, nan=np.nanmin(arr))
    if percentile_clip:
        low, high = np.nanpercentile(arr, [2, 98])
        arr = np.clip(arr, low, high)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn == 0:
        return np.zeros_like(arr)
    arr = (arr - mn) / (mx - mn)
    arr = np.power(arr, gamma)
    return np.clip(arr, 0, 1)

def calculate_local_std(image, window_size=5):
    def local_std_func(values):
        return np.std(values)
    return generic_filter(image, local_std_func, size=window_size)

def get_pixel_area_m2(profile):
    transform = profile.get('transform')
    if transform is not None:
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])
        crs = profile.get('crs')
        if crs is not None and crs.is_geographic:
            pixel_width_m = pixel_width * 111320
            pixel_height_m = pixel_height * 111320
        else:
            pixel_width_m = pixel_width
            pixel_height_m = pixel_height
        return pixel_width_m * pixel_height_m
    return 100.0


def check_bimodality(data, n_bins=256):
    """
    Test bimodality of 1D data using Ashman's D coefficient.
    
    Method:
      Fit two Gaussian components (simplified via Otsu split) and compute separation:
        D = sqrt(2) * |mu1 - mu2| / sqrt(sigma1^2 + sigma2^2)
      D > 2 indicates the two components are reliably separable (bimodal).
    
    Reference:
      Ashman, K.A., Bird, C.M. & Zepf, S.E. (1994). 
      Detecting bimodality in astronomical datasets. 
      The Astronomical Journal, 108, 2348-2361.
    
    Auxiliary criterion (histogram valley depth ratio):
      The valley between two peaks should be significantly lower than both peaks:
        valley_ratio = valley_min / min(peak1, peak2)
      valley_ratio < 0.75 supports bimodality.
    
    Parameters:
        data: 1D array, data to test (e.g., valid MNDWI pixel values)
        n_bins: int, number of histogram bins
    
    Returns:
        is_bimodal: bool, whether the distribution is bimodal
        ashman_d: float, Ashman's D coefficient
        details: dict, diagnostic details
    """
    details = {}
    
    if len(data) < 100:
        details['reason'] = 'Insufficient sample size'
        return False, 0.0, details
    
    # ---- 1. Simplified GMM fitting via Otsu split ----
    try:
        otsu_val = threshold_otsu(data)
    except Exception:
        details['reason'] = 'Otsu computation failed'
        return False, 0.0, details
    
    group1 = data[data <= otsu_val]
    group2 = data[data > otsu_val]
    
    if len(group1) < 10 or len(group2) < 10:
        details['reason'] = f'Groups too small: group1={len(group1)}, group2={len(group2)}'
        return False, 0.0, details
    
    mu1, sigma1 = np.mean(group1), np.std(group1)
    mu2, sigma2 = np.mean(group2), np.std(group2)
    
    # ---- 2. Compute Ashman's D ----
    ashman_d = np.sqrt(2) * abs(mu1 - mu2) / (np.sqrt(sigma1**2 + sigma2**2) + 1e-10)
    
    details['otsu_threshold'] = otsu_val
    details['mu1'] = mu1
    details['sigma1'] = sigma1
    details['n1'] = len(group1)
    details['mu2'] = mu2
    details['sigma2'] = sigma2
    details['n2'] = len(group2)
    details['ashman_d'] = ashman_d
    
    # ---- 3. Auxiliary criterion: histogram valley depth ----
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    from scipy.ndimage import uniform_filter1d
    hist_smooth = uniform_filter1d(hist.astype(float), size=max(n_bins // 30, 3))
    
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            peaks.append(i)
    
    details['n_peaks'] = len(peaks)
    
    valley_ratio = 1.0
    if len(peaks) >= 2:
        peak_heights = [(hist_smooth[p], p) for p in peaks]
        peak_heights.sort(reverse=True)
        p1_idx = peak_heights[0][1]
        p2_idx = peak_heights[1][1]
        
        left_idx = min(p1_idx, p2_idx)
        right_idx = max(p1_idx, p2_idx)
        
        if right_idx > left_idx + 1:
            valley_min = np.min(hist_smooth[left_idx:right_idx+1])
            peak_min_height = min(hist_smooth[p1_idx], hist_smooth[p2_idx])
            valley_ratio = valley_min / (peak_min_height + 1e-10)
            
            details['peak1_center'] = bin_centers[p1_idx]
            details['peak2_center'] = bin_centers[p2_idx]
            details['valley_min'] = valley_min
            details['valley_ratio'] = valley_ratio
    
    details['valley_ratio'] = valley_ratio
    
    # ---- 4. Combined decision ----
    is_bimodal = (ashman_d > BIMODAL_D_THRESHOLD) and (valley_ratio < 0.75)
    
    if ashman_d > 3.0 and not is_bimodal:
        is_bimodal = True
        details['override'] = 'D>3, forced bimodal classification'
    
    ratio_groups = max(len(group1), len(group2)) / (min(len(group1), len(group2)) + 1)
    details['group_ratio'] = ratio_groups
    if ratio_groups > 20 and ashman_d < 3.0:
        is_bimodal = False
        details['override'] = f'Highly imbalanced groups ({ratio_groups:.1f}:1), classified as non-bimodal'
    
    return is_bimodal, ashman_d, details


def calculate_river_score(region, region_mask, mndwi, water_texture_std, 
                          image_shape, water_mask_full):
    """
    Dynamically compute a "river score" (0~1) for a water connected component.
    
    Combines multiple feature dimensions:
    1. Skeleton length ratio (skeleton_ratio): skeleton pixels / area, very high for rivers
    2. Width consistency (width_consistency): CV of width along skeleton, low for rivers
    3. MNDWI stability (spectral_stability): MNDWI std, low for rivers (uniform water)
    4. Boundary touch (boundary_touch): whether it touches image boundary, rivers often span the scene
    5. Sinuosity: skeleton length / endpoint distance, typically >1 for rivers
    6. Aspect ratio: auxiliary feature, no longer the sole criterion
    
    Parameters:
        region: regionprops object
        region_mask: binary mask of this region
        mndwi: MNDWI image
        water_texture_std: water texture standard deviation
        image_shape: (H, W) image dimensions
        water_mask_full: full water mask (for skeleton computation)
    
    Returns:
        score: float, composite river score 0~1
        details: dict, per-dimension score details
    """
    area = region.area
    h, w = image_shape
    
    details = {}
    scores = []
    weights = []
    
    # ======== 1. Skeleton length ratio ========
    try:
        bbox = region.bbox
        pad = 2
        r0 = max(0, bbox[0] - pad)
        c0 = max(0, bbox[1] - pad)
        r1 = min(h, bbox[2] + pad)
        c1 = min(w, bbox[3] + pad)
        
        local_mask = region_mask[r0:r1, c0:c1].copy()
        skeleton = skeletonize(local_mask)
        skeleton_length = np.sum(skeleton)
        
        skeleton_ratio = skeleton_length / (np.sqrt(area) + 1e-6)
        skeleton_score = np.clip((skeleton_ratio - 1.0) / 3.0, 0, 1)
        details['skeleton_ratio'] = skeleton_ratio
        details['skeleton_score'] = skeleton_score
        scores.append(skeleton_score)
        weights.append(0.25)
    except Exception:
        skeleton = None
        skeleton_score = 0
        details['skeleton_ratio'] = 0
        details['skeleton_score'] = 0
    
    # ======== 2. Width consistency ========
    try:
        if skeleton is not None and np.sum(skeleton) > 5:
            local_dist = distance_transform_edt(local_mask)
            widths = local_dist[skeleton]
            widths = widths[widths > 0]
            
            if len(widths) > 3:
                width_mean = np.mean(widths)
                width_std = np.std(widths)
                width_cv = width_std / (width_mean + 1e-6)
                width_score = np.clip(1.0 - (width_cv / 0.6), 0, 1)
                details['width_cv'] = width_cv
                details['width_mean_px'] = width_mean
            else:
                width_score = 0.3
                details['width_cv'] = -1
        else:
            width_score = 0.3
            details['width_cv'] = -1
        
        details['width_score'] = width_score
        scores.append(width_score)
        weights.append(0.2)
    except Exception:
        width_score = 0.3
        details['width_score'] = width_score
        details['width_cv'] = -1
        scores.append(width_score)
        weights.append(0.2)
    
    # ======== 3. MNDWI spectral stability ========
    region_mndwi = mndwi[region_mask]
    region_mndwi_valid = region_mndwi[np.isfinite(region_mndwi)]
    
    if len(region_mndwi_valid) > 10:
        mndwi_mean = np.mean(region_mndwi_valid)
        mndwi_std = np.std(region_mndwi_valid)
        
        mndwi_score_val = np.clip((mndwi_mean - 0.1) / 0.5, 0, 1)
        stability_score = np.clip(1.0 - mndwi_std / 0.15, 0, 1)
        spectral_score = 0.5 * mndwi_score_val + 0.5 * stability_score
        
        details['mndwi_mean'] = mndwi_mean
        details['mndwi_std'] = mndwi_std
    else:
        spectral_score = 0.3
        details['mndwi_mean'] = -1
        details['mndwi_std'] = -1
    
    details['spectral_score'] = spectral_score
    scores.append(spectral_score)
    weights.append(0.15)
    
    # ======== 4. Boundary touch ========
    bbox = region.bbox
    touches_top = (bbox[0] <= 2)
    touches_bottom = (bbox[2] >= h - 2)
    touches_left = (bbox[1] <= 2)
    touches_right = (bbox[3] >= w - 2)
    
    n_borders_touched = sum([touches_top, touches_bottom, touches_left, touches_right])
    
    if n_borders_touched >= 2:
        boundary_score = 1.0
    elif n_borders_touched == 1:
        boundary_score = 0.6
    else:
        boundary_score = 0.0
    
    details['borders_touched'] = n_borders_touched
    details['boundary_score'] = boundary_score
    scores.append(boundary_score)
    weights.append(0.15)
    
    # ======== 5. Sinuosity ========
    try:
        if skeleton is not None and np.sum(skeleton) > 10:
            skel_coords = np.argwhere(skeleton)
            
            if len(skel_coords) > 2:
                from scipy.spatial.distance import cdist
                if len(skel_coords) > 500:
                    idx = np.random.choice(len(skel_coords), 500, replace=False)
                    sample_coords = skel_coords[idx]
                else:
                    sample_coords = skel_coords
                
                dists = cdist(sample_coords, sample_coords)
                max_idx = np.unravel_index(np.argmax(dists), dists.shape)
                endpoint_dist = dists[max_idx[0], max_idx[1]]
                
                path_length = np.sum(skeleton)
                sinuosity = path_length / (endpoint_dist + 1e-6)
                sinuosity_score = np.clip((sinuosity - 1.0) / 1.5, 0, 1)
            else:
                sinuosity = 1.0
                sinuosity_score = 0.0
        else:
            sinuosity = 1.0
            sinuosity_score = 0.0
        
        details['sinuosity'] = sinuosity
        details['sinuosity_score'] = sinuosity_score
        scores.append(sinuosity_score)
        weights.append(0.1)
    except Exception:
        details['sinuosity'] = 1.0
        details['sinuosity_score'] = 0.0
        scores.append(0.0)
        weights.append(0.1)
    
    # ======== 6. Aspect ratio (auxiliary) ========
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    aspect_ratio = major_axis / (minor_axis + 1e-6)
    
    ar_score = np.clip((aspect_ratio - 2.0) / 6.0, 0, 1)
    
    details['aspect_ratio'] = aspect_ratio
    details['ar_score'] = ar_score
    scores.append(ar_score)
    weights.append(0.15)
    
    # ======== Weighted total score ========
    weights = np.array(weights)
    scores = np.array(scores)
    total_score = np.sum(scores * weights) / np.sum(weights)
    
    details['total_score'] = total_score
    
    return total_score, details


# ========================================
# ========== Main Program ==========
# ========================================

print("=" * 60)
print("Urban Flood Detection System - Remote Sensing Based (Dynamic River Recognition v7)")
print("=" * 60)

# ---------- Step 1. Read imagery ----------
print("\n[Step 1] Reading remote sensing imagery...")
with rasterio.open(INPUT_TIF) as src:
    n_bands = src.count
    print(f"  Total bands: {n_bands}")
    
    aerosol = src.read(BAND_MAP['aerosol']).astype(np.float32) / 10000.0
    blue    = src.read(BAND_MAP['blue']).astype(np.float32) / 10000.0
    green   = src.read(BAND_MAP['green']).astype(np.float32) / 10000.0
    red     = src.read(BAND_MAP['red']).astype(np.float32) / 10000.0
    nir     = src.read(BAND_MAP['nir']).astype(np.float32) / 10000.0
    swir1   = src.read(BAND_MAP['swir1']).astype(np.float32) / 10000.0
    swir2   = src.read(BAND_MAP['swir2']).astype(np.float32) / 10000.0
    
    has_scl = False
    if n_bands >= 15:
        scl = src.read(BAND_MAP['scl']).astype(np.uint8)
        scl_unique = np.unique(scl)
        if np.max(scl_unique) <= 11:
            has_scl = True
            print(f"  SCL band loaded, unique values: {scl_unique}")
        else:
            print(f"  Warning: Band {BAND_MAP['scl']} value range abnormal, skipping SCL cloud mask")
    else:
        print(f"  Warning: Image has only {n_bands} bands, no SCL band")
    
    profile = src.profile

pixel_area_m2 = get_pixel_area_m2(profile)
print(f"  Pixel area: {pixel_area_m2:.1f} m2")

print("\nBand value ranges:")
for name, band in zip(["aerosol","blue","green","red","nir","swir1","swir2"],
                      [aerosol, blue, green, red, nir, swir1, swir2]):
    print(f"  {name:8s} min/max: {np.nanmin(band):.4f} / {np.nanmax(band):.4f}")

# ---------- Step 2. Cloud removal ----------
print("\n[Step 2] Atmospheric correction and cloud removal...")
if has_scl:
    cloud_mask = np.isin(scl, [3, 8, 9, 10])
    print(f"  Using SCL cloud mask, cloud pixel ratio: {np.sum(cloud_mask)/cloud_mask.size:.2%}")
else:
    cloud_brightness = (blue + green + red + nir) / 4.0
    mndwi_temp = (green - swir1) / (green + swir1 + 1e-10)
    cloud_mask = (cloud_brightness > 0.35) & (mndwi_temp < 0.0) & (nir > 0.25)
    print(f"  Using spectral cloud detection (fallback), cloud pixel ratio: {np.sum(cloud_mask)/cloud_mask.size:.2%}")

valid_mask = (
    (blue  > 0) & (blue  < 1.0) &
    (green > 0) & (green < 1.0) &
    (red   > 0) & (red   < 1.0) &
    (nir   > 0) & (nir   < 1.5) &
    (swir1 > 0) & (swir1 < 1.0) &
    (swir2 > 0) & (swir2 < 1.0) &
    (~cloud_mask)
)
valid_ratio = np.sum(valid_mask) / valid_mask.size
print(f"  Valid pixel ratio: {valid_ratio:.2%}")
if valid_ratio < 0.1:
    print("  *** Warning: Less than 10% valid pixels ***")

# ---------- Step 3. NDVI & FVC & Adaptive L ----------
print("\n[Step 3] Computing NDVI and FVC, generating adaptive L parameter...")
ndvi = (nir - red) / (nir + red + 1e-10)
ndvi = np.where(valid_mask, ndvi, np.nan)

ndvi_values = ndvi[np.isfinite(ndvi)]
if len(ndvi_values) > 0:
    ndvi_min = np.percentile(ndvi_values, 5)
    ndvi_max = np.percentile(ndvi_values, 95)
    print(f"  NDVI_min (5%): {ndvi_min:.4f}, NDVI_max (95%): {ndvi_max:.4f}")
    fvc = (ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-12)
    fvc = np.clip(fvc, 0, 1)
else:
    fvc = np.zeros_like(ndvi)

L = np.zeros_like(fvc)
L = np.where(fvc > 0.75, 0.1, L)
L = np.where((fvc >= 0.15) & (fvc <= 0.75), 0.5, L)
L = np.where(fvc < 0.15, 0.9, L)

L_values = L[np.isfinite(L)]
if len(L_values) > 0:
    print(f"  L parameter distribution: min={np.min(L_values):.2f}, max={np.max(L_values):.2f}, mean={np.mean(L_values):.2f}")

# ---------- Step 4. Compute remote sensing indices ----------
print("\n[Step 4] Computing remote sensing indices...")

mndwi = (green - swir1) / (green + swir1 + 1e-10)
mndwi = np.where(valid_mask, mndwi, np.nan)

savi = ((nir - red) / (nir + red + L + 1e-12)) * (1 + L)

nbi_raw = np.where(valid_mask, (red * swir1) / (nir + 1e-12), np.nan)
nbi_min, nbi_max = np.nanmin(nbi_raw), np.nanmax(nbi_raw)
nbi = (nbi_raw - nbi_min) / (nbi_max - nbi_min + 1e-12)

d2 = (nir - red)**2
sigma = np.median(d2[d2>0])*2.0 if np.any(d2>0) else 1.0
kndvi = np.tanh((d2/(2*sigma+1e-12))**2)

IBI_nbi4 = np.where(valid_mask, 
                    np.clip((2 * nbi - ((1-L)*kndvi + mndwi + L*savi) / 2) / 
                           ((2 * nbi + ((1-L)*kndvi + mndwi + L*savi) / 2) + 1e-12), 
                           -1, 1), 
                    np.nan)

print(f"  MNDWI range: [{np.nanmin(mndwi):.3f}, {np.nanmax(mndwi):.3f}]")
print(f"  IBI_NBI4 range: [{np.nanmin(IBI_nbi4):.3f}, {np.nanmax(IBI_nbi4):.3f}]")

# ---------- Step 5. Water extraction (MNDWI + Bimodality test + Otsu/fallback threshold) ----------
print("\n[Step 5] Water extraction...")

mndwi_values = mndwi[np.isfinite(mndwi)]

if len(mndwi_values) > 0:
    # ====== Bimodality test ======
    print("  Performing MNDWI histogram bimodality test...")
    is_bimodal, ashman_d, bimodal_details = check_bimodality(mndwi_values)
    
    print(f"    Ashman's D = {ashman_d:.3f} (threshold: {BIMODAL_D_THRESHOLD:.1f})")
    print(f"    Detected histogram peaks: {bimodal_details.get('n_peaks', '?')}")
    if 'valley_ratio' in bimodal_details:
        print(f"    Valley depth ratio: {bimodal_details['valley_ratio']:.3f} (threshold: 0.75)")
    if 'mu1' in bimodal_details:
        print(f"    Component 1: mu={bimodal_details['mu1']:.3f}, sigma={bimodal_details['sigma1']:.3f}, n={bimodal_details['n1']}")
        print(f"    Component 2: mu={bimodal_details['mu2']:.3f}, sigma={bimodal_details['sigma2']:.3f}, n={bimodal_details['n2']}")
    if 'group_ratio' in bimodal_details:
        print(f"    Group ratio: {bimodal_details['group_ratio']:.1f}:1")
    if 'override' in bimodal_details:
        print(f"    Override rule: {bimodal_details['override']}")
    
    if is_bimodal:
        water_threshold = threshold_otsu(mndwi_values)
        print(f"  * Bimodality test passed (D={ashman_d:.3f} > {BIMODAL_D_THRESHOLD})")
        print(f"    Otsu threshold: {water_threshold:.3f}")
        
        if water_threshold < 0.1:
            print(f"    Warning: Otsu threshold too low ({water_threshold:.3f}), below MNDWI water safety lower bound 0.1")
            print(f"    Correction: Values in 0~0.1 are mostly shadow/bare soil interference, corrected to 0.1")
            water_threshold = 0.1
        
        if water_threshold > 0.8:
            print(f"    Warning: Otsu threshold too high ({water_threshold:.3f}), corrected to 0.3")
            water_threshold = 0.3
        
        print(f"    Final water threshold: {water_threshold:.3f}")
    else:
        water_threshold = BIMODAL_FALLBACK_THRESHOLD
        print(f"  * Bimodality test failed (D={ashman_d:.3f} <= {BIMODAL_D_THRESHOLD})")
        print(f"    MNDWI histogram does not satisfy bimodal distribution -> very few water pixels in scene")
        print(f"    Falling back to conservative threshold: MNDWI > {water_threshold}")
        
        otsu_ref = threshold_otsu(mndwi_values)
        print(f"    (Reference: Otsu threshold is {otsu_ref:.3f}, but not adopted due to non-bimodality)")
else:
    water_threshold = BIMODAL_FALLBACK_THRESHOLD
    print(f"  No valid MNDWI values, using default threshold: {water_threshold}")

visible_brightness = (red + green + blue) / 3.0

# === Water ratio sanity check ===
water_preliminary = (mndwi > water_threshold) & valid_mask
water_ratio_preliminary = np.sum(water_preliminary) / (np.sum(valid_mask) + 1)
print(f"  Preliminary water pixel ratio: {water_ratio_preliminary:.2%} (threshold={water_threshold:.3f})")

WATER_RATIO_WARN = 0.20
if water_ratio_preliminary > WATER_RATIO_WARN and water_threshold < 0.1:
    mndwi_positive = mndwi_values[mndwi_values > 0]
    if len(mndwi_positive) > 50:
        conservative_threshold = np.percentile(mndwi_positive, 25)
        conservative_threshold = max(conservative_threshold, 0.1)
    else:
        conservative_threshold = 0.1
    
    print(f"  Warning: Water ratio abnormally high ({water_ratio_preliminary:.1%} > {WATER_RATIO_WARN:.0%})")
    print(f"    Current threshold ({water_threshold:.3f}) may be too low, tightening to: {conservative_threshold:.3f}")
    water_threshold = conservative_threshold
    
    water_preliminary = (mndwi > water_threshold) & valid_mask
    water_ratio_revised = np.sum(water_preliminary) / (np.sum(valid_mask) + 1)
    print(f"    Revised water ratio: {water_ratio_revised:.2%}")

# === Dynamic brightness threshold (two-step method) ===
high_conf_water = (mndwi > max(water_threshold + 0.1, 0.1)) & valid_mask
high_conf_count = np.sum(high_conf_water)

if high_conf_count > 50:
    water_brightness_values = visible_brightness[high_conf_water]
    water_bright_mean = np.mean(water_brightness_values)
    water_bright_std = np.std(water_brightness_values)
    water_bright_p95 = np.percentile(water_brightness_values, 95)
    
    dynamic_brightness_threshold = water_bright_p95 + 2.0 * water_bright_std
    dynamic_brightness_threshold = np.clip(dynamic_brightness_threshold, 0.1, 0.8)
    
    print(f"  High-confidence water samples: {high_conf_count}")
    print(f"  Water brightness distribution: mean={water_bright_mean:.3f}, std={water_bright_std:.3f}, P95={water_bright_p95:.3f}")
    print(f"  Dynamic brightness threshold: {dynamic_brightness_threshold:.3f}")
else:
    dynamic_brightness_threshold = WATER_BRIGHTNESS_FALLBACK
    print(f"  Insufficient high-confidence water ({high_conf_count}), using fallback brightness threshold: {WATER_BRIGHTNESS_FALLBACK:.3f}")

brightness_mask = visible_brightness < dynamic_brightness_threshold

water_mask = (mndwi > water_threshold) & valid_mask & brightness_mask
print(f"  MNDWI water pixels: {np.sum(water_mask)}")

water_ratio_final = np.sum(water_mask) / (np.sum(valid_mask) + 1)
print(f"  Final water ratio: {water_ratio_final:.2%}")

water_mask = closing(water_mask, disk(WATER_CLOSING_RADIUS))

save_tif(water_mask.astype(np.float32), profile, 
         os.path.join(OUTPUT_DIR, '01_Water_Mask_Initial.tif'))

water_labels, num_water = ndi_label(water_mask)
print(f"  Detected water connected components: {num_water}")

if USE_TEXTURE_FEATURES and num_water > 0:
    print("  Computing water texture features...")
    water_texture_std = calculate_local_std(mndwi, window_size=TEXTURE_WINDOW_SIZE)
    water_texture_std = np.nan_to_num(water_texture_std, nan=0)
    save_tif(water_texture_std, profile, 
             os.path.join(OUTPUT_DIR, '01b_Water_Texture_Std.tif'))
    print(f"    Water texture std range: [{np.min(water_texture_std):.4f}, {np.max(water_texture_std):.4f}]")
else:
    water_texture_std = None

# ---------- Step 6. River and lake exclusion (dynamic river scoring + auto threshold) ----------
print("\n[Step 6] River and lake exclusion...")

image_shape = water_mask.shape
regions = regionprops(water_labels)
print(f"  Total water connected components: {len(regions)}")

# ====== First pass: compute features for all water bodies ======
print("  First pass: computing features for all water bodies...")
region_data = []

for region in regions:
    area = region.area
    region_mask = (water_labels == region.label)
    
    info = {
        'label': region.label,
        'area': area,
        'region': region,
        'region_mask': region_mask,
        'river_score': None,
        'river_details': None,
        'compactness': None,
        'aspect_ratio': None,
    }
    
    if area < FLOOD_MIN_SIZE:
        info['category'] = 'small'
        region_data.append(info)
        continue
    
    if area > WATER_MAX_SIZE:
        info['category'] = 'lake_large'
        region_data.append(info)
        continue
    
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    perimeter = region.perimeter if region.perimeter > 0 else 1.0
    info['aspect_ratio'] = major_axis / (minor_axis + 1e-6)
    info['compactness'] = 4 * np.pi * area / (perimeter ** 2)
    
    if area >= RIVER_MIN_AREA:
        river_score, river_details = calculate_river_score(
            region, region_mask, mndwi, water_texture_std,
            image_shape, water_mask
        )
        info['river_score'] = river_score
        info['river_details'] = river_details
    
    info['category'] = 'pending'
    region_data.append(info)

# ====== Auto-determine compactness threshold ======
lake_candidate_compactness = [
    d['compactness'] for d in region_data
    if d['compactness'] is not None 
    and d['area'] > LAKE_MIN_AREA 
    and d['aspect_ratio'] is not None 
    and d['aspect_ratio'] < 2.5
]

if COMPACTNESS_AUTO and len(lake_candidate_compactness) >= 3:
    comp_arr = np.array(lake_candidate_compactness)
    comp_sorted = np.sort(comp_arr)
    
    gaps = np.diff(comp_sorted)
    
    if len(gaps) > 0:
        max_gap_idx = np.argmax(gaps)
        max_gap = gaps[max_gap_idx]
        gap_threshold = (comp_sorted[max_gap_idx] + comp_sorted[max_gap_idx + 1]) / 2.0
        
        mean_gap = np.mean(gaps)
        gap_significant = max_gap > mean_gap * 2.0 and max_gap > 0.05
        
        if gap_significant and 0.2 < gap_threshold < 0.8:
            COMPACTNESS_THRESHOLD = gap_threshold
            print(f"  Compactness natural break threshold: {COMPACTNESS_THRESHOLD:.3f}")
            print(f"    Gap: {comp_sorted[max_gap_idx]:.3f} | gap={max_gap:.3f} | {comp_sorted[max_gap_idx+1]:.3f}")
        else:
            COMPACTNESS_THRESHOLD = np.median(comp_arr)
            COMPACTNESS_THRESHOLD = np.clip(COMPACTNESS_THRESHOLD, 0.25, 0.6)
            print(f"  Gap not significant, using median threshold: {COMPACTNESS_THRESHOLD:.3f}")
    else:
        COMPACTNESS_THRESHOLD = 0.3
        print(f"  Insufficient compactness data, using default: {COMPACTNESS_THRESHOLD:.3f}")
    
    print(f"  Lake candidate compactness statistics (n={len(comp_arr)}):")
    print(f"    min={np.min(comp_arr):.3f}, median={np.median(comp_arr):.3f}, max={np.max(comp_arr):.3f}")

else:
    COMPACTNESS_THRESHOLD = 0.3
    if not COMPACTNESS_AUTO:
        print(f"  Using fixed compactness threshold: {COMPACTNESS_THRESHOLD:.3f}")
    else:
        print(f"  Insufficient lake candidates ({len(lake_candidate_compactness)}), using default: {COMPACTNESS_THRESHOLD:.3f}")

# ====== Auto-determine river score threshold (v7 fix: small sample protection) ======
all_river_scores = [d['river_score'] for d in region_data 
                    if d['river_score'] is not None]

if RIVER_SCORE_AUTO_THRESHOLD and len(all_river_scores) >= 5:
    all_river_scores_arr = np.array(all_river_scores)
    all_river_scores_sorted = np.sort(all_river_scores_arr)
    n_scores = len(all_river_scores_sorted)
    
    # ===== v7 fix: small sample protection branch =====
    # When score count < RIVER_SMALL_SAMPLE_N, IQR and natural break methods lack
    # statistical significance and can produce unreliable thresholds.
    # Use median minus offset as threshold instead.
    # Rationale: median is the most robust central tendency estimator for small samples;
    #            subtracting a small offset (0.02) prevents boundary rivers from being missed.
    if n_scores < RIVER_SMALL_SAMPLE_N:
        median_score = np.median(all_river_scores_arr)
        RIVER_SCORE_THRESHOLD = median_score - 0.02
        RIVER_SCORE_THRESHOLD = np.clip(RIVER_SCORE_THRESHOLD, 0.35, 0.70)
        
        print(f"  * Small sample protection (n={n_scores} < {RIVER_SMALL_SAMPLE_N}):")
        print(f"    Score median: {median_score:.3f}")
        print(f"    Threshold = median - 0.02 = {median_score - 0.02:.3f}, clipped: {RIVER_SCORE_THRESHOLD:.3f}")
        print(f"    (IQR/break methods unstable for small samples, using median method)")
    else:
        # ===== Normal sample size: natural break + IQR fallback =====
        gaps = np.diff(all_river_scores_sorted)
        
        search_start = n_scores // 2
        
        if len(gaps) > search_start:
            upper_gaps = gaps[search_start:]
            max_gap_idx_in_upper = np.argmax(upper_gaps)
            max_gap_idx = search_start + max_gap_idx_in_upper
            max_gap = upper_gaps[max_gap_idx_in_upper]
            
            gap_threshold = (all_river_scores_sorted[max_gap_idx] + 
                            all_river_scores_sorted[max_gap_idx + 1]) / 2.0
            
            upper_gap_mean = np.mean(upper_gaps)
            upper_gap_std = np.std(upper_gaps) if len(upper_gaps) > 1 else 0.01
            gap_significance = (max_gap - upper_gap_mean) / (upper_gap_std + 1e-6)
            
            print(f"  Natural break analysis:")
            print(f"    Max gap: {max_gap:.4f} (between score #{max_gap_idx+1}/{n_scores})")
            print(f"    Gap sides: {all_river_scores_sorted[max_gap_idx]:.3f} | gap={max_gap:.4f} | {all_river_scores_sorted[max_gap_idx+1]:.3f}")
            print(f"    Gap significance: {gap_significance:.2f} (upper half mean={upper_gap_mean:.4f}, std={upper_gap_std:.4f})")
            
            # v7 fix: relaxed significance condition from > 1.0 to >= 1.0
            if gap_significance >= 1.0 and max_gap > 0.05:
                RIVER_SCORE_THRESHOLD = gap_threshold
                RIVER_SCORE_THRESHOLD = np.clip(RIVER_SCORE_THRESHOLD, 0.35, 0.85)
                print(f"    -> Gap significant, using natural break threshold: {RIVER_SCORE_THRESHOLD:.3f}")
            else:
                Q1 = np.percentile(all_river_scores_arr, 25)
                Q3 = np.percentile(all_river_scores_arr, 75)
                IQR = Q3 - Q1
                if IQR > 0.01:
                    RIVER_SCORE_THRESHOLD = Q3 + 1.5 * IQR
                    RIVER_SCORE_THRESHOLD = np.clip(RIVER_SCORE_THRESHOLD, 0.35, 0.85)
                    print(f"    -> Gap not significant, fallback IQR(k=1.0): Q3={Q3:.3f}+IQR={IQR:.3f} -> {RIVER_SCORE_THRESHOLD:.3f}")
                else:
                    RIVER_SCORE_THRESHOLD = RIVER_SCORE_THRESHOLD_MANUAL
                    print(f"    -> IQR too small, using manual threshold: {RIVER_SCORE_THRESHOLD:.3f}")
        else:
            RIVER_SCORE_THRESHOLD = RIVER_SCORE_THRESHOLD_MANUAL
            print(f"  Insufficient scores for gap analysis, using manual threshold: {RIVER_SCORE_THRESHOLD:.3f}")
    
    Q1 = np.percentile(all_river_scores_arr, 25)
    Q3 = np.percentile(all_river_scores_arr, 75)
    print(f"\n  River score distribution statistics (n={n_scores}):")
    print(f"    min={np.min(all_river_scores_arr):.3f}, "
          f"Q25={Q1:.3f}, "
          f"median={np.median(all_river_scores_arr):.3f}, "
          f"Q75={Q3:.3f}, "
          f"max={np.max(all_river_scores_arr):.3f}")
    
    n_above = np.sum(all_river_scores_arr >= RIVER_SCORE_THRESHOLD)
    n_below = np.sum(all_river_scores_arr < RIVER_SCORE_THRESHOLD)
    print(f"    Above threshold ({RIVER_SCORE_THRESHOLD:.3f}, river): {n_above}, below (non-river): {n_below}")
    
else:
    RIVER_SCORE_THRESHOLD = RIVER_SCORE_THRESHOLD_MANUAL
    if len(all_river_scores) < 5:
        print(f"  Insufficient scores ({len(all_river_scores)}), using manual threshold: {RIVER_SCORE_THRESHOLD:.3f}")
    else:
        print(f"  Using manual river score threshold: {RIVER_SCORE_THRESHOLD:.3f}")

# ====== Second pass: classify based on thresholds ======
print(f"\n  Second pass: classifying (river threshold={RIVER_SCORE_THRESHOLD:.3f})...")

non_river_lake_mask = np.zeros_like(water_labels, dtype=bool)
excluded_count = 0
small_area_kept = 0
exclusion_reasons = {
    'large_area': 0, 
    'river_score': 0,
    'lake_compact': 0, 
    'high_mndwi': 0, 
    'perimeter_area': 0, 
    'convexity': 0, 
    'texture': 0
}
river_score_log = []

for info in region_data:
    area = info['area']
    region_mask = info['region_mask']
    region = info['region']
    
    if info['category'] == 'small':
        non_river_lake_mask |= region_mask
        small_area_kept += 1
        continue
    
    if info['category'] == 'lake_large':
        excluded_count += 1
        exclusion_reasons['large_area'] += 1
        continue
    
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    perimeter = region.perimeter if region.perimeter > 0 else 1.0
    aspect_ratio = info['aspect_ratio'] if info['aspect_ratio'] is not None else major_axis / (minor_axis + 1e-6)
    compactness = info['compactness'] if info['compactness'] is not None else 4 * np.pi * area / (perimeter ** 2)
    
    if info['river_score'] is not None:
        river_score = info['river_score']
        river_details = info['river_details']
        
        river_details['label'] = info['label']
        river_details['area'] = area
        river_score_log.append(river_details)
        
        # v7 fix: changed from > to >= to ensure boundary values (e.g., exactly median) are also excluded
        if river_score >= RIVER_SCORE_THRESHOLD:
            excluded_count += 1
            exclusion_reasons['river_score'] += 1
            print(f"    Excluding river: label={info['label']}, area={area}, "
                  f"score={river_score:.3f}, AR={aspect_ratio:.1f}, "
                  f"skeleton={river_details.get('skeleton_ratio',0):.2f}, "
                  f"width_cv={river_details.get('width_cv',0):.2f}, "
                  f"borders={river_details.get('borders_touched',0)}")
            continue
    
    if area > LAKE_MIN_AREA and aspect_ratio < 2.5 and compactness > COMPACTNESS_THRESHOLD:
        excluded_count += 1
        exclusion_reasons['lake_compact'] += 1
        continue
    
    region_mndwi = mndwi[region_mask]
    region_mndwi_valid = region_mndwi[np.isfinite(region_mndwi)]
    if len(region_mndwi_valid) > 0:
        mndwi_mean = np.mean(region_mndwi_valid)
        mndwi_std = np.std(region_mndwi_valid)
        if mndwi_mean > 0.5 and mndwi_std < 0.1 and area > 1000:
            excluded_count += 1
            exclusion_reasons['high_mndwi'] += 1
            continue
    
    perimeter_area_ratio = perimeter / (np.sqrt(area) + 1e-6)
    if perimeter_area_ratio < 8.0 and area > 2000:
        excluded_count += 1
        exclusion_reasons['perimeter_area'] += 1
        continue
    
    convex_area = region.convex_area
    convexity = area / (convex_area + 1e-6)
    if convexity > 0.95 and area > 1500 and area < WATER_MAX_SIZE:
        excluded_count += 1
        exclusion_reasons['convexity'] += 1
        continue
    
    if USE_TEXTURE_FEATURES and water_texture_std is not None:
        region_texture = water_texture_std[region_mask]
        texture_mean = np.mean(region_texture)
        if texture_mean < 0.05 and area > 2000:
            excluded_count += 1
            exclusion_reasons['texture'] += 1
            continue
    
    non_river_lake_mask |= region_mask

river_lake_mask = water_mask.astype(bool) & ~non_river_lake_mask

save_tif(non_river_lake_mask.astype(np.float32), profile, 
         os.path.join(OUTPUT_DIR, '02_Non_River_Lake_Water.tif'))
save_tif(river_lake_mask.astype(np.float32), profile, 
         os.path.join(OUTPUT_DIR, '03_River_Lake_Excluded.tif'))

print(f"\n  Small water bodies retained (<{FLOOD_MIN_SIZE} pixels): {small_area_kept}")
print(f"  Excluded river/lake components: {excluded_count}")
print(f"  Exclusion reasons:")
for reason, count in exclusion_reasons.items():
    print(f"    - {reason}: {count}")
print(f"  Retained suspected flood water components: {num_water - excluded_count}")

if len(river_score_log) > 0:
    print(f"\n  River score details (top 10 highest):")
    sorted_log = sorted(river_score_log, key=lambda x: x['total_score'], reverse=True)
    for i, d in enumerate(sorted_log[:10]):
        print(f"    #{i+1} label={d['label']}, area={d['area']}, "
              f"total={d['total_score']:.3f}, "
              f"skel={d.get('skeleton_score',0):.2f}, "
              f"width={d.get('width_score',0):.2f}, "
              f"spectral={d.get('spectral_score',0):.2f}, "
              f"boundary={d.get('boundary_score',0):.2f}, "
              f"sinuosity={d.get('sinuosity_score',0):.2f}, "
              f"AR={d.get('ar_score',0):.2f}")

# ---------- Step 7. Impervious surface extraction ----------
print("\n[Step 7] Impervious surface extraction...")

ibi_values = IBI_nbi4[np.isfinite(IBI_nbi4)]
if len(ibi_values) > 0:
    print("  Performing IBI_NBI4 histogram bimodality test...")
    ibi_is_bimodal, ibi_ashman_d, ibi_bimodal_details = check_bimodality(ibi_values)
    
    print(f"    Ashman's D = {ibi_ashman_d:.3f} (threshold: {BIMODAL_D_THRESHOLD:.1f})")
    print(f"    Detected histogram peaks: {ibi_bimodal_details.get('n_peaks', '?')}")
    if 'valley_ratio' in ibi_bimodal_details:
        print(f"    Valley depth ratio: {ibi_bimodal_details['valley_ratio']:.3f}")
    if 'mu1' in ibi_bimodal_details:
        print(f"    Component 1: mu={ibi_bimodal_details['mu1']:.3f}, sigma={ibi_bimodal_details['sigma1']:.3f}, n={ibi_bimodal_details['n1']}")
        print(f"    Component 2: mu={ibi_bimodal_details['mu2']:.3f}, sigma={ibi_bimodal_details['sigma2']:.3f}, n={ibi_bimodal_details['n2']}")
    if 'override' in ibi_bimodal_details:
        print(f"    Override rule: {ibi_bimodal_details['override']}")
    
    if ibi_is_bimodal:
        impervious_threshold = threshold_otsu(ibi_values)
        print(f"  * IBI bimodality test passed (D={ibi_ashman_d:.3f} > {BIMODAL_D_THRESHOLD})")
        print(f"    Otsu threshold: {impervious_threshold:.3f}")
        
        if impervious_threshold < 0.1:
            print(f"    Warning: Otsu threshold too low ({impervious_threshold:.3f}), below IBI impervious surface safety lower bound 0.1")
            print(f"    Correction: Values in 0~0.1 are mostly bare soil/mixed pixel interference, corrected to 0.1")
            impervious_threshold = 0.1
        
        print(f"    Final impervious surface threshold: {impervious_threshold:.3f}")
    else:
        impervious_threshold = 0.1
        print(f"  * IBI bimodality test failed (D={ibi_ashman_d:.3f} <= {BIMODAL_D_THRESHOLD})")
        print(f"    IBI_NBI4 histogram does not satisfy bimodal distribution, Otsu threshold unreliable")
        print(f"    Falling back to conservative threshold: IBI_NBI4 > {impervious_threshold}")
        
        otsu_ref = threshold_otsu(ibi_values)
        print(f"    (Reference: Otsu threshold is {otsu_ref:.3f}, but not adopted due to non-bimodality)")
    
    impervious_mask = (IBI_nbi4 > impervious_threshold) & valid_mask
else:
    impervious_mask = np.zeros_like(IBI_nbi4, dtype=bool)

impervious_mask = closing(impervious_mask, disk(BUILDING_CLOSING_RADIUS))
impervious_mask = remove_small_objects(impervious_mask, min_size=BUILDING_MIN_SIZE)

if USE_TEXTURE_FEATURES:
    print("  Computing building texture features...")
    building_texture_std = calculate_local_std(nir, window_size=TEXTURE_WINDOW_SIZE)
    building_texture_std = np.nan_to_num(building_texture_std, nan=0)
    
    if np.sum(impervious_mask) > 0:
        high_texture_mask = building_texture_std > np.percentile(
            building_texture_std[impervious_mask], 30)
        impervious_mask_refined = impervious_mask & high_texture_mask
    else:
        impervious_mask_refined = impervious_mask
    
    save_tif(building_texture_std, profile, 
             os.path.join(OUTPUT_DIR, '04b_Building_Texture_Std.tif'))
    print(f"    Pixels before texture filtering: {np.sum(impervious_mask)}")
    print(f"    Pixels after texture filtering: {np.sum(impervious_mask_refined)}")

save_tif(impervious_mask.astype(np.float32), profile, 
         os.path.join(OUTPUT_DIR, '04_Impervious_Surface.tif'))

impervious_labels, num_impervious = label(impervious_mask, return_num=True)
print(f"  Impervious surface connected components: {num_impervious}")

# ---------- Step 8. Flood area determination ----------
print("\n[Step 8] Flood area determination...")

water_labels_filtered, _ = ndi_label(non_river_lake_mask)
flooded_water_mask = np.zeros_like(water_labels_filtered, dtype=bool)
flood_confidence_map = np.zeros_like(mndwi, dtype=np.float32)
dist_to_impervious = distance_transform_edt(~impervious_mask)

flood_count = 0
small_area_flood = 0
small_area_rejected = 0
large_area_flood = 0
extra_recalled = 0

for region in regionprops(water_labels_filtered):
    region_mask = (water_labels_filtered == region.label)
    area = region.area
    
    min_dist = np.min(dist_to_impervious[region_mask])
    
    if area < FLOOD_MIN_SIZE:
        if min_dist < BUILDING_WATER_DISTANCE * 1.5:
            flooded_water_mask |= region_mask
            flood_confidence_map[region_mask] = 0.5
            small_area_flood += 1
        else:
            small_area_rejected += 1
        continue
    
    if min_dist < BUILDING_WATER_DISTANCE:
        flooded_water_mask |= region_mask
        flood_confidence_map[region_mask] = 0.8
        large_area_flood += 1
    else:
        if min_dist < BUILDING_WATER_DISTANCE * 2:
            region_ndvi = ndvi[region_mask]
            region_ndvi_valid = region_ndvi[np.isfinite(region_ndvi)]
            region_mndwi = mndwi[region_mask]
            region_mndwi_valid = region_mndwi[np.isfinite(region_mndwi)]
            
            ndvi_ok = (len(region_ndvi_valid) > 0 and np.mean(region_ndvi_valid) < 0.15)
            mndwi_ok = (len(region_mndwi_valid) > 0 and np.mean(region_mndwi_valid) > water_threshold)
            area_ok = (area >= FLOOD_MIN_SIZE and area < 5000)
            
            if ndvi_ok and mndwi_ok and area_ok:
                flooded_water_mask |= region_mask
                flood_confidence_map[region_mask] = 0.5
                extra_recalled += 1

flood_count = small_area_flood + large_area_flood + extra_recalled
print(f"  Small area classified as flood: {small_area_flood} (rejected: {small_area_rejected})")
print(f"  Distance-based flood: {large_area_flood}")
print(f"  Auxiliary recall: {extra_recalled}")
print(f"  Total flood water bodies: {flood_count}")

final_flood_mask = (flooded_water_mask) & (~river_lake_mask)

save_tif(final_flood_mask.astype(np.float32), profile, 
         os.path.join(OUTPUT_DIR, '06_Final_Flood_Mask.tif'))
save_tif(flood_confidence_map, profile, 
         os.path.join(OUTPUT_DIR, '06b_Flood_Confidence.tif'))

flood_area_pixels = np.sum(final_flood_mask)
flood_area_m2 = flood_area_pixels * pixel_area_m2
print(f"  Flood area pixels: {flood_area_pixels}")
print(f"  Flood area: {flood_area_m2:.0f} m2 ({flood_area_m2/1e6:.4f} km2)")
print(f"  Flood area ratio: {flood_area_pixels/np.sum(valid_mask):.2%}")

# ---------- Step 8b. River threshold sensitivity analysis ----------
if len(all_river_scores) >= 5:
    print("\n[Step 8b] River threshold sensitivity analysis...")
    print("  Threshold | Excluded rivers | Retained water | Flood area change")
    print("  " + "-" * 55)
    
    all_scores_arr = np.array(all_river_scores)
    test_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    for t in test_thresholds:
        n_excluded = np.sum(all_scores_arr >= t)  # v7 fix: consistent with exclusion logic, using >=
        n_kept = len(all_scores_arr) - n_excluded
        marker = " <-- current" if abs(t - RIVER_SCORE_THRESHOLD) < 0.025 else ""
        print(f"  {t:.2f}  |    {n_excluded:4d}    |    {n_kept:4d}    |{marker}")
    
    print(f"\n  Current threshold: {RIVER_SCORE_THRESHOLD:.3f} ({'small sample median method' if n_scores < RIVER_SMALL_SAMPLE_N else 'natural break/IQR auto' if RIVER_SCORE_AUTO_THRESHOLD else 'manual setting'})")

# ---------- Step 9. Visualization ----------
print("\n[Step 9] Generating visualization...")

rgb = np.stack([normalize_for_vis(red), 
                normalize_for_vis(green), 
                normalize_for_vis(blue)], axis=-1)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, 
                       wspace=0.02, hspace=0.10,
                       width_ratios=[1, 1, 1], 
                       height_ratios=[1, 1, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(rgb)
ax1.set_title('(a) Original RGB Image', fontsize=11)
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(water_mask, cmap='Blues')
ax2.set_title('(b) Initial Water Extraction', fontsize=11)
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(river_lake_mask, cmap='Blues')
ax3.set_title('(c) Excluded Natural Water Bodies', fontsize=11)
ax3.axis('off')

ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(impervious_mask, cmap='Greys')
ax4.set_title('(d) Impervious Surface', fontsize=11)
ax4.axis('off')

ax5 = fig.add_subplot(gs[1, 1])
ax5.imshow(flooded_water_mask, cmap='Reds')
ax5.set_title('(e) Detected Flood Area', fontsize=11)
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 2])
ax6.imshow(rgb)
ax6.imshow(final_flood_mask, cmap='Reds', alpha=0.65)
ax6.set_title('(f) Flood Overlay on RGB', fontsize=11)
ax6.axis('off')

ax7 = fig.add_subplot(gs[2, 0])
conf_display = np.where(final_flood_mask, flood_confidence_map, np.nan)
im7 = ax7.imshow(conf_display, cmap='RdYlGn', vmin=0, vmax=1)
ax7.set_title('(g) Flood Confidence', fontsize=11)
ax7.axis('off')
plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

ax8 = fig.add_subplot(gs[2, 1])
mndwi_display = np.where(valid_mask, mndwi, np.nan)
im8 = ax8.imshow(mndwi_display, cmap='RdBu', vmin=-0.5, vmax=0.5)
ax8.set_title('(h) MNDWI', fontsize=11)
ax8.axis('off')
plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

ax9 = fig.add_subplot(gs[2, 2])
ndvi_display = np.where(valid_mask, ndvi, np.nan)
im9 = ax9.imshow(ndvi_display, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
ax9.set_title('(i) NDVI', fontsize=11)
ax9.axis('off')
plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

output_fig_path = os.path.join(OUTPUT_DIR, '08_Flood_Detection_Results.png')
plt.savefig(output_fig_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"  Result figure saved: {output_fig_path}")

print("\n" + "=" * 60)
print("Processing complete! All results saved to:", OUTPUT_DIR)
print("=" * 60)

plt.show()
