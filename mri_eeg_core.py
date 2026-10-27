"""
mri_eeg_core.py
================
Compact, explainable core processing module for Brain MRI & EEG Analysis.
Designed for CIE-2 Digital Image and Video Processing demonstration.

This module provides:
1. MRI OpenCV Pipeline: Preprocessing, CLAHE, Thresholding, Morphology, Contour Detection, Feature Extraction.
2. EEG Signal Processing: Cleaning, Normalization, Time-Domain Feature Extraction, FFT Spectrum Analysis.
3. Model Inference: Explainable Decision Tree classification.
"""

import cv2
import numpy as np
import pandas as pd
import pickle
import os

# =============================================================================
# 1. MRI IMAGE PROCESSING PIPELINE (OpenCV)
# =============================================================================

def preprocess_mri(image_bgr):
    """
    Step 2 & 3: Preprocessing and Contrast Enhancement.
    - cv2.resize: Standardizes dimensions to 256x256 for consistent spatial resolution.
    - cv2.cvtColor: Converts 3-channel BGR image to single-channel Grayscale.
    - cv2.GaussianBlur: Suppresses high-frequency acquisition noise with 5x5 kernel.
    - cv2.createCLAHE: Contrast Limited Adaptive Histogram Equalization to enhance
      local contrast without over-amplifying background noise.
    """
    # Resize to standard 256x256
    resized = cv2.resize(image_bgr, (256, 256))

    # Convert to Grayscale if image is 3-channel
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Gaussian Blur (Noise Reduction)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast Enhancement via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return gray, blurred, enhanced


def segment_mri(enhanced_img, method="otsu", threshold_val=160):
    """
    Step 4: Segmentation of Suspicious High-Intensity Regions.
    - cv2.threshold: Segments bright, hyperintense tumor tissue from surrounding brain matter.
    - Otsu mode: Computes optimal threshold based on intra-class variance of tissue pixels.
    """
    if method == "otsu":
        tissue_pixels = enhanced_img[enhanced_img > 40]
        if len(tissue_pixels) > 0:
            calc_thresh, _ = cv2.threshold(tissue_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Enforce hyperintensity tier floor (~155) so normal gray/white matter is excluded
            calc_thresh = max(int(calc_thresh), 155)
        else:
            calc_thresh = 160
        _, mask = cv2.threshold(enhanced_img, calc_thresh, 255, cv2.THRESH_BINARY)
    else:
        calc_thresh = threshold_val
        _, mask = cv2.threshold(enhanced_img, threshold_val, 255, cv2.THRESH_BINARY)

    return mask, calc_thresh


def apply_morphology(mask, kernel_size=(5, 5)):
    """
    Step 5: Morphological Operations.
    - cv2.getStructuringElement: Generates rectangular structuring element.
    - cv2.morphologyEx(..., cv2.MORPH_OPEN): Opening (Erosion -> Dilation) removes
      small isolated noise pixels and thin false bridges.
    - cv2.morphologyEx(..., cv2.MORPH_CLOSE): Closing (Dilation -> Erosion) fills
      small dark holes inside the detected candidate.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def detect_candidate(morph_mask, original_bgr, min_area=300):
    """
    Step 6 & 7: Contour Detection and Tumor Candidate Visualization.
    - cv2.findContours: Extracts boundaries of all segmented regions.
    - cv2.contourArea: Computes physical pixel area of each candidate.
    - cv2.boundingRect: Computes axis-aligned bounding box (x, y, w, h).
    - cv2.drawContours: Outlines suspicious boundary in Red.
    - cv2.rectangle: Draws bounding box in Green.
    """
    contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = morph_mask.shape
    total_area = float(img_h * img_w)

    valid_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Exclude tiny noise and huge contours (e.g., full outer skull > 40%)
        if min_area <= area <= (0.40 * total_area):
            valid_candidates.append((cnt, area))

    overlay = original_bgr.copy()
    if not valid_candidates:
        return None, overlay, 0.0, 0.0, None

    # Select the largest candidate contour
    valid_candidates.sort(key=lambda x: x[1], reverse=True)
    best_cnt, best_area = valid_candidates[0]

    x, y, w, h = cv2.boundingRect(best_cnt)
    area_pct = (best_area / total_area) * 100.0

    # Draw Red contour boundary and Green bounding box
    cv2.drawContours(overlay, [best_cnt], -1, (0, 0, 255), 2)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(overlay, f"Candidate: {int(best_area)}px ({area_pct:.1f}%)", (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

    return best_cnt, overlay, best_area, area_pct, (x, y, w, h)


def extract_features(gray_img, contour, bbox):
    """
    Extracts numerical morphological and statistical features for ML and analysis:
    - mean_intensity: Average brightness inside the candidate region.
    - std_intensity: Pixel intensity standard deviation inside candidate.
    - area: Total pixel area of candidate.
    - contour_area_ratio (solidity): Contour area / Bounding box area.
    - bbox_width: Width of bounding rectangle.
    - bbox_height: Height of bounding rectangle.
    - aspect_ratio: Width / Height ratio.
    """
    if contour is None or bbox is None:
        return {
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "area": 0.0,
            "contour_area_ratio": 0.0,
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "aspect_ratio": 0.0,
            "has_candidate": False
        }

    x, y, w, h = bbox
    area = float(cv2.contourArea(contour))
    bbox_area = float(w * h)
    solidity = (area / bbox_area) if bbox_area > 0 else 0.0
    aspect_ratio = (float(w) / float(h)) if h > 0 else 0.0

    # Create binary mask of contour to compute interior intensity statistics
    mask = np.zeros(gray_img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val, std_val = cv2.meanStdDev(gray_img, mask=mask)

    return {
        "mean_intensity": round(float(mean_val[0][0]), 2),
        "std_intensity": round(float(std_val[0][0]), 2),
        "area": round(area, 2),
        "contour_area_ratio": round(solidity, 4),
        "bbox_width": float(w),
        "bbox_height": float(h),
        "aspect_ratio": round(aspect_ratio, 4),
        "has_candidate": True
    }


def analyze_mri_pipeline(image_bgr, threshold_method="otsu", threshold_val=160, min_area=300):
    """
    Executes the end-to-end MRI pipeline and returns all intermediate images and metrics.
    """
    # Resize input image to standard 256x256
    resized_bgr = cv2.resize(image_bgr, (256, 256))

    gray, blurred, enhanced = preprocess_mri(resized_bgr)
    mask, used_thresh = segment_mri(enhanced, method=threshold_method, threshold_val=threshold_val)
    morph = apply_morphology(mask)
    cnt, overlay, area, area_pct, bbox = detect_candidate(morph, resized_bgr, min_area=min_area)
    features = extract_features(gray, cnt, bbox)

    return {
        "original_resized": resized_bgr,
        "gray": gray,
        "blurred": blurred,
        "enhanced": enhanced,
        "mask": mask,
        "morph": morph,
        "overlay": overlay,
        "contour": cnt,
        "area": area,
        "area_pct": area_pct,
        "bbox": bbox,
        "features": features,
        "used_thresh": used_thresh
    }


# =============================================================================
# 2. EEG SIGNAL PROCESSING PIPELINE
# =============================================================================

def load_and_clean_eeg(filepath_or_buffer):
    """
    Loads EEG CSV, identifies numeric columns, and handles missing values.
    Returns: DataFrame and list of numeric channel names.
    """
    df = pd.read_csv(filepath_or_buffer)
    # Forward-fill and backward-fill missing values, replace residual NaNs with 0
    df = df.ffill().bfill().fillna(0)

    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numerical signal columns found in EEG CSV file.")

    return df, numeric_cols


def calculate_eeg_statistics(signal):
    """
    Calculates explainable time-domain statistics for an EEG signal:
    - Mean: Baseline offset / DC component.
    - Standard Deviation: Overall amplitude dispersion.
    - Variance: Signal energy dispersion.
    - Minimum: Maximum negative potential deflection.
    - Maximum: Maximum positive potential deflection.
    - Peak-to-Peak: Total voltage range (Max - Min).
    - RMS (Root Mean Square): Effective signal power.
    """
    sig = np.array(signal, dtype=np.float64)
    sig_mean = float(np.mean(sig))
    sig_std = float(np.std(sig))
    sig_var = float(np.var(sig))
    sig_min = float(np.min(sig))
    sig_max = float(np.max(sig))
    sig_ptp = float(sig_max - sig_min)
    sig_rms = float(np.sqrt(np.mean(sig ** 2)))

    return {
        "mean": round(sig_mean, 4),
        "std": round(sig_std, 4),
        "variance": round(sig_var, 4),
        "min": round(sig_min, 4),
        "max": round(sig_max, 4),
        "peak_to_peak": round(sig_ptp, 4),
        "rms": round(sig_rms, 4)
    }


def calculate_eeg_fft(signal, fs=256):
    """
    Calculates frequency-domain properties using NumPy real FFT (np.fft.rfft):
    - Computes frequency spectrum from 0 to Nyquist (fs/2).
    - Identifies dominant frequency in physiological band (0.5 - 35 Hz).
    - Computes average power in standard clinical bands:
      * Delta (0.5 - 4 Hz): Deep sleep / pathological slowing
      * Theta (4 - 8 Hz): Drowsiness / focal lesions
      * Alpha (8 - 13 Hz): Relaxed wakefulness (posterior)
      * Beta (13 - 30 Hz): Active concentration (frontal)
    """
    sig = np.array(signal, dtype=np.float64)
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(sig))

    # Identify dominant frequency excluding 0 Hz DC offset
    valid_idx = np.where((freqs >= 0.5) & (freqs <= 35.0))[0]
    if len(valid_idx) > 0:
        dom_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
        dominant_freq = float(freqs[dom_idx])
    else:
        dominant_freq = 0.0

    # Band powers
    def get_band_power(f_low, f_high):
        idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
        return float(np.mean(fft_vals[idx])) if len(idx) > 0 else 0.0

    delta_pwr = get_band_power(0.5, 4.0)
    theta_pwr = get_band_power(4.0, 8.0)
    alpha_pwr = get_band_power(8.0, 13.0)
    beta_pwr = get_band_power(13.0, 30.0)

    return {
        "freqs": freqs,
        "fft_magnitude": fft_vals,
        "dominant_freq": round(dominant_freq, 2),
        "delta_power": round(delta_pwr, 2),
        "theta_power": round(theta_pwr, 2),
        "alpha_power": round(alpha_pwr, 2),
        "beta_power": round(beta_pwr, 2)
    }


# =============================================================================
# 3. OPTIONAL MACHINE LEARNING INFERENCE
# =============================================================================

def load_mri_model(model_path="models/model.pkl"):
    """
    Loads saved Decision Tree model if available.
    """
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                payload = pickle.load(f)
            return payload
        except Exception:
            return None
    return None


def predict_mri_tumor(features, model_payload):
    """
    Uses trained Decision Tree model to classify extracted MRI features.
    """
    if model_payload is None:
        return "Model not loaded (Running in standalone OpenCV mode)", 0.0

    model = model_payload["model"]
    feature_names = model_payload["feature_names"]
    classes = model_payload["classes"]

    feature_vec = [features.get(f, 0.0) for f in feature_names]
    pred_idx = model.predict([feature_vec])[0]
    pred_label = classes.get(pred_idx, "Unknown")

    proba = 1.0
    if hasattr(model, "predict_proba"):
        proba = float(np.max(model.predict_proba([feature_vec])))

    return pred_label, round(proba * 100, 1)


# =============================================================================
# 4. EXPLAINABILITY CHEATSHEET (For Faculty Viva & Demonstration)
# =============================================================================

EXPLANATION_GUIDE = {
    "cv2.resize": "Resizes images to standard 256x256 dimensions ensuring uniform spatial resolution across datasets.",
    "cv2.cvtColor": "Converts RGB/BGR to single-channel Grayscale for intensity-based analysis, reducing dimensionality by 66%.",
    "cv2.GaussianBlur": "Applies a 2D Gaussian kernel to attenuate high-frequency acquisition noise while preserving primary structural edges.",
    "cv2.createCLAHE": "Contrast Limited Adaptive Histogram Equalization computes localized histograms with clip limit to prevent over-amplification of noise.",
    "cv2.threshold": "Binarizes the enhanced image separating suspicious hyperintense lesions (high intensity) from normal brain tissue.",
    "cv2.morphologyEx": "Executes Opening (erosion followed by dilation to remove small noise) and Closing (dilation followed by erosion to fill interior holes).",
    "cv2.findContours": "Extracts topological contours of connected binary pixel components based on Suzuki's border following algorithm.",
    "cv2.contourArea": "Computes exact physical pixel area using Green's theorem over contour boundary vertices.",
    "cv2.boundingRect": "Calculates minimal axis-aligned bounding rectangle [x, y, width, height] enclosing the candidate region.",
    "Decision Tree": "Transparent, rule-based classifier that makes decisions through simple if-else hierarchical splits without opaque black-box layers."
}
