"""
app.py
======
Brain MRI & EEG Analysis for Brain Tumor — OpenCV + ML + Streamlit
Academic prototype for CIE-2 Digital Image and Video Processing demonstration.

Faculty Demonstration Flow:
1. Select Analysis Mode (MRI Analysis / EEG Analysis).
2. Upload file or click "Load Sample Data".
3. Visualize stepwise DIP pipeline (Original -> Preprocessed -> Segmentation -> Morphology -> Detected Candidate).
4. Review quantitative metrics, extracted features, and explainable Decision Tree classification.
5. Review academic disclaimer & viva theory guide.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

import mri_eeg_core as core

# =============================================================================
# PATH RESOLUTION & COMPATIBILITY HELPERS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_asset_path(relative_path):
    """
    Resolves relative file paths safely across diverse working directories.
    Checks:
    1. Relative to the script's directory (BASE_DIR)
    2. Relative to the parent of BASE_DIR
    3. Directly against the current working directory
    """
    candidates = [
        os.path.join(BASE_DIR, relative_path),
        os.path.join(os.path.dirname(BASE_DIR), relative_path),
        os.path.abspath(relative_path),
        relative_path
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return relative_path

def render_image(img, clamp=False):
    """Safely renders image across Streamlit versions with full container width."""
    try:
        st.image(img, width="stretch", clamp=clamp)
    except Exception:
        st.image(img, use_container_width=True, clamp=clamp)

def render_dataframe(df):
    """Safely renders dataframe across Streamlit versions with full container width."""
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.dataframe(df, use_container_width=True)

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Brain MRI & EEG Analysis | CIE-2 Prototype",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic Presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 15px;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 10px 15px;
        border-radius: 6px;
        border-left: 4px solid #F59E0B;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown('<div class="main-title">🧠 Brain MRI & EEG Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">OpenCV-Based Academic Prototype — CIE-2 Digital Image and Video Processing</div>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>Academic Disclaimer:</strong> This application is a teaching and research prototype developed strictly for educational demonstration.
    It is <strong>NOT</strong> a medical diagnostic device. EEG signal characteristics are supplementary and cannot independently diagnose brain tumors.
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.title("Navigation & Settings")
app_mode = st.sidebar.radio(
    "Select Analysis Module:",
    ["MRI Image Analysis", "EEG Signal Analysis", "Faculty Viva & Theory Guide"]
)

# Load Pretrained ML Model if present
model_payload = core.load_mri_model(get_asset_path("models/model.pkl"))

# =============================================================================
# MODULE 1: MRI IMAGE ANALYSIS
# =============================================================================
if app_mode == "MRI Image Analysis":
    st.subheader("1. Brain MRI Image Processing Pipeline")
    st.write("Upload an axial brain MRI scan (JPG, JPEG, PNG, TIFF) to execute the stepwise OpenCV tumor candidate detection pipeline.")

    # Sidebar parameters for faculty interactive demo
    st.sidebar.markdown("---")
    st.sidebar.subheader("MRI Pipeline Parameters")
    thresh_method_choice = st.sidebar.selectbox(
        "Thresholding Method:",
        ["Otsu Automatic", "Manual Binary Threshold"]
    )
    thresh_method = "otsu" if thresh_method_choice == "Otsu Automatic" else "manual"

    manual_thresh_val = 160
    if thresh_method == "manual":
        manual_thresh_val = st.sidebar.slider("Intensity Threshold Value:", 80, 240, 160, step=5)

    min_area_param = st.sidebar.slider("Minimum Candidate Area (pixels):", 100, 1500, 300, step=50)
    kernel_size_param = st.sidebar.selectbox("Morphological Kernel Size:", [3, 5, 7], index=1)

    # Sample data loader shortcuts
    st.sidebar.markdown("---")
    st.sidebar.write("💡 **Quick Demo Samples:**")
    col_s1, col_s2 = st.sidebar.columns(2)
    load_sample_tumor = col_s1.button("Tumor MRI")
    load_sample_normal = col_s2.button("Normal MRI")

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Scan:",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
        help="Supports standard formats: PNG, JPG, JPEG, TIFF"
    )

    # Determine input image
    image_bgr = None
    source_label = ""

    tumor_sample_path = get_asset_path("sample_data/sample_mri.png")
    normal_sample_path = get_asset_path("sample_data/sample_mri_normal.png")

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Error decoding image file: {str(e)}")
    elif load_sample_tumor:
        if os.path.exists(tumor_sample_path):
            image_bgr = cv2.imread(tumor_sample_path)
            source_label = "Built-in Sample: Brain MRI with Suspicious Tumor Lesion"
        else:
            st.error(f"Sample file '{tumor_sample_path}' not found.")
    elif load_sample_normal:
        if os.path.exists(normal_sample_path):
            image_bgr = cv2.imread(normal_sample_path)
            source_label = "Built-in Sample: Normal Brain MRI Slice"
        else:
            st.error(f"Sample file '{normal_sample_path}' not found.")
    else:
        # Default starting state: load tumor sample so the app is immediately alive and ready for faculty review
        if os.path.exists(tumor_sample_path):
            image_bgr = cv2.imread(tumor_sample_path)
            source_label = "Demo Ready: Loaded Sample Tumor MRI (Upload your own scan above or try Normal MRI)"

    if image_bgr is not None:
        st.info(f"**Active Source:** {source_label}")

        # Execute Pipeline
        with st.spinner("Processing MRI through OpenCV pipeline..."):
            pipeline_res = core.analyze_mri_pipeline(
                image_bgr,
                threshold_method=thresh_method,
                threshold_val=manual_thresh_val,
                min_area=min_area_param
            )

        # 1. Stepwise Visual Display (5 Columns)
        st.markdown("### Stepwise OpenCV Pipeline Stages")
        cols = st.columns(5)

        with cols[0]:
            st.caption("1. Original Resized (256x256)")
            render_image(cv2.cvtColor(pipeline_res["original_resized"], cv2.COLOR_BGR2RGB))

        with cols[1]:
            st.caption("2. Preprocessed (CLAHE)")
            render_image(pipeline_res["enhanced"], clamp=True)

        with cols[2]:
            st.caption(f"3. Binary Mask (Thresh={pipeline_res['used_thresh']})")
            render_image(pipeline_res["mask"], clamp=True)

        with cols[3]:
            st.caption(f"4. Morphological Result")
            render_image(pipeline_res["morph"], clamp=True)

        with cols[4]:
            st.caption("5. Detected Candidate Overlay")
            render_image(cv2.cvtColor(pipeline_res["overlay"], cv2.COLOR_BGR2RGB))

        st.markdown("---")

        # 2. Candidate Detection Findings & Classification
        has_candidate = pipeline_res["features"]["has_candidate"]
        area = pipeline_res["area"]
        area_pct = pipeline_res["area_pct"]
        bbox = pipeline_res["bbox"]

        st.markdown("### Analysis Findings & Classification")

        # Call ML Model Prediction
        ml_prediction, ml_confidence = core.predict_mri_tumor(pipeline_res["features"], model_payload)

        col_res1, col_res2 = st.columns([1.5, 1])

        with col_res1:
            if has_candidate and area >= min_area_param:
                st.warning(f"🚨 **Analysis Result: Suspicious Region / Tumor Candidate Detected**")
                st.write(
                    f"A hyperintense connected region occupying **{area:.1f} pixels** "
                    f"(**{area_pct:.2f}%** of total brain scan area) was localized."
                )
            else:
                st.success("✅ **Analysis Result: No Significant Suspicious Region Detected**")
                st.write("No connected hyperintense region met the candidate area threshold criteria.")

            st.caption("🔍 *Terminology Notice: We refer to detected regions as 'Suspicious Region / Tumor Candidate' for academic rigor.*")

        with col_res2:
            st.markdown("#### Optional ML Classifier Result")
            if model_payload is not None:
                if ml_prediction == "Suspicious / Tumor Candidate":
                    st.error(f"**ML Classification:** {ml_prediction}")
                else:
                    st.success(f"**ML Classification:** {ml_prediction}")
                st.write(f"Model: **Explainable Decision Tree**")
                st.write(f"Confidence: **{ml_confidence}%**")
                st.caption(f"Model Validation Accuracy: {model_payload.get('accuracy', 1.0) * 100:.1f}%")
            else:
                st.info("Standalone OpenCV Mode active (No ML model loaded).")

        # 3. Quantitative Metrics
        st.markdown("### Extracted Morphological & Intensity Features")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Candidate Area", f"{area:.1f} px", f"{area_pct:.2f}% of image")
        m_col2.metric("Mean Intensity", f"{pipeline_res['features']['mean_intensity']:.1f} / 255")
        m_col3.metric("Solidity (Area Ratio)", f"{pipeline_res['features']['contour_area_ratio']:.3f}")
        aspect_r = pipeline_res['features']['aspect_ratio']
        m_col4.metric("Aspect Ratio (W/H)", f"{aspect_r:.2f}")

        with st.expander("📊 View Complete Feature Table"):
            bbox_str = f"x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}" if bbox else "None"
            feat_df = pd.DataFrame([
                {"Feature": "Mean Intensity", "Value": f"{pipeline_res['features']['mean_intensity']:.2f}", "Description": "Average brightness inside candidate (0-255)"},
                {"Feature": "Standard Deviation", "Value": f"{pipeline_res['features']['std_intensity']:.2f}", "Description": "Pixel intensity variation inside candidate"},
                {"Feature": "Area (pixels)", "Value": f"{area:.1f} px", "Description": "Total physical pixel area enclosed by contour"},
                {"Feature": "Area Percentage", "Value": f"{area_pct:.2f}%", "Description": "Fraction of 256x256 image area"},
                {"Feature": "Bounding Box", "Value": bbox_str, "Description": "Axis-aligned bounding rectangle coordinates"},
                {"Feature": "Solidity", "Value": f"{pipeline_res['features']['contour_area_ratio']:.3f}", "Description": "Contour Area / Bounding Box Area"},
                {"Feature": "Aspect Ratio", "Value": f"{aspect_r:.2f}", "Description": "Bounding box width divided by height"}
            ])
            # Ensure Value column is consistently typed as string to avoid PyArrow serialization issues
            feat_df["Value"] = feat_df["Value"].astype(str)
            render_dataframe(feat_df)

# =============================================================================
# MODULE 2: EEG SIGNAL ANALYSIS
# =============================================================================
elif app_mode == "EEG Signal Analysis":
    st.subheader("2. Neurological EEG Signal Analysis Module")
    st.write(
        "Upload an EEG CSV recording to inspect signal characteristics, time-domain features, "
        "and frequency spectra. This serves as a supplementary neurological signal-processing component."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("EEG Signal Parameters")
    sampling_rate = st.sidebar.number_input("Sampling Frequency (Hz):", min_value=64, max_value=2048, value=256, step=64)
    norm_choice = st.sidebar.selectbox("Signal Normalization:", ["None (Raw uV)", "Z-Score Standardization", "Min-Max Normalization"])

    load_sample_eeg = st.sidebar.button("Load Built-in Sample EEG CSV")

    eeg_file = st.file_uploader(
        "Upload EEG CSV File:",
        type=["csv"],
        help="Upload CSV containing multi-channel or single-channel EEG samples"
    )

    eeg_df = None
    eeg_source = ""
    sample_eeg_path = get_asset_path("sample_data/sample_eeg.csv")

    if eeg_file is not None:
        try:
            eeg_df, num_channels = core.load_and_clean_eeg(eeg_file)
            eeg_source = f"Uploaded CSV: {eeg_file.name}"
        except Exception as e:
            st.error(f"Error loading EEG CSV: {str(e)}")
    elif load_sample_eeg or os.path.exists(sample_eeg_path):
        try:
            eeg_df, num_channels = core.load_and_clean_eeg(sample_eeg_path)
            eeg_source = "Built-in Sample EEG (5s multi-channel: Fp1, Fp2, C3, C4, Cz, Pz)"
        except Exception as e:
            st.error(f"Error loading sample EEG: {str(e)}")

    if eeg_df is not None:
        st.info(f"**Active Source:** {eeg_source} | **Samples:** {len(eeg_df)} rows")

        # Select channel to plot
        # Exclude 'Time_sec' or index from channel dropdown if possible
        candidate_channels = [c for c in num_channels if "time" not in c.lower()]
        if not candidate_channels:
            candidate_channels = num_channels

        selected_channel = st.selectbox("Select EEG Electrode / Channel to Analyze:", candidate_channels, index=0)
        raw_signal = eeg_df[selected_channel].values

        # Apply Normalization if requested
        display_signal = raw_signal.copy()
        y_label = "Amplitude (uV)"
        if norm_choice == "Z-Score Standardization":
            s_std = np.std(raw_signal)
            if s_std > 0:
                display_signal = (raw_signal - np.mean(raw_signal)) / s_std
            y_label = "Standardized Amplitude (Z-score)"
        elif norm_choice == "Min-Max Normalization":
            s_ptp = np.ptp(raw_signal)
            if s_ptp > 0:
                display_signal = (raw_signal - np.min(raw_signal)) / s_ptp
            y_label = "Normalized Amplitude [0, 1]"

        # Compute Features
        eeg_stats = core.calculate_eeg_statistics(raw_signal)
        eeg_fft = core.calculate_eeg_fft(raw_signal, fs=sampling_rate)

        # Plot Waveform
        st.markdown(f"### EEG Waveform — Channel `{selected_channel}`")

        # Time vector
        if "Time_sec" in eeg_df.columns:
            time_vector = eeg_df["Time_sec"].values
            time_label = "Time (seconds)"
        else:
            time_vector = np.arange(len(raw_signal)) / float(sampling_rate)
            time_label = "Time (seconds, estimated from fs)"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1.2]})
        plt.subplots_adjust(hspace=0.35)

        # Time-domain waveform
        ax1.plot(time_vector, display_signal, color="#1D4ED8", linewidth=1.1)
        ax1.set_title(f"EEG Time-Domain Signal ({selected_channel})", fontsize=11, fontweight="bold")
        ax1.set_xlabel(time_label, fontsize=9)
        ax1.set_ylabel(y_label, fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Frequency-domain spectrum
        ax2.plot(eeg_fft["freqs"], eeg_fft["fft_magnitude"], color="#B91C1C", linewidth=1.1)
        ax2.set_xlim(0, min(50, sampling_rate / 2))
        ax2.axvline(eeg_fft["dominant_freq"], color="#047857", linestyle="--", label=f"Dominant: {eeg_fft['dominant_freq']} Hz")
        ax2.set_title("Frequency Spectrum (NumPy FFT Magnitude)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)", fontsize=9)
        ax2.set_ylabel("Magnitude", fontsize=9)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)
        plt.close(fig)

        # Metrics display
        st.markdown("### Extracted Numerical Signal Features")
        e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)
        e_col1.metric("Mean Voltage", f"{eeg_stats['mean']:.2f} uV")
        e_col2.metric("Standard Deviation", f"{eeg_stats['std']:.2f} uV")
        e_col3.metric("RMS Power", f"{eeg_stats['rms']:.2f} uV")
        e_col4.metric("Peak-to-Peak", f"{eeg_stats['peak_to_peak']:.2f} uV")
        e_col5.metric("Dominant Frequency", f"{eeg_fft['dominant_freq']:.1f} Hz")

        st.markdown("### Clinical Band Power Distribution")
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        b_col1.metric("Delta (0.5 - 4 Hz)", f"{eeg_fft['delta_power']:.2f}")
        b_col2.metric("Theta (4 - 8 Hz)", f"{eeg_fft['theta_power']:.2f}")
        b_col3.metric("Alpha (8 - 13 Hz)", f"{eeg_fft['alpha_power']:.2f}")
        b_col4.metric("Beta (13 - 30 Hz)", f"{eeg_fft['beta_power']:.2f}")

        st.success("EEG feature analysis completed successfully.")

        st.markdown("""
        > [!NOTE]
        > **Neurological Interpretation:**
        > EEG analysis provides signal characteristics and can be included as an additional neurological signal-analysis component.
        > It **does not** independently confirm the presence of a brain tumor. MRI image processing remains the primary tumor-analysis component.
        """)

# =============================================================================
# MODULE 3: FACULTY VIVA & THEORY GUIDE
# =============================================================================
elif app_mode == "Faculty Viva & Theory Guide":
    st.subheader("🎓 Faculty Viva & Technical Theory Guide")
    st.write("This section details the line-by-line explanation of every Digital Image Processing (DIP) and Machine Learning function used in this project.")

    st.markdown("### 1. OpenCV Functions Explained (Line-by-Line)")
    for func, desc in core.EXPLANATION_GUIDE.items():
        with st.expander(f"📌 `{func}`"):
            st.write(desc)

    st.markdown("### 2. Complete DIP Processing Sequence")
    st.markdown("""
    ```mermaid
    graph LR
        A["1. Read Image (cv2.imread)"] --> B["2. Resize to 256x256 (cv2.resize)"]
        B --> C["3. Grayscale (cv2.cvtColor)"]
        C --> D["4. Gaussian Blur (cv2.GaussianBlur)"]
        D --> E["5. CLAHE Enhancement (cv2.createCLAHE)"]
        E --> F["6. Otsu / Binary Thresholding (cv2.threshold)"]
        F --> G["7. Morphological Opening & Closing (cv2.morphologyEx)"]
        G --> H["8. Contour Detection (cv2.findContours)"]
        H --> I["9. Area & BBox Filtering (cv2.contourArea)"]
        I --> J["10. Overlay & ML Decision Tree"]
    ```
    """)

    st.markdown("### 3. Faculty Frequently Asked Questions (Viva Questions)")
    with st.expander("Q1: Why do we convert the image to Grayscale before processing?"):
        st.write("""
        **Answer:** MRI images represent proton density or relaxation times ($T_1, T_2$), which are inherently scalar intensity maps.
        Converting from 3-channel RGB/BGR to 1-channel Grayscale reduces computational complexity by 66% while preserving all contrast and luminance information required for segmentation.
        """)

    with st.expander("Q2: Why use CLAHE instead of standard Global Histogram Equalization?"):
        st.write("""
        **Answer:** Standard Global Histogram Equalization operates on the entire image histogram, which frequently over-amplifies noise in homogeneous regions and washes out subtle brain parenchyma borders.
        **CLAHE** (Contrast Limited Adaptive Histogram Equalization) divides the image into contextual tiles (8x8) and clips the contrast histogram at a predefined limit (`clipLimit=2.0`), enhancing local tumor contrast without blowing out the background.
        """)

    with st.expander("Q3: What are Morphological Opening and Closing, and why are both used?"):
        st.write("""
        **Answer:**
        - **Opening ($\text{Erode} \rightarrow \text{Dilate}$):** Removes small, isolated, high-intensity noise dots and severs thin accidental bridges between the tumor candidate and skull boundary.
        - **Closing ($\text{Dilate} \rightarrow \text{Erode}$):** Fills internal dark pinholes and crevices within the segmented tumor mask, creating a contiguous solid candidate.
        """)

    with st.expander("Q4: Why choose a Decision Tree instead of a Deep Learning CNN?"):
        st.write("""
        **Answer:**
        - **Explainability:** Decision Trees formulate transparent, interpretable if-else rules (e.g. `mean_intensity > 162`). Every classification can be audited by a human clinician or faculty examiner.
        - **Lightweight Execution:** Runs instantaneously on any standard CPU without requiring GPU acceleration or massive gigabyte weight files.
        - **CIE-2 Syllabus Alignment:** Direct grounding in feature extraction from fundamental image descriptors (Mean, Area, Aspect Ratio).
        """)

    with st.expander("Q5: What is the clinical relationship between EEG and MRI in Brain Tumors?"):
        st.write("""
        **Answer:**
        MRI provides spatial, anatomical visualization of structural lesions (mass effect, edema, hyperintensity).
        EEG measures macroscopic cortical electrical oscillations. While localized tumors can induce focal delta/theta slowing or epileptiform spikes due to cortical compression, EEG **cannot** independently confirm or delineate a brain tumor. Thus, MRI is the primary diagnostic modality, and EEG is supplementary signal data.
        """)
