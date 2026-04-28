import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import pandas as pd
import io

# Page configuration
st.set_page_config(
    page_title="Blood Cell Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 10px;
    }
    .cell-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function for colors
WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}
COLOR_RBC = (0, 0, 255)       # Red in BGR
COLOR_PLATELETS = (0, 200, 0) # Green in BGR
COLOR_WBC = (255, 80, 0)     # Blue in BGR

def color_for(name: str) -> tuple[int, int, int]:
    if name == "RBC":
        return COLOR_RBC
    if name == "Platelets":
        return COLOR_PLATELETS
    if name in WBC_SUBTYPES:
        return COLOR_WBC
    return (200, 200, 200)

def annotate_image(img, results):
    # Ultralytics results objects contain everything we need
    # We'll use the original image (numpy array) and draw on it
    annotated_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft, bt = 0.5, 1, 2
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    names = results[0].names
    
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        name = names[int(c)]
        col = color_for(name)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        
        # Draw rectangle
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), col, bt)
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(name, font, fs, ft)
        ly = y1 - 2
        if ly - th - 2 < 0:
            ly = y1 + th + 4
        cv2.rectangle(annotated_img, (x1, ly - th - 2), (x1 + tw + 2, ly + 1), col, -1)
        
        # Draw label text
        cv2.putText(annotated_img, name, (x1 + 1, ly - 1), font, fs,
                    (255, 255, 255), ft, cv2.LINE_AA)
                    
    return annotated_img

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def main():
    st.title("🔬 Blood Cell Detector")
    st.markdown("Automated detection and classification of cells in peripheral blood smear images.")
    st.divider()

    # Sidebar
    st.sidebar.header("Configuration")
    model_path = "blood_detector_model.pt"
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.7, 0.05)
    
    st.sidebar.info("This tool uses a YOLOv8 model fine-tuned for blood cell detection.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a blood smear image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        # Run inference
        if st.button("Run Detection"):
            with st.spinner("Analyzing image..."):
                model = load_model(model_path)
                results = model.predict(
                    source=img_array, # Ultralytics can handle RGB numpy arrays
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=640,
                    device='cpu',
                    verbose=False
                )
                
                # Annotate
                annotated_bgr = annotate_image(img_bgr, results)
                # Convert BGR back to RGB for Streamlit
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Detected Cells")
                    st.image(annotated_rgb, use_container_width=True)
                
                # Analysis
                st.divider()
                st.subheader("Analysis Summary")
                
                # Count classes
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                counts = Counter(names[int(c)] for c in classes)
                
                # Metrics
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Total Cells", len(classes))
                m_col2.metric("RBCs", counts.get("RBC", 0))
                m_col3.metric("Platelets", counts.get("Platelets", 0))
                
                wbc_count = sum(counts.get(sub, 0) for sub in WBC_SUBTYPES)
                m_col4.metric("Total WBCs", wbc_count)
                
                # Detailed Breakdown
                st.write("### Detailed WBC Differential")
                wbc_data = {sub: counts.get(sub, 0) for sub in WBC_SUBTYPES}
                df_wbc = pd.DataFrame(list(wbc_data.items()), columns=["Cell Type", "Count"])
                st.bar_chart(df_wbc.set_index("Cell Type"))
                
                # Download button
                img_to_download = Image.fromarray(annotated_rgb)
                buf = io.BytesIO()
                img_to_download.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Annotated Image",
                    data=byte_im,
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/jpeg"
                )
    else:
        st.info("Please upload an image to start the analysis.")
        
        # Show example images
        st.write("### Try a sample image")
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        # This is a bit tricky if we don't have the files accessible by URL, 
        # but we can show names at least.
        st.write("You can find sample images in the `test_images/` folder of this repository.")

if __name__ == "__main__":
    main()
