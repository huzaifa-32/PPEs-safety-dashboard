import streamlit as st
import os
from rfdetr import RFDETRSmall
import supervision as sv
from PIL import Image

st.set_page_config(page_title="AI Safety Dashboard", layout="wide")
st.title("AI Visual Inspection Safety Dashboard")

# 1. Model Loading (The simplified, correct way)
@st.cache_resource
def get_model():
    # PATH where file SHOULD be
    file_path = "https://huggingface.co/HuzaifaUrRehman/checkpoint_best_total/resolve/main/checkpoint_best_total.pth"
    
    # Check if file exists
    if not os.path.exists(file_path):
        st.error(f"File {file_path} nahi mili. App folder me upload karein.")
        return None
        
    try:
        # Load directly using the library method
        # Adjust 'num_classes' if your model was trained on a different count
        model = RFDETRSmall(num_classes=8)
        model.load_weights(file_path) 
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model load nahi ho saka: {e}")
        return None

model = get_model()

# 2. UI
uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        # Predict using the model
        detections = model.predict(image)
        
        # Annotate
        annotator = sv.BoxAnnotator()
        annotated_frame = annotator.annotate(scene=image.copy(), detections=detections)
        st.image(annotated_frame, caption="Detected", use_container_width=True)
