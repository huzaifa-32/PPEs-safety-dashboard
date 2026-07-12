import streamlit as st
import os
import torch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from fpdf import FPDF
from PIL import Image
import supervision as sv
from collections import Counter
from rfdetr import RFDETRSmall

# --- Page Configuration (Premium Theme) ---
st.set_page_config(
    page_title="AI Safety Inspector",
    page_icon="🔍",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-title { font-size:40px; font-weight:bold; color:#2E8B57; text-align:center; }
    .sub-title { font-size:18px; color:#646464; text-align:center; margin-bottom:30px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Visual Inspection Safety Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time PPE Detection & Automated Reporting Framework</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
threshold = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=1.00, value=0.25, step=0.05)
receiver_email = st.sidebar.text_input("Receiver Email", value="huzaifar2005@gmail.com")

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_model():
    CLASS_NAMES = ['goggles', 'helmet', 'no-goggles', 'no-helmet', 'no-vest', 'vest', 'class_6', 'class_7']
    model = RFDETRSmall(num_classes=8)
    if os.path.exists("checkpoint_best_total.pth"):
        model.load_state_dict(torch.load("checkpoint_best_total.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, CLASS_NAMES
    else:
        st.error("❌ 'checkpoint_best_total.pth' file directory me nahi mili!")
        return None, CLASS_NAMES

model_inference, CLASS_NAMES = load_model()

# --- Main UI Layout ---
uploaded_file = st.file_uploader("📸 Upload a site image or frame...", type=["jpg", "jpeg", "png"])

if uploaded_file and model_inference:
    # Target columns layout
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Original Frame")
        st.image(image, use_container_width=True)
        
    with st.spinner("⏳ AI Agent analysis run kar raha hai..."):
        # Model Prediction
        detections = model_inference.predict(image, threshold=threshold)
        
        # Annotations
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Save compressed image for PDF
        output_image_path = "detected_result.jpg"
        if annotated_image.mode in ("RGBA", "P"):
            annotated_image = annotated_image.convert("RGB")
        annotated_image.thumbnail((1600, 1600))
        annotated_image.save(output_image_path, "JPEG", quality=75)

    with col2:
        st.subheader("AI Detection Result")
        st.image(annotated_image, use_container_width=True)

    # --- Metrics & Analysis Breakdown ---
    st.write("---")
    st.subheader("📊 Safety Analytics Breakdown")
    
    detected_classes = []
    if detections.class_id is not None:
        detected_classes = [CLASS_NAMES[cid] for cid in detections.class_id if cid < len(CLASS_NAMES)]
    object_counts = Counter(detected_classes)
    
    if len(object_counts) == 0:
        st.warning("⚠️ No safety gear or violations detected at this threshold.")
    else:
        # Show professional metric blocks
        metric_cols = st.columns(len(object_counts))
        for idx, (obj_name, count) in enumerate(object_counts.items()):
            with metric_cols[idx]:
                st.metric(label=obj_name.upper(), value=count)

    # --- PDF & Email Automation Button ---
    st.write("---")
    if st.button("📧 Generate Report & Send Email"):
        with st.spinner("📩 PDF report compile aur email routing ho rahi hai..."):
            # PDF Creation
            pdf_path = "Inference_Detection_Report.pdf"
            pdf = FPDF()
            pdf.add_page()
            
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(46, 139, 87)
            pdf.cell(0, 15, "AI Visual Inspection Safety Report", ln=True, align="C")
            pdf.ln(10)
            
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 10, "1. Detected Safety Objects Breakdown", ln=True)
            pdf.set_font("Helvetica", "", 12)
            
            for obj_name, count in object_counts.items():
                pdf.cell(0, 8, f"- {obj_name.upper()}: {count} instance(s) found", ln=True)
            
            pdf.ln(10)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "2. Visual Evidence Log", ln=True)
            pdf.ln(5)
            pdf.image(output_image_path, x=25, y=None, w=160)
            pdf.output(pdf_path)
            
            # SMTP Email Setup (Fetching from secure secrets)
            try:
                YOUR_GMAIL = st.secrets["YOUR_GMAIL"]
                YOUR_APP_PASSWORD = st.secrets["YOUR_APP_PASSWORD"]
                
                msg = MIMEMultipart()
                msg['From'] = YOUR_GMAIL
                msg['To'] = receiver_email
                msg['Subject'] = "🔍 AI Web App Alert: New Safety Detection Report"
                
                body = "Assalam-o-Alaikum,\n\nPlease find attached the automated site safety compliance report generated via the web dashboard."
                msg.attach(MIMEText(body, 'plain'))
                
                with open(pdf_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename= {pdf_path}")
                msg.attach(part)
                
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(YOUR_GMAIL, YOUR_APP_PASSWORD)
                server.sendmail(YOUR_GMAIL, receiver_email, msg.as_string())
                server.quit()
                
                st.success(f"🚀 BOOM! Report successfully sent to {receiver_email}")
            except Exception as e:
                st.error(f"❌ Connection Timeout/Error: {e}")