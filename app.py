import streamlit as st
import os
import requests
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from rfdetr import RFDETRSmall
import supervision as sv
from PIL import Image
from fpdf import FPDF

# Streamlit Page Setup
st.set_page_config(page_title="PPE Safety Dashboard", layout="wide")
st.title("🛡️ AI Visual Inspection Safety Dashboard")
st.subheader("Upload Image, Detect Violations, Generate Report & Email Directly")

# 1. Paste your GitHub Release Direct Link here
DOWNLOAD_URL = "https://github.com/huzaifa-32/PPEs-safety-dashboard/releases/download/v1.0.0/checkpoint_best_total.pth"
MODEL_PATH = "checkpoint_best_total.pth"

# Streaming Model Downloader & Loader
@st.cache_resource
def load_production_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading 127MB Model weights from GitHub Releases... Please wait."):
            try:
                response = requests.get(DOWNLOAD_URL, stream=True)
                if response.status_code == 200:
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    st.error("Model download link invalid hai! Please check your Release URL.")
                    return None
            except Exception as e:
                st.error(f"Download Error: {e}")
                return None
                
    try:
        # Correct official method for RF-DETR weight loading
        model = RFDETRSmall.from_checkpoint(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model initialization error: {e}")
        return None

model = load_production_model()

# Helper Function: PDF Report Generator
def generate_pdf_report(detections_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(200, 10, txt="AI PPE Safety Inspection Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=12)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(200, 10, txt=f"Inspection Date & Time: {current_time}", ln=True)
    pdf.cell(200, 10, txt=f"Total Safety Gear / Violations Detected: {detections_count}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", style='B', size=12)
    if detections_count > 0:
        pdf.cell(200, 10, txt="Status: Action Required - Safety elements detected/missing.", ln=True)
    else:
        pdf.cell(200, 10, txt="Status: Clear - No specific safety flags identified.", ln=True)
        
    report_name = "safety_inspection_report.pdf"
    pdf.output(report_name)
    return report_name

# Helper Function: Automated Email Sender via Streamlit Secrets
def send_email_with_attachment(recipient_email, attachment_path):
    # Fetch credentials securely from Streamlit Cloud Secrets
    try:
        sender_email = st.secrets["YOUR_GMAIL"]
        app_password = st.secrets["YOUR_APP_PASSWORD"]
    except Exception:
        st.error("Secrets configuration missing! Streamlit Dashboard standard me settings add karein.")
        return False

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "⚠️ Automated PPE Safety Compliance Report"
    
    body = f"Greetings,\n\nPlease find attached the automated AI Visual Safety Inspection Report generated on {datetime.datetime.now().strftime('%Y-%m-%d')}."
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={attachment_path}")
            msg.attach(part)
            
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.close()
        return True
    except Exception as e:
        st.error(f"Email Transmit Error: {e}")
        return False

# 3. Main Dashboard UI
if model:
    uploaded_file = st.file_uploader("Choose an image for inspection...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Uploaded Image", use_container_width=True)
            
        with col2:
            with st.spinner("Running Real-Time AI Inference..."):
                # Inference using RF-DETR
                detections = model.predict(image, threshold=0.25)
                
                # Annotate using Supervision
                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
                st.image(annotated_frame, caption="AI Detection Output", use_container_width=True)
        
        # Calculate statistics Safely
        total_detections = len(detections) if hasattr(detections, '__len__') else 0
        st.success(f"Analysis Complete! Total items detected: {total_detections}")
        
        # Report & Email Section
        st.markdown("---")
        st.subheader("📩 Export & Email Results")
        
        email_input = st.text_input("Enter Recipient Email Address:", placeholder="example@domain.com")
        
        if st.button("Generate Report & Send Email"):
            if email_input.strip() == "":
                st.warning("Please enter a valid email address first.")
            else:
                with st.spinner("Generating PDF and sending email..."):
                    # 1. Make PDF
                    pdf_file = generate_pdf_report(total_detections)
                    
                    # 2. Send Email
                    success = send_email_with_attachment(email_input, pdf_file)
                    
                    if success:
                        st.balloons()
                        st.success(f"🚀 Report successfully emailed to {email_input}!")
                        
                        # Cleanup local PDF from server space after sending
                        if os.path.exists(pdf_file):
                            os.remove(pdf_file)
