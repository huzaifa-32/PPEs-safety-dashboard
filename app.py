import streamlit as st
import os
import torch
import smtplib
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from fpdf import FPDF
from PIL import Image
import supervision as sv
from collections import Counter
from rfdetr import RFDETRSmall

# --- Page Setup ---
st.set_page_config(page_title="AI Safety Inspector", page_icon="🔍", layout="wide")
st.markdown('<h1 style="text-align:center; color:#2E8B57;">AI Visual Inspection Safety Dashboard</h1>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration")
threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.45, 0.05)
receiver_email = st.sidebar.text_input("Receiver Email", value="huzaifar2005@gmail.com")

# --- Model Loading Logic ---
@st.cache_resource
def load_model():
    CLASS_NAMES = ['goggles', 'helmet', 'no-goggles', 'no-helmet', 'no-vest', 'vest', 'class_6', 'class_7']
    file_path = "checkpoint_best_total.pth"
    
    # Clean broken files
    if os.path.exists(file_path) and os.path.getsize(file_path) < 1000000:
        os.remove(file_path)

    # Download if not present
    if not os.path.exists(file_path):
        with st.spinner("📥 Model download ho raha hai... Please wait."):
            # ⚠️ Apna Hugging Face ya direct link yahan daalein!
            DOWNLOAD_URL = "hf download hf://HuzaifaUrRehman/checkpoint_best_total/checkpoint_best_total.pth"
            
            try:
                response = requests.get(DOWNLOAD_URL, stream=True)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                return None, f"Download Error: {e}. Check your link."

    # Load Model
    try:
        model = RFDETRSmall(num_classes=8)
        weights = torch.load(file_path, map_location="cpu", weights_only=False)
        
        # Extract state dict safely
        state_dict = weights.get('model', weights) if isinstance(weights, dict) else weights.state_dict()
        
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(state_dict, strict=False)
        elif hasattr(model, 'model'):
            model.model.load_state_dict(state_dict, strict=False)
            
        # Avoid eval() AttributeError
        if hasattr(model, 'eval'):
            model.eval()
            
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return None, f"Model Load Error: {e}. File deleted, please refresh page."

    return model, CLASS_NAMES

# --- Main App Execution ---
try:
    model_inference, status_or_classes = load_model()

    if model_inference is None:
        st.error(status_or_classes) # Shows the exact error on screen
    else:
        CLASS_NAMES = status_or_classes
        uploaded_file = st.file_uploader("📸 Upload site image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file).convert("RGB")
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
                
            with st.spinner("⏳ Analyzing image..."):
                detections = model_inference.predict(image, threshold=threshold)
                
                annotated_image = image.copy()
                annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
                annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)
                
                output_path = "result.jpg"
                annotated_image.thumbnail((1600, 1600))
                annotated_image.save(output_path, "JPEG")

            with col2:
                st.image(annotated_image, caption="AI Detection Result", use_container_width=True)

            # Analytics
            st.write("---")
            st.subheader("📊 Analytics")
            detected_names = [CLASS_NAMES[cid] for cid in detections.class_id if cid < len(CLASS_NAMES)]
            counts = Counter(detected_names)
            
            if not counts:
                st.warning("⚠️ No safety gear detected.")
            else:
                cols = st.columns(len(counts))
                for idx, (name, count) in enumerate(counts.items()):
                    cols[idx].metric(label=name.upper(), value=count)

            # Email Report
            st.write("---")
            if st.button("📧 Send Report"):
                try:
                    # Check if secrets exist safely
                    if "YOUR_GMAIL" not in st.secrets or "YOUR_APP_PASSWORD" not in st.secrets:
                        st.error("⚠️ Streamlit Secrets set nahi hain! Left menu se secrets add karen.")
                    else:
                        with st.spinner("Sending email..."):
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Helvetica", "B", 16)
                            pdf.cell(0, 10, "AI Safety Report", ln=True)
                            for name, count in counts.items():
                                pdf.cell(0, 8, f"- {name.upper()}: {count}", ln=True)
                            pdf.image(output_path, w=160)
                            pdf.output("report.pdf")
                            
                            msg = MIMEMultipart()
                            msg['From'] = st.secrets["YOUR_GMAIL"]
                            msg['To'] = receiver_email
                            msg['Subject'] = "Safety Report"
                            msg.attach(MIMEText("Please find the attached report.", 'plain'))
                            
                            with open("report.pdf", "rb") as f:
                                part = MIMEBase("application", "octet-stream")
                                part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header("Content-Disposition", "attachment; filename=report.pdf")
                            msg.attach(part)
                            
                            server = smtplib.SMTP("smtp.gmail.com", 587)
                            server.starttls()
                            server.login(st.secrets["YOUR_GMAIL"], st.secrets["YOUR_APP_PASSWORD"])
                            server.sendmail(st.secrets["YOUR_GMAIL"], receiver_email, msg.as_string())
                            server.quit()
                            st.success("✅ Email Sent!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

except Exception as critical_error:
    st.error(f"🚨 Application Error: {critical_error}")
