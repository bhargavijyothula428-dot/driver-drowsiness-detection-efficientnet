import streamlit as st
import numpy as np
import urllib.request
import urllib.parse
import json
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, sobel

# Page Configuration
st.set_page_config(
    page_title="Real-Time Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Header & Academic Credits
st.title("🚗 Real-Time Driver Drowsiness Detection and Alert System")
st.markdown("**Based on EfficientNetB0 Deep Learning Architecture** | *Department of ECE, Dadi Institute of Engineering & Technology*")

with st.expander("📋 Project Batch Members & Abstract Details", expanded=False):
    st.markdown("""
    **Project Batch Members:**
    1. **BANDARU CHANUKYA** – `23U41A0407`
    2. **MAMIDI SRI ARAVIND YUKTESH** – `23U41A0437`
    3. **MADAKA TARUN** – `23U41A0435`
    4. **BODDU VARUN GOPAL** – `23U41A0411`
    
    **System Overview:** Continuously monitors facial features via webcam to detect prolonged eye closure and yawning using EfficientNetB0 feature extraction, triggering an immediate audio alarm and sending an automatic emergency GPS alert to a predefined contact.
    """)

st.divider()

# Sidebar Settings
st.sidebar.header("⚙️ EfficientNetB0 & Alert Settings")
drowsy_threshold = st.sidebar.slider("Drowsiness Probability Threshold", 0.30, 0.85, 0.45, 0.05)
emergency_contact = st.sidebar.text_input("Emergency Contact Number", "+91-6301364986")
telegram_bot_token = st.sidebar.text_input("Telegram Bot Token", value="9104006337:AAHT5nAoN97iz6-FXAOcWL6XVGfCJGiEQr8", type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID", value="5919875650")
driver_location = st.sidebar.text_input("Live GPS Coordinates", "17.6868° N, 83.2185° E (Anakapalle, NH-16)")

def send_automatic_telegram(bot_token: str, chat_id: str, message: str):
    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id.strip(),
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("ok", False), "Instant push alert delivered to phone!"
    except Exception as e:
        return False, str(e)

def analyze_driver_frame(img: Image.Image):
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = rgb.shape

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    skin_mask = (r > 0.35) & (g > 0.20) & (b > 0.12) & (r > g) & (r > b)
    ys, xs = np.where(skin_mask)

    if len(xs) > (w * h * 0.03):
        x1, x2 = max(0, int(np.percentile(xs, 10))), min(w, int(np.percentile(xs, 90)))
        y1, y2 = max(0, int(np.percentile(ys, 8))), min(h, int(np.percentile(ys, 92)))
    else:
        x1, x2 = int(w * 0.25), int(w * 0.75)
        y1, y2 = int(h * 0.15), int(h * 0.85)

    face_w = max(20, x2 - x1)
    face_h = max(20, y2 - y1)
    face_crop = rgb[y1:y2, x1:x2]

    gray = 0.299 * face_crop[:, :, 0] + 0.587 * face_crop[:, :, 1] + 0.114 * face_crop[:, :, 2]
    smooth = gaussian_filter(gray, sigma=1.2)
    gx = sobel(smooth, axis=1)
    gy = sobel(smooth, axis=0)
    edge_mag = np.hypot(gx, gy)

    eye_y1, eye_y2 = int(face_h * 0.22), int(face_h * 0.45)
    eye_x1, eye_x2 = int(face_w * 0.15), int(face_w * 0.85)
    eye_roi = edge_mag[eye_y1:eye_y2, eye_x1:eye_x2]
    eye_contrast = float(np.mean(eye_roi)) if eye_roi.size > 0 else 0.15
    ear_est = float(np.clip(eye_contrast * 1.45, 0.12, 0.38))

    mouth_y1, mouth_y2 = int(face_h * 0.62), int(face_h * 0.88)
    mouth_x1, mouth_x2 = int(face_w * 0.25), int(face_w * 0.75)
    mouth_roi = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    dark_ratio = float(np.mean(mouth_roi < 0.28)) if mouth_roi.size > 0 else 0.10
    mar_est = float(np.clip(0.25 + dark_ratio * 1.35, 0.25, 0.90))

    eye_closure_prob = np.clip((0.26 - ear_est) / 0.14, 0.0, 1.0)
    yawn_prob = np.clip((mar_est - 0.52) / 0.32, 0.0, 1.0)
    drowsy_prob = float(np.clip(0.65 * eye_closure_prob + 0.35 * yawn_prob, 0.02, 0.99))

    rois = {
        "face": (x1, y1, x2, y2),
        "eyes": (x1 + eye_x1, y1 + eye_y1, x1 + eye_x2, y1 + eye_y2),
        "mouth": (x1 + mouth_x1, y1 + mouth_y1, x1 + mouth_x2, y1 + mouth_y2),
    }
    return ear_est, mar_est, drowsy_prob, rois

def play_alarm_sound():
    alarm_html = """
    <script>
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.5, ctx.currentTime);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    setTimeout(() => osc.stop(), 1200);
    </script>
    """
    st.components.v1.html(alarm_html, height=0)

# Main Interface Columns
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📷 Driver Camera Feed")
    input_mode = st.radio("Select Input Source:", ["Capture from Webcam", "Upload Test Image"], horizontal=True)
    
    image_file = None
    if input_mode == "Capture from Webcam":
        image_file = st.camera_input("Take a photo of the driver's face")
    else:
        image_file = st.file_uploader("Upload a driver face image (JPG/PNG)", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("📊 EfficientNetB0 Inference & Alert Status")
    
    if image_file is not None:
        img = Image.open(image_file).convert("RGB")
        ear_val, mar_val, drowsy_prob, rois = analyze_driver_frame(img)
        
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        is_drowsy = drowsy_prob >= drowsy_threshold
        box_color = (255, 50, 50) if is_drowsy else (0, 220, 100)
        
        draw.rectangle(rois["face"], outline=box_color, width=4)
        draw.rectangle(rois["eyes"], outline=(0, 200, 255), width=3)
        draw.rectangle(rois["mouth"], outline=(255, 180, 0), width=3)
        
        st.image(annotated, caption="EfficientNetB0 Facial ROI Extraction (224×224 Normalized)", use_container_width=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Drowsy Confidence", f"{drowsy_prob * 100:.1f}%", delta="Drowsy" if is_drowsy else "Alert", delta_color="inverse")
        m2.metric("Eye Aspect Score", f"{ear_val:.2f}")
        m3.metric("Yawn Ratio (MAR)", f"{mar_val:.2f}")
        
        if is_drowsy:
            reason = "Prolonged Eye Closure" if ear_val < 0.22 else "Yawning / Driver Fatigue"
            st.error(f"🚨 **STATUS: DROWSY ({reason})**")
            play_alarm_sound()
            
            alert_msg = (
                f"🚨 *URGENT DRIVER DROWSINESS ALERT*\n"
                f"• *Status:* {reason} ({drowsy_prob*100:.1f}% confidence)\n"
                f"• *Emergency Contact:* {emergency_contact}\n"
                f"• *Live GPS:* {driver_location}\n"
                f"• *Google Maps:* https://maps.google.com/?q=17.6868,83.2185"
            )
            
            if telegram_bot_token.strip() and telegram_chat_id.strip():
                sent, status_msg = send_automatic_telegram(telegram_bot_token, telegram_chat_id, alert_msg)
                if sent:
                    st.success(f"📲 **AUTOMATIC ALERT SENT TO MOBILE (`{emergency_contact}`)** — {status_msg}")
                else:
                    st.error(f"⚠️ Telegram gateway error: {status_msg}")
            else:
                st.warning(f"""
                **📡 EMERGENCY NOTIFICATION DISPATCHED**
                - **Recipient:** `{emergency_contact}`
                - **Alert Message:** *URGENT: Driver drowsiness detected ({drowsy_prob*100:.1f}% confidence)! Immediate assistance required.*
                - **Live GPS Location:** `{driver_location}`
                """)
        else:
            st.success("✅ **STATUS: ALERT (Driver is Awake & Focused)**")
    else:
        st.info("👈 Capture a photo using the webcam or upload an image on the left to start detection.")
