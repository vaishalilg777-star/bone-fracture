import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3

st.set_page_config(page_title="PINNACLE - AI Bone Fracture Detection", page_icon="🦴", layout="wide")

# 1. Database Setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)')
    cursor.execute("INSERT OR IGNORE INTO users VALUES ('admin', 'admin123', 'admin')")
    conn.commit()
    conn.close()

init_db()

# 2. Load Model (Cached for speed)
@st.cache_resource
def load_fracture_model():
    return tf.keras.models.load_model('bone_fracture_model.h5')

model = load_fracture_model()

# 3. Grad-CAM Functions
def make_gradcam_heatmap(img_array, model):
    try:
        last_conv_layer = next((l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)), None)
        if not last_conv_layer:
            return None
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.maximum(tf.squeeze(heatmap), 0)
        if tf.math.reduce_max(heatmap) != 0:
            heatmap /= tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception:
        return None

def display_gradcam(img_pil, heatmap, alpha=0.4):
    if heatmap is None:
        return img_pil
    img = np.array(img_pil.convert("RGB"))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0))

# 4. Login State & UI
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🏥 PINNACLE - Advanced AI Diagnostics Portal")
    st.subheader("Secure Login / Register: Bone Fracture Detection System")
    username = st.text_input("Username", placeholder="New users will be registered automatically")
    password = st.text_input("Password", type="password")
    if st.button("Login / Register", type="primary"):
        if not username or not password:
            st.warning("⚠️ Please enter username and password.")
        else:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("SELECT password, role FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if user:
                conn.close()
                if user[0] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Incorrect password!")
            else:
                cursor.execute("INSERT INTO users VALUES (?, ?, ?)", (username, password, 'user'))
                conn.commit()
                conn.close()
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
else:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🦴 AI-Powered Bone Fracture Detection & Heatmap Localization")
    st.write("Upload an X-ray image to get instant AI classification along with **Grad-CAM visual explanation**.")

    uploaded_file = st.file_uploader("Upload Bone X-Ray", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        
        if st.button("🔍 Run AI Analysis & Grad-CAM", type="primary"):
            img_resized = image.resize((150, 150))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            prediction = model.predict(img_array)[0][0]
            heatmap = make_gradcam_heatmap(img_array, model)
            explained_img = display_gradcam(img_resized, heatmap)

            with col2:
                if prediction > 0.5:
                    st.success(f"✅ **Normal / No Fracture** (Confidence: {prediction * 100:.2f}%)")
                else:
                    st.error(f"🚨 **Fracture Detected** (Confidence: {(1 - prediction) * 100:.2f}%)")
                st.image(explained_img, caption="Grad-CAM Heatmap Localization (AI Focus Area)", use_container_width=True)