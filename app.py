import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
import gdown
import os

# 1. Google Drive-la irunthu model-a download panrom
file_id = '1ABCxyz_UNGA_FILE_ID_INGA_PODUNGA'  # <--- Unga Google Drive file ID-a inga podunga
url = f'https://drive.google.com/uc?id={file_id}'
output = 'bone_fracture_model.h5'

if not os.path.exists(output):
    with st.spinner("Downloading AI model from Google Drive... Please wait."):
        gdown.download(url, output, quiet=False)

# 2. Database setup (Multi-user login & auto-signup)
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT
        )
    ''')
    cursor.execute("INSERT OR IGNORE INTO users VALUES ('admin', 'admin123', 'admin')")
    conn.commit()
    conn.close()

init_db()

# 3. Model load panrom
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(output)

model = load_model()

# 4. Grad-CAM Heatmap Generation Function
def make_gradcam_heatmap(img_array, model):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            return None
            
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        if tf.math.reduce_max(heatmap) != 0:
            heatmap = heatmap / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print("Grad-CAM Error:", e)
        return None

def display_gradcam(img_path, heatmap, alpha=0.4):
    if heatmap is None:
        return img_path
    img = np.array(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(superimposed_img)

# 5. Prediction & Explanation Logic
def predict_and_explain(image):
    if image is None:
        return "⚠️ Please upload an X-ray image first.", None
    img_resized = image.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    heatmap = make_gradcam_heatmap(img_array, model)
    if heatmap is not None:
        explained_img = display_gradcam(image.resize((150, 150)), heatmap)
    else:
        explained_img = image.resize((150, 150))
        
    if prediction > 0.5:
        result_text = f"✅ **Normal / No Fracture**\n* Confidence: {prediction * 100:.2f}%\n* Status: Bone structure appears normal."
    else:
        result_text = f"🚨 **Fracture Detected**\n* Confidence: {(1 - prediction) * 100:.2f}%\n* Status: Attention required. Highlighted region shows potential fracture zone."
    return result_text, explained_img

# 6. Streamlit UI Design
st.set_page_config(page_title="PINNACLE - Bone Fracture Detection", page_icon="🏥", layout="centered")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.markdown("# 🏥 PINNACLE - Advanced AI Diagnostics Portal")
    st.markdown("### Secure Login / Register: Bone Fracture Detection System")
    
    username_input = st.text_input("Username", placeholder="Enter username (New users will be registered automatically)")
    password_input = st.text_input("Password", type="password", placeholder="Enter password")
    
    if st.button("Login / Register", type="primary"):
        if not username_input or not password_input:
            st.warning("⚠️ Please enter username and password.")
        else:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("SELECT password, role FROM users WHERE username = ?", (username_input,))
            user = cursor.fetchone()
            if user:
                stored_password, role = user
                conn.close()
                if stored_password == password_input:
                    st.session_state.logged_in = True
                    st.session_state.username = username_input
                    st.rerun()
                else:
                    st.error("❌ Incorrect password!")
            else:
                cursor.execute("INSERT INTO users VALUES (?, ?, ?)", (username_input, password_input, 'user'))
                conn.commit()
                conn.close()
                st.session_state.logged_in = True
                st.session_state.username = username_input
                st.success(f"✨ Account created! Welcome, {username_input}.")
                st.rerun()
else:
    st.markdown(f"# 🦴 AI-Powered Bone Fracture Detection & Heatmap Localization")
    st.markdown(f"Welcome back, **{st.session_state.username}**! Upload an X-ray image to get instant AI classification along with **Grad-CAM visual explanation**.")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
        
    col1, col2 = st.columns(2)
    with col1:
        img_file = st.file_uploader("Upload Bone X-Ray", type=["jpg", "jpeg", "png"])
        if img_file is not None:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
            if st.button("🔍 Run AI Analysis & Grad-CAM", type="primary"):
                with st.spinner("Analyzing X-ray image..."):
                    res_text, exp_img = predict_and_explain(image)
                    with col2:
                        st.markdown("### Diagnostic Result")
                        st.markdown(res_text)
                        if exp_img:
                            st.image(exp_img, caption="Grad-CAM Heatmap Localization", use_column_width=True)
