import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    eye_model = load_model("eye_model.h5")
    mouth_model = load_model("mouth_model.h5")
    return eye_model, mouth_model

eye_model, mouth_model = load_models()

# ----------------------------
# FUNCTIONS
# ----------------------------
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    processed = preprocess_image(image)

    eye_pred = eye_model.predict(processed)[0][0]
    mouth_pred = mouth_model.predict(processed)[0][0]

    eye_status = "Closed" if eye_pred > 0.5 else "Open"
    mouth_status = "Yawn" if mouth_pred > 0.5 else "No Yawn"

    # Fatigue Logic
    if eye_status == "Closed":
        fatigue = "Severe Fatigue"
    elif mouth_status == "Yawn":
        fatigue = "Mild Fatigue"
    else:
        fatigue = "Alert"

    return eye_status, mouth_status, fatigue


# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📌 Project Overview",
    "📊 Dataset + EDA",
    "📥 Image Prediction",
    "📈 Results",
    "⚙️ Limitations"
])

# ----------------------------
# PROJECT OVERVIEW
# ----------------------------
if page == "📌 Project Overview":
    st.title("Driver Drowsiness Detection System")

    st.write("""
Driver fatigue is one of the leading causes of road accidents worldwide.  
This project detects drowsiness using computer vision by analyzing **eye closure and yawning behavior**.

### 🎯 Key Objectives:
- Detect Eye State → Open / Closed  
- Detect Mouth State → Yawn / No Yawn  
- Estimate Fatigue Level  

### 🧠 Approach:
- Built models using CNN & MobileNetV2  
- Developed:
  - Eye Detection Model  
  - Mouth Detection Model  
- Used Transfer Learning  

### 🚀 Final Output:
- Alert  
- Mild Fatigue  
- Severe Fatigue  

### 🌍 Real-World Applications:
- Smart Vehicles  
- Driver Monitoring Systems  
- Road Safety Applications  
    """)

# ----------------------------
# DATASET + EDA
# ----------------------------
elif page == "📊 Dataset + EDA":
    st.title("Dataset & EDA")

    st.subheader("📂 Dataset Description")
    st.write("""
Dataset contains 4 classes:
- Open
- Closed
- Yawn
- No Yawn  

Split:
- 70% Training  
- 15% Validation  
- 15% Testing  

Preprocessing:
- Resize → 224×224  
- Normalization  
- Augmentation (rotation, zoom, brightness)  
    """)

    st.subheader("📊 Class Distribution")
    st.image("class_distribution.png", caption="Class Distribution")

    st.write("""
- Shows number of images per class  
- Helps identify imbalance  
- Balanced dataset improves performance  
    """)

    st.subheader("🖼️ Sample Images")
    st.image("sample_images.png", caption="Sample Images from Dataset")

    st.write("""
- Verifies correct labeling  
- Shows visual differences between classes  
    """)

    st.subheader("📐 Image Size Analysis")
    st.image("image_size.png", caption="Image Dimension Distribution")

    st.write("""
- Images are not uniform  
- Resized to 224×224  
- All are RGB images  
    """)

# ----------------------------
# IMAGE PREDICTION (FINAL CLEAN UI VERSION)
# ----------------------------
elif page == "📥 Image Prediction":

    st.title("📥 Driver Drowsiness Prediction")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    image_type = st.radio(
        "Select Image Type:",
        ["Eye Image (Open / Closed)", "Full Face (Yawn / No Yawn)"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

        st.markdown("---")

        status = st.empty()
        status.info("🔍 Analyzing image...")

        try:
            processed = preprocess_image(image)

            # ----------------------------
            # 👁️ EYE IMAGE CASE
            # ----------------------------
            if image_type == "Eye Image (Open / Closed)":

                eye_pred = float(eye_model.predict(processed, verbose=0)[0][0])
                eye_status = "Open" if eye_pred > 0.5 else "Closed"
                status.empty()

                st.subheader("🔍 Prediction Results")

                st.metric("Eye Status", eye_status)

                with st.expander("🔬 Confidence Score"):
                    st.write(f"Eye Score: {eye_pred:.4f}")

                if eye_status == "Closed":
                    st.error("🚨 SEVERE FATIGUE (Eyes Closed)")
                else:
                    st.success("✅ ALERT (Eyes Open)")

            # ----------------------------
            # 😐 FULL FACE CASE (FINAL FIX)
            # ----------------------------
            else:

              pred = mouth_model.predict(processed, verbose=0)[0]

              class_id = pred.argmax()

              class_labels = ['closed', 'no_yawn', 'open', 'yawn']
              predicted_class = class_labels[class_id]

              status.empty()

              st.subheader("🔍 Prediction Results")

              # ✅ SMART LOGIC
              if predicted_class == "yawn":
                 mouth_status = "Yawn"
                 st.metric("Mouth Status", mouth_status)
                 st.warning("⚠️ MILD FATIGUE (Yawning)")

              elif predicted_class == "no_yawn":
                   mouth_status = "No Yawn"
                   st.metric("Mouth Status", mouth_status)
                   st.success("✅ ALERT (No Yawn)")

              elif predicted_class in ["open", "closed"]:
                  # Treat as NO YAWN (safe assumption)
                  mouth_status = "No Yawn"
                  st.metric("Mouth Status", mouth_status)
                  st.success("✅ ALERT (No Yawn)")

             
                # Debug info (useful for testing/viva)
                  with st.expander("🔬 Confidence Scores"):
                    st.write(f"Raw Output: {pred}")
                    st.write(f"Predicted Class: {predicted_class}")

        except Exception as e:
            status.empty()
            st.error(f"Prediction Error: {e}")
# ----------------------------            
# RESULTS
# ----------------------------
elif page == "📈 Results":
    st.title("Model Results & Analysis")

    # ----------------------------
    # MODEL PERFORMANCE
    # ----------------------------
    st.subheader("Model Performance")

    st.write("""
- Two deep learning models were developed:
  - Eye State Detection Model (Open / Closed)
  - Mouth State Detection Model (Yawn / No Yawn)

- Both models achieved strong performance on validation data
- Models successfully learned key facial features such as eye closure and yawning
- Performance is consistent across different image conditions
    """)

    # ----------------------------
    # CONFUSION MATRIX (4-CLASS)
    # ----------------------------
    st.subheader("Confusion Matrix (4-Class Classification)")

    st.image("confusion_matrix.png", caption="Actual vs Predicted Classes")

    st.write("""
- Displays comparison between actual and predicted classes
- Majority of predictions are correct
- Minor confusion observed between similar classes:
  - Open vs No Yawn
  - Yawn vs Open
    """)

    # ----------------------------
    # 3-LEVEL FATIGUE VISUALIZATION
    # ----------------------------
    st.subheader("3-Level Fatigue Classification")

    st.image("fatigue_confusion_matrix.png", caption="Fatigue Level Classification")

    st.write("""
The 4-class predictions are mapped into 3 fatigue levels:

- **Alert** → Open + No Yawn  
- **Mild Fatigue** → Yawning detected  
- **Severe Fatigue** → Eyes Closed  

This mapping improves real-world interpretation of driver condition.

- The confusion matrix shows how accurately fatigue levels are predicted
- Most cases are correctly classified into Alert, Mild, and Severe categories
- This makes the system suitable for real-time driver monitoring applications
    """)

# ----------------------------
# LIMITATIONS
# ----------------------------
elif page == "⚙️ Limitations":
    st.title("Limitations")

    st.write("""
- Model may fail in low-light or blurry images  
- Confusion between similar classes (Open vs No Yawn)  
- Works mainly on static images  
- Requires clear visibility of face  
- Limited dataset affects generalization  
    """)