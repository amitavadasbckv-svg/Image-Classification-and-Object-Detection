import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO("best.pt")   # Use yolov8s.pt, yolov8m.pt, yolov8l.pt for better accuracy


st.title("Object Detection App (YOLOv8 + Streamlit)")
st.write("Upload an image to detect objects")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to array
    img_array = np.array(image)

    # Run YOLO detection
    st.write("Detecting objects...")
    results = model.predict(img_array)

    # Draw bounding boxes
    result_img = results[0].plot()   # returns image with bounding boxes

    # Display result
    st.image(result_img, caption="Detection Result", use_container_width =True)

    # Show detected labels
    st.subheader("Detected Objects:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[cls_id]
        st.write(f"{class_name} - {confidence:.2f}")
