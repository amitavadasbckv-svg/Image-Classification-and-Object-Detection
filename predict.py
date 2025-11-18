import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
def predict_image(image_path, model, transform):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)   # add batch dimension

    with torch.no_grad():
        outputs = model(img_t)
        # softmax → convert to probability
        probs = torch.softmax(outputs, dim=1)
        # confidence & class
        confidence, predicted_class = torch.max(probs, dim=1)
        st.write(f"Confidence score: {confidence.item():.2f}")
        _, predicted = torch.max(outputs, 1)

    classes = ['bird', 'drone']
    return classes[predicted.item()]
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img)
    import torch
    from torch.serialization import add_safe_globals
    import torch.nn.modules.container

    add_safe_globals([torch.nn.modules.container.Sequential])
    model = torch.load(
    r"transfer_learning.pth",
    map_location=torch.device("cpu"), weights_only=False
  )
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])
    result = predict_image(uploaded, model, transform)
    st.write("Prediction:", result)