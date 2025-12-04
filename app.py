import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ======================================================
# CONFIG
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace this with your actual 10 selected classes,
# in the SAME order you used during training.
SELECTED_CLASSES = [
    'Faces',
    'Faces_easy',
    'Leopards',
    'Motorbikes',
    'accordion',
    'airplanes',
    'bass',
    'bonsai',
    'brain',
    'buddha'
]


# Path to your trained model weights
MODEL_PATH = "resnet18_caltech10.pth"   

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ======================================================
# TRANSFORMS (must match your eval_transform + normalization)
# ======================================================
eval_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def normalize_tensor(t):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std

# ======================================================
# MODEL LOADER
# ======================================================
@st.cache_resource
def load_model():
    num_classes = len(SELECTED_CLASSES)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Replace FC layer to match your training
    model.fc = nn.Linear(512, num_classes)

    # Load your fine-tuned weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ======================================================
# PREDICTION FUNCTION
# ======================================================
def predict_image(image: Image.Image):
    # Apply same eval transform as validation/test
    img_t = eval_transform(image)       # [3, H, W]
    img_t = img_t.unsqueeze(0)          # [1, 3, H, W]

    img_t = normalize_tensor(img_t)     # normalize with ImageNet stats
    img_t = img_t.to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]   # [num_classes]

    # Top-1
    top1_prob, top1_idx = torch.max(probs, dim=0)
    top1_class = SELECTED_CLASSES[top1_idx.item()]

    # Top-3
    top3_prob, top3_idx = torch.topk(probs, k=min(3, len(SELECTED_CLASSES)))
    top3 = [
        (SELECTED_CLASSES[idx.item()], prob.item())
        for idx, prob in zip(top3_idx, top3_prob)
    ]

    return top1_class, top1_prob.item(), top3

# ======================================================
# STREAMLIT APP LAYOUT
# ======================================================
st.title("🖼️ Caltech-101 Image Classifier (ResNet18, 10 Classes)")
st.write("Upload an image and test the model on **unseen data**.")

st.sidebar.header("Model Info")
st.sidebar.write(f"**Device:** {DEVICE}")
st.sidebar.write(f"**Num classes:** {len(SELECTED_CLASSES)}")
st.sidebar.write("**Classes:**")
for c in SELECTED_CLASSES:
    st.sidebar.write(f"- {c}")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with PIL
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    # Run prediction
    top1_class, top1_prob, top3 = predict_image(image)

    with col2:
        st.subheader("Prediction")
        st.markdown(f"**Top-1 Prediction:** `{top1_class}`")
        st.markdown(f"**Confidence:** `{top1_prob*100:.2f}%`")

        st.markdown("**Top-3 Classes:**")
        for cls, prob in top3:
            st.write(f"- {cls}: {prob*100:.2f}%")

else:
    st.info("👆 Upload an image file to see the model's prediction on unseen data.")
