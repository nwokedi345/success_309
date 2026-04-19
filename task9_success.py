import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request

# --- 1. UI CONFIG ---
st.set_page_config(page_title="DeepVision Classifier", page_icon="🔍", layout="wide")

# --- 2. PROFESSIONAL BLUE & WHITE CSS ---
st.markdown("""
    <style>
    /* Clean White Background */
    .stApp { background-color: #f4f7f6; }
    
    /* Modern Blue Header */
    .tech-header {
        background-color: #004aad;
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: left;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 8px solid #00b4d8;
    }
    .tech-header h1 { font-family: 'Arial', sans-serif; font-weight: 700; margin: 0; color: white; }
    .tech-header p { margin: 5px 0 0 0; font-size: 1.1rem; color: #e0e0e0; }

    /* Minimalist Result Cards */
    .card-dog { border-left: 6px solid #28a745; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .card-cat { border-left: 6px solid #fd7e14; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .card-null { border-left: 6px solid #6c757d; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    
    .card-title { font-size: 1.8rem; font-weight: 800; color: #333; margin-bottom: 5px;}
    .card-data { font-size: 1.1rem; color: #555; }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC (UNCHANGED) ---
@st.cache_resource
def load_ai_engine():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        urllib.request.urlretrieve(url, "imagenet_classes.txt")
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
    except Exception:
        categories = [f"Object {i}" for i in range(1000)]

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, categories, transform

# --- 4. SUCCESS'S DASHBOARD ---
st.markdown("""
<div class="tech-header">
    <h1>🔍 DeepVision AI Image Classifier</h1>
    <p><b>Developer:</b> Samuel Success Akachukwu | <b>Task 9</b> | CSC 309</p>
</div>
""", unsafe_allow_html=True)

model, categories, transform = load_ai_engine()

# --- NEW SPLIT LAYOUT ---
col_img, col_ai = st.columns([1, 1.2], gap="large")

with col_img:
    st.markdown("### 1. Input Source")
    uploaded_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Target Image Acquired", use_container_width=True)

with col_ai:
    st.markdown("### 2. Neural Analysis")
    if uploaded_file is None:
        st.info("Awaiting image upload on the left panel...")
    else:
        if st.button("Initialize Neural Scan", type="primary", use_container_width=True):
            with st.spinner("Processing tensors..."):
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = model(tensor)

                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_id = torch.topk(probs, 1)
                cid = top_id.item()
                conf = top_prob.item() * 100
                label = categories[cid].replace('_', ' ').title()

                st.markdown("<br>", unsafe_allow_html=True)
                
                if 151 <= cid <= 268: # Dog
                    st.markdown(f"""
                    <div class="card-dog">
                        <div class="card-title">CANINE DETECTED</div>
                        <div class="card-data"><b>Classification:</b> {label}<br><b>Accuracy Score:</b> {conf:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif 281 <= cid <= 285: # Cat
                    st.markdown(f"""
                    <div class="card-cat">
                        <div class="card-title">FELINE DETECTED</div>
                        <div class="card-data"><b>Classification:</b> {label}<br><b>Accuracy Score:</b> {conf:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else: # Other
                    st.markdown(f"""
                    <div class="card-null">
                        <div class="card-title">UNKNOWN ENTITY</div>
                        <div class="card-data"><b>Prediction:</b> {label}<br><b>Accuracy Score:</b> {conf:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)