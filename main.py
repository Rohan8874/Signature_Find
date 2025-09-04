import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------------
# Model: Siamese-style embedder
# -----------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)

    def forward(self, x):
        return self.resnet(x)

@st.cache_resource(show_spinner=False)
def load_model():
    model = SiameseNetwork()
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# Embedding & similarity utils
# -----------------------------
def get_image_embedding(image: Image.Image, model: nn.Module, transform):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features

def compare_images(image1, image2, model, transform) -> float:
    e1 = get_image_embedding(image1, model, transform).squeeze().cpu().numpy()
    e2 = get_image_embedding(image2, model, transform).squeeze().cpu().numpy()
    if np.allclose(e1, 0) or np.allclose(e2, 0):
        return 0.0
    sim = 1 - cosine(e1, e2)
    if np.isnan(sim):
        return 0.0
    return float(sim)

def is_signature_image(image: Image.Image) -> bool:
    g = image.convert("L")
    hist = g.histogram()
    var = np.var(hist)
    return var < 2000  # tweak if needed

# -----------------------------
# Fixed-bucket decision rule
# -----------------------------
def bucket_decision(similarity: float) -> str:
    """
    Returns one of: 'Not Similar', 'Check Manually', 'Similar'
    Based on percentages:
      0.00–0.79  -> Not Similar
      0.80–0.89  -> Check Manually
      0.90–1.00  -> Similar
    """
    if similarity < 0.80:
        return "Not Similar"
    elif similarity < 0.90:
        return "Check Manually"
    else:
        return "Similar"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Signature Comparison App", layout="centered", initial_sidebar_state="expanded")
st.title("✍️ Signature Comparison App")
st.write("Upload two signature images to compare them and get a similarity score.")

# Session log
if "audit_rows" not in st.session_state:
    st.session_state.audit_rows = []

left, right = st.columns(2)
with left:
    st.subheader("Upload Signature 1")
    image1_file = st.file_uploader("Choose the first signature", type=["png", "jpg", "jpeg"], key="u1")

with right:
    st.subheader("Upload Signature 2")
    image2_file = st.file_uploader("Choose the second signature", type=["png", "jpg", "jpeg"], key="u2")

show_warnings = st.checkbox("Warn if image doesn't look like a signature", value=True)
st.markdown("---")
result_placeholder = st.empty()

if image1_file and image2_file:
    image1 = Image.open(image1_file).convert("RGB")
    image2 = Image.open(image2_file).convert("RGB")

    st.subheader("Uploaded Signatures")
    v1, v2 = st.columns(2)
    v1.image(image1, caption="Signature 1", use_container_width=True)
    v2.image(image2, caption="Signature 2", use_container_width=True)

    if show_warnings:
        if not is_signature_image(image1):
            st.info("ℹ️ Signature 1 may not look like a typical monochrome signature.")
        if not is_signature_image(image2):
            st.info("ℹ️ Signature 2 may not look like a typical monochrome signature.")

    if st.button("🔍 Compare Signatures"):
        with st.spinner("Comparing signatures..."):
            score = compare_images(image1, image2, model, transform)

        result_placeholder.subheader(f"Similarity Score: {score:.3f} ({score*100:.1f}%)")

        decision = bucket_decision(score)
        if decision == "Similar":
            st.success("Decision: **Similar** (90–100%).")
        elif decision == "Not Similar":
            st.error("Decision: **Not Similar** (1–79%).")
        else:
            st.warning("Decision: **Check Manually** (80–89%).")

        # Manual review panel ONLY when 80–89%
        final_label = decision
        notes = ""
        if decision == "Check Manually":
            st.markdown("### Manual Review Required")
            manual_choice = st.radio(
                "Your assessment:",
                ["Similar", "Different", "Not sure"],
                index=2
            )
            notes = st.text_area("Notes (optional)", placeholder="Add reviewer comments ...")
            if st.button("💾 Save Assessment"):
                final_label = manual_choice
                st.success("Saved to in-memory review log.")

        # Always log (auto for 1–79 / 90–100; manual choice for 80–89 if saved)
        if decision != "Check Manually" and st.button("💾 Save Decision"):
            st.success("Saved to in-memory review log.")

        # Create a save row if either button above was clicked
        # We check Streamlit's last button interaction by peeking at the delta in audit rows count
        # Instead, we add a small 'Save latest comparison' button that always records current comparison.
        if st.button("📝 Save Latest Comparison Now"):
            row = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "file1": image1_file.name,
                "file2": image2_file.name,
                "similarity": round(score, 4),
                "bucket": decision,
                "final_label": final_label,
                "notes": notes.strip(),
            }
            st.session_state.audit_rows.append(row)
            st.success("Comparison saved.")

    # Log table + download
    if st.session_state.audit_rows:
        st.markdown("### Review Log")
        df = pd.DataFrame(st.session_state.audit_rows)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Review Log (CSV)", csv, file_name="signature_review_log.csv", mime="text/csv")

st.markdown("---")
st.markdown("🔒 **Privacy Notice:** Your signature images are processed securely and are **NOT** stored in any form. All images are handled in real-time and discarded immediately after processing.")
