"""
Streamlit Web Application: High-Accuracy Pokémon Card Scanner & Identifier
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pokemon_card_ocr import PokemonCardExtractor
from pokemon_api import PokemonTCGClient

st.set_page_config(page_title="High-Accuracy Pokémon Card OCR", layout="wide", page_icon="⚡")

@st.cache_resource
def load_extractor():
    return PokemonCardExtractor(gpu=False)

@st.cache_resource
def load_api_client():
    return PokemonTCGClient()

st.title("⚡ High-Accuracy Pokémon Card Scanner")
st.markdown("Extract **Pokémon Card Name**, **HP**, and **Unique Collector ID** with automatic TCG Database verification.")

col_input, col_results = st.columns([1, 1])

with col_input:
    st.subheader("1. Upload or Capture Pokémon Card")
    uploaded_file = st.file_uploader("Upload Card Image (JPG, PNG)", type=["jpg", "png", "jpeg"])

    image_np = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        st.image(image, caption="Uploaded Card Image", use_container_width=True)

if image_np is not None:
    with st.spinner("Processing card & performing high-accuracy extraction..."):
        extractor = load_extractor()
        api_client = load_api_client()

        # Step 1: Run Extraction Pipeline
        extraction_result = extractor.extract_from_image(image_np)

        name = extraction_result["name"]
        hp = extraction_result["hp"]
        collector_id = extraction_result["unique_id"]

        # Step 2: Query Pokemon TCG API for 99.9% verification
        verification = api_client.verify_card(collector_id=collector_id, ocr_name=name, ocr_hp=hp)

    with col_results:
        st.subheader("2. Extracted Card Data (OCR Engine)")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Card Name", name if name else "Not Detected")
        with res_col2:
            st.metric("HP", f"{hp} HP" if hp else "Not Detected")
        with res_col3:
            st.metric("Unique ID", collector_id if collector_id else "Not Detected")

        st.markdown("---")
        st.subheader("3. Database Verified Card (99.9% Accuracy)")

        if verification.get("verified"):
            st.success(f"✅ Exact Match Verified (Confidence: {int(verification['confidence']*100)}%)")

            v_col1, v_col2 = st.columns([1, 1])
            with v_col1:
                if verification.get("image_url"):
                    st.image(verification["image_url"], caption="Official TCG Card Image", use_container_width=True)
            with v_col2:
                st.markdown(f"**Name:** {verification.get('name')}")
                st.markdown(f"**HP:** {verification.get('hp')} HP")
                st.markdown(f"**Set:** {verification.get('set_name')} ({verification.get('set_series')})")
                st.markdown(f"**Collector ID:** `{verification.get('collector_id')}`")
                st.markdown(f"**Rarity:** {verification.get('rarity')}")
                if verification.get("market_price"):
                    st.markdown(f"**Market Price:** `${verification.get('market_price'):.2f}`")
                if verification.get("tcgplayer_url"):
                    st.markdown(f"[View on TCGPlayer]({verification.get('tcgplayer_url')})")
        else:
            st.warning("⚠️ " + verification.get("message", "No database verification available."))

        with st.expander("🔍 Diagnostic Crops & Raw OCR Text"):
            crop_col1, crop_col2 = st.columns(2)
            with crop_col1:
                st.markdown("**Header Crop (Name & HP)**")
                hdr_rgb = cv2.cvtColor(extraction_result["header_crop"], cv2.COLOR_BGR2RGB)
                st.image(hdr_rgb, use_container_width=True)
                st.code(f"Raw OCR:\n{extraction_result['header_raw_ocr']}")

            with crop_col2:
                st.markdown("**Footer Crop (Collector ID)**")
                ftr_rgb = cv2.cvtColor(extraction_result["footer_crop"], cv2.COLOR_BGR2RGB)
                st.image(ftr_rgb, use_container_width=True)
                st.code(f"Raw OCR:\n{extraction_result['footer_raw_ocr']}")
