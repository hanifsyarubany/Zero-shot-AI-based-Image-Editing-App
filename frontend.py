import streamlit as st
import requests
import base64

st.set_page_config(page_title="AI Image Editor", layout="wide")
st.title("🪄 AI-based Image Editing App")

# Main 2-column layout
col1, col2 = st.columns([1, 1])  # col1 = input, col2 = output

with col1:
    st.subheader("📥 Input Section")

    # Sub-columns inside input section
    subcol1, subcol2 = st.columns([1, 1])  # subcol1 = query, subcol2 = reference

    with subcol1:
        st.markdown("#### 🖼️ Query Image + Instruction")
        query_image = st.file_uploader("Upload Query Image", type=["png", "jpg", "jpeg"], key="query")
        if query_image:
            st.image(query_image, width=200, caption="Query Image")

        user_query = st.text_area("Instruction", placeholder="e.g., Remove the object on the right...")

    with subcol2:
        st.markdown("#### 🎨 Reference Image (Optional)")
        use_reference = st.checkbox("Use reference image?")
        ref_image = None
        if use_reference:
            ref_image = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"], key="ref")
            if ref_image:
                st.image(ref_image, width=200, caption="Reference Image")

    submitted = st.button("✨ Generate Edited Image")

with col2:
    st.subheader("🖼️ Output Result")

    if submitted:
        if not query_image or not user_query.strip():
            st.error("Please upload a query image and enter an instruction.")
        else:
            # Convert query image
            query_b64 = base64.b64encode(query_image.read()).decode("utf-8")

            if use_reference and ref_image:
                ref_b64 = base64.b64encode(ref_image.read()).decode("utf-8")
                payload = {
                    "user_query": user_query,
                    "base64_img_query": query_b64,
                    "base64_img_ref": ref_b64
                }
                endpoint = "http://localhost:8080/diffusion-reference"
            else:
                payload = {
                    "user_query": user_query,
                    "base64_img": query_b64
                }
                endpoint = "http://localhost:8080/diffusion-prompt"

            with st.spinner("⏳ Processing... please wait..."):
                try:
                    response = requests.post(endpoint, json=payload)
                    response.raise_for_status()
                    result_b64 = response.json()["base64_img"]

                    st.success("✅ Done!")
                    st.image(base64.b64decode(result_b64), caption="✨ Edited Image")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
