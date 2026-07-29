import streamlit as st
from PIL import Image

from predict import analyze_image

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI PPE Compliance System",
    page_icon="🦺",
    layout="centered"
)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🦺 AI PPE Compliance System")

st.write("Upload an image to check PPE compliance.")

# -------------------------------------------------
# Upload Image
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# If image uploaded
# -------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -------------------------------------------------
    # Analyze Button
    # -------------------------------------------------
    if st.button("🔍 Analyze Image"):

        # Save uploaded image
        image.save("uploaded_image.jpg")

        st.success("✅ Image uploaded successfully!")

        # Run AI
        result_image, workers = analyze_image("uploaded_image.jpg")

        st.success("✅ AI analysis completed!")

        # -------------------------------------------------
        # Show Final Annotated Image
        # -------------------------------------------------
        st.subheader("Prediction Result")

        st.image(
            result_image,
            use_container_width=True
        )

        # -------------------------------------------------
        # Compliance Results
        # -------------------------------------------------
        st.subheader("Compliance Results")

        for i, worker in enumerate(workers):

            st.markdown(f"### 👷 Worker {i+1}")

            if worker.status == "SAFE":

                st.success("Status: SAFE")

            else:

                st.error("Status: VIOLATION")

            if worker.missing:

                st.write("**Missing PPE:**")

                for item in worker.missing:

                    st.write(f"• {item}")

            else:

                st.write("✅ All required PPE detected.")

        # -------------------------------------------------
        # Download Button
        # -------------------------------------------------
        with open(result_image, "rb") as file:

            st.download_button(
                label="⬇ Download Annotated Image",
                data=file,
                file_name="PPE_Result.jpg",
                mime="image/jpeg"
            )