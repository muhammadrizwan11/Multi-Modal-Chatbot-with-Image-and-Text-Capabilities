# from dotenv import load_dotenv
# load_dotenv()  # Load environment variables
!pip install google-generativeai==0.8.0

import streamlit as st
import os
import io
import base64
import requests
from PIL import Image
import google.generativeai as genai
import random

# Configure Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Utility functions
def encode_image(image):
    """Encode PIL Image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_image_description(image_b64, api_token):
    """Retrieve image description using Hugging Face API."""
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    response = requests.post(API_URL, headers=headers, json={"inputs": image_b64})
    if response.status_code == 200:
        result = response.json()
        return result[0].get("generated_text", "Generic image description")
    else:
        st.warning(f"Error fetching description: {response.status_code}")
        return None

def generate_contextual_image(description, api_token, context="creative variation", max_retries=2):
    """Generate an image variation with retry logic and optimized parameters."""
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    payload = {
        "inputs": f"{description}, {context}",
        "parameters": {
            "num_inference_steps": 30,  # Reduced for faster processing
            "guidance_scale": 7.5,
            "seed": random.randint(1, 1000000)  # Random seed for unique outputs
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
    
    st.error("Failed to generate an image after multiple attempts.")
    return None


def generate_variations(image, api_token, num_variations=3):
    """Generate image variations with context preservation."""
    image_b64 = encode_image(image)
    image_description = get_image_description(image_b64, api_token)

    if not image_description:
        st.warning("Could not extract image description.")
        return []

    variations = []
    for _ in range(num_variations):
        variation = generate_contextual_image(
            image_description,
            api_token,
            context="creative variation"
        )
        if variation:
            variations.append(variation)

    return variations

# Initialize Streamlit App
st.set_page_config(page_title="Multi-Modal Chatbot")

st.header("Multi-Modal Chatbot")

# Initialize chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Tabs for functionality
tabs = st.tabs(["Text Chat", "Image Explanation", "Image Variations"])

# Tab 1: Text Chat
with tabs[0]:
    st.subheader("Text Chat")
    input_text = st.text_input("Enter your question:", key="text_input")
    submit_text = st.button("Send Question", key="text_submit")

    if submit_text and input_text:
        # Fetch chatbot response
        chat = genai.GenerativeModel("gemini-pro").start_chat(history=[])
        response = chat.send_message(input_text, stream=True)
        for chunk in response:
            st.write(chunk.text)
            st.session_state["chat_history"].append(("Bot", chunk.text))
        st.session_state["chat_history"].append(("You", input_text))

    # Display chat history
    st.subheader("Chat History")
    for role, text in st.session_state["chat_history"]:
        st.write(f"{role}: {text}")

# Tab 2: Image Explanation
with tabs[1]:
    st.subheader("Image Explanation")
    image_prompt = st.text_input("Input Prompt:", key="image_input")
    uploaded_image = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        image = image.convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Explain Image"):
        if uploaded_image:
            api_token = os.getenv("HF_AUTH_TOKEN")
            image_b64 = encode_image(Image.open(uploaded_image))
            description = get_image_description(image_b64, api_token)
            if description:
                st.write("Image Description:", description)
            else:
                st.error("Failed to fetch image description.")
        else:
            st.warning("Please upload an image first.")

# Tab 3: Image Variations
with tabs[2]:
    st.subheader("Generate Image Variations")
    if uploaded_image:
        api_token = os.getenv("HF_AUTH_TOKEN")
        variations = generate_variations(Image.open(uploaded_image), api_token)
        for i, variation in enumerate(variations):
            st.image(variation, caption=f"Variation {i+1}", use_container_width=True)
    else:
        st.warning("Upload an image in the 'Image Explanation' tab to generate variations.")
