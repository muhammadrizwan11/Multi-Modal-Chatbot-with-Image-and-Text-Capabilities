# Multi-Modal Chatbot with Image and Text Capabilities

This chatbot integrates advanced AI capabilities for both text and image interactions. Users can engage in intelligent conversations, upload images for analysis, and generate creative variations of uploaded images. The system leverages Google's Gemini AI for text-based interactions and Hugging Face's BLIP and Stable Diffusion models for image-related tasks.

## Features

### Text Chat
- **Interactive conversations**: Ask questions and receive intelligent responses powered by Google's Gemini AI.
- **Chat history**: View past interactions for better context during conversations.

### Image Explanation
- **Image analysis**: Upload an image, and the chatbot will provide a detailed description using the BLIP model.
- **User-friendly interface**: Simple image upload and explanation process.

### Image Variations
- **Creative outputs**: Generate multiple artistic variations of an uploaded image using Stable Diffusion.
- **Customizable context**: Add creative variations tailored to the image's description.

## Installation

### Prerequisites
- Python 3.7 or higher
- Access to Google Gemini API and Hugging Face models (API keys required)

### Steps
   Clone the Repository:
   ```bash
   git clone <repository-url>
   ```
  Navigate to the project folder
  ```bash
   cd <repository-folder>
   ```
 Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  Set Up Environment Variables Create a .env file in the project root and add the following keys:
  ```bash
  GOOGLE_API_KEY=your_google_api_key
  HF_AUTH_TOKEN=your_huggingface_api_token
  ```

  Run the Streamlit app
  ```bash
  run app.py
  ```

## How It Works
### Text Chat
User inputs a question or statement.
The Google Gemini API processes the input and generates a response.
Streamlit displays the interaction alongside the chat history.

### Image Explanation
User uploads an image.
The BLIP model (via Hugging Face API) generates a caption or description.
The description is displayed in the app interface.

### Image Variations
User uploads an image.
The Stable Diffusion model creates multiple variations based on the image description.
Variations are displayed for download or further use.