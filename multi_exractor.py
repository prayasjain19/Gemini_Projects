import streamlit as st
import google.generativeai as genai
from PIL import Image

# Directly assign the API key here
GOOGLE_API_KEY = "AIzaSyCMzvMg_vSz5nhrSSMP-XO-8q0QGeNIfko"  # Replace with your actual Google API key
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

#Take the prompt and input from user
def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes = uploaded_file.getvalue()
        
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")

st.set_page_config(page_title="MultiLanguage Invoice Extractor")

#streamlit
st.header("MultiLanguage Invoice Extractor")

#input prompt
input = st.text_input("Input Prompt: ", key="input")

#Upload File
uploaded_file = st.file_uploader("Choose an image of the invoice...", type=["jpg", "png", "jpeg"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)

submit = st.button("Tell me about this image")

#input prompt
input_prompt = """
You are an expert in understanding invoices. We will upload an image of an invoice, and you will have to answer any question based on the uploaded invoice image.
"""

#submiting response
if submit:
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("The Response is")
    st.write(response)
