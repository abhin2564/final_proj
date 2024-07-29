import streamlit as st 
import cv2  
import pytesseract  
from PIL import Image  
import numpy as np  

def extract_text_from_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    st.image(gray, caption='Grayscale Image', use_column_width=True)
    st.image(blur, caption='Blurred Image', use_column_width=True)
    st.image(thresh, caption='Thresholded Image', use_column_width=True)
    text = pytesseract.image_to_string(thresh)
    return text

st.title("Image to Text Extractor")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Extracting text...")
    extracted_text = extract_text_from_image(image)
    st.header("Extracted Text")
    st.write(extracted_text)
