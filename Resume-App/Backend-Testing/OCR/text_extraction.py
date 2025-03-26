import pytesseract
from pdf2image import convert_from_path
import os

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    
    extracted_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        extracted_text += f"\n--- Page {i+1} ---\n{text}\n"
    
    return extracted_text

