import os
import json
import requests
from text_extraction import extract_text_from_pdf

# Load API key from environment variable (safer than hardcoding)
API_KEY = "AIzaSyCwFzKMYFOQG__sr86s2bDJ9ZmIdzlUTVw"  # Set this in your terminal or environment
if not API_KEY:
    raise ValueError("API key not found. Set GEMINI_API_KEY as an environment variable.")

# Gemini API URL
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Path to the resume PDF
pdf_path = "resume.pdf"

# Extract text from the resume PDF
try:
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text.strip():
        raise ValueError("Extracted text is empty. Ensure the PDF contains readable text.")
except Exception as e:
    print(f"Error extracting text from PDF: {e}")
    exit(1)  # Exit the script if text extraction fails

# Define the structured prompt for Gemini API
prompt = f"""
Extract the following details from the given resume text and return the output as a JSON object:
- **ExperienceYears**: Total years of work experience.
- **Education**: Degree or qualification (e.g., B.Sc, B.Tech, M.Sc).
- **Certifications**: List all certifications.
- **JobRole**: The main role the person is applying for.
- **SalaryExpectation**: Expected salary (as a single number, not a range).
- **ProjectsCount**: Number of projects completed.
- **AIScore**: A score (0-100) based on resume quality.
- **Skill1, Skill2, Skill3**: Top three skills.

### Resume Content:


Return the extracted data **only** in valid JSON format.
"""

# Request payload for Gemini API
payload = {
    "contents": [{"parts": [{"text": prompt}]}]
}

# Headers for the API request
headers = {"Content-Type": "application/json"}

try:
    # Make the API request
    response = requests.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()  # Raise error for HTTP failures
    
    # Extract response data
    data = response.json()
    generated_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")

    # Parse the response as JSON
    try:
        parsed_json = json.loads(generated_text)
        print("Extracted Resume Data:\n", json.dumps(parsed_json, indent=4))
        
        # Save the output to a file
        with open("resume_analysis.json", "w") as f:
            json.dump(parsed_json, f, indent=4)
        print("\nData saved to resume_analysis.json")

    except json.JSONDecodeError:
        print("Error: The API did not return valid JSON. Here is the raw response:")
        print(generated_text)

except requests.exceptions.RequestException as e:
    print(f"API Request Error: {e}")
