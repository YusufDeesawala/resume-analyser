import requests
from model import extract_text_from_pdf
# Set your API key here
API_KEY = "your api key"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

pdf_path = 'path to pdf'

# Define the prompt
prompt = f'''Given to you is the entire content of a resume. I want you to extract ['ExperienceYears','Education','Certifications','JobRole','SalaryExpectation','ProjectsCount','AIScore','Skill1','Skill2','Skill3'].
the resume: {extract_text_from_pdf(pdf_path)}
ExperienceYears should have the experience of that person
Education should have the type of qualification they have. Example B.Sc, B.Tech, etc. Menttion the type of degree.
Certifications should include all their certifications 
JobRole should contain the top role they are applying for
SalaryExpectation should be a number not a range 
ProjectsCountn should contain the number of proojects they have done.
AIScore is the score u would give to the person based on their resume range(0-100)
Skill1, Skill2, Skill3 should contain their top three skills.
Give me all this data in a such a way that i can feed it into a model for exaluation.'''

# Request payload
payload = {
    "contents": [
        {
            "parts": [{"text": prompt}]
        }
    ]
}

# Headers
headers = {
    "Content-Type": "application/json"
}

# Make the API request
response = requests.post(API_URL, json=payload, headers=headers)

# Check response
if response.status_code == 200:
    data = response.json()
    generated_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response received.")
    print("Gemini API Response:\n")
    print(generated_text)
else:
    print(f"Error {response.status_code}: {response.text}")
