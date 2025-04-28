from flask import Flask, render_template, request, jsonify , url_for
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from serpapi import GoogleSearch
import requests
import json
import os
import pytesseract
from pdf2image import convert_from_path
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from google import genai
from gtts import gTTS
import whisper
import uuid


load_dotenv()

app = Flask(__name__)

# Load the trained CatBoost model
model_path = "models/catboost_model_1.pkl"  # Ensure this file is in your project folder
model = pickle.load(open(model_path, "rb"))

# Define categories for categorical features (ensure same order as training)
education_levels = ['B.Sc', 'B.Tech', 'MBA', 'PhD', 'M.Tech']
certifications = ['None', 'Deep Learning Specialization', 'AWS Certified', 'Google ML']
job_roles = ['AI Researcher', 'Cybersecurity Analyst', 'Data Scientist', 'Software Engineer']
skills = ['sql', 'python', 'tensorflow', 'pytorch', 'cybersecurity', 'ethical hacking', 'networking', 'java', 'react', 'deep learning', 'c', 'machine learning', 'linux', 'nlp']


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set upload folder in the Flask app config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load API key from environment variable (safer than hardcoding)
API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in your terminal or environment
if not API_KEY:
    raise ValueError("API key not found. Set GEMINI_API_KEY as an environment variable.")

# Gemini API URL
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Whisper Model
Whisper_Model = whisper.load_model("base")

#Gemini Client
client = genai.Client(api_key=API_KEY)

# Function used in interview component
def create_system_prompt(name, role):
    return {
        'role': 'model',
        "parts": [{
            "text": f"You are interviewing {name} for the {role} position. Ask relevant questions, if the user is able to answer that question correctly then proceed to increase the difficulty of the questions. Your name is Alex and you will refer to the user as {name}."
        }]
    }


# Define the system prompt for the interview
SYSTEM_PROMPT = {
     'role': 'model', 
    "parts": [{"text":"You are interviewing the user for the Machine Learning Developer position. Ask relevant questions, if the user is able to answer that question correctly then proceed to increase the difficulty of the questions. Your name is Alex and you will refer to the user with the name resource."}]
}



# Function to encode categorical values the same way as training
def encode_category(value, category_list):
    return category_list.index(value) if value in category_list else -1  # -1 for unseen values

# Function to extract text from a PDF file using OCR
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    
    extracted_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        extracted_text += f"\n--- Page {i+1} ---\n{text}\n"
    
    return extracted_text


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/job-search')
def job():
    return render_template('Job.html')

@app.route('/resume')   
def resume():
    return render_template('Resume_Extraction.html')

@app.route('/resume-analyzer')
def resume_analyzer():
    return render_template('Resume_Analyzer.html',  education_levels=education_levels, certifications=certifications, job_roles=job_roles, skills=skills)

@app.route('/interview')
def interview():
    return render_template('Interview.html')

@app.route('/ai-score', methods=['POST'])
def ai_score():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    resume_file = request.files['resume']
    
    if resume_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file to the specified folder
    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_path)
    
    try:
        # Extract text from the uploaded resume file
        resume_ai_text = extract_text_from_pdf(resume_path)
        
        if not resume_ai_text.strip():
            return jsonify({'error': 'Extracted text is empty. Ensure the PDF contains readable text.'}), 400
        
        prompt = f"""
        Perform a comprehensive analysis of the resume = {resume_ai_text}. and rate my resume from 0 to 100.
        Note= Just return the number as the response no text are allowed."""
        
        payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            # Send request to AI API
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            # Extract and parse response
            response_data = response.json()
            generated_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
            
            # Clean and parse the AI score number (e.g., "75")
            generated_text = generated_text.strip("```json").strip("```")
            ai_score = int(generated_text)  # Assuming the response is just a number
            
            return jsonify(ai_score)  # Return only the AI score as a number
        
        except json.JSONDecodeError as json_err:
            print(f"JSON Decode Error: {json_err}")
            return jsonify({
                "error": "Failed to parse AI-generated response",
                "details": str(json_err)
            }), 500
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        experience = int(request.form['experience'])
        education = encode_category(request.form['education'], education_levels)
        certification = encode_category(request.form['certification'], certifications)
        job_role = encode_category(request.form['job_role'], job_roles)
        salary_expectation = int(request.form['salary_expectation'])
        projects_count = int(request.form['projects_count'])
        ai_score = int(request.form['ai_score'])
        skill1 = encode_category(request.form['skill1'], skills)
        skill2 = encode_category(request.form['skill2'], skills)
        skill3 = encode_category(request.form['skill3'], skills)

        # Prepare input data in model format
        input_data = pd.DataFrame([{
            'ExperienceYears': experience,
            'Education': education,
            'Certifications': certification,
            'JobRole': job_role,
            'SalaryExpectation': salary_expectation,
            'ProjectsCount': projects_count,
            'AIScore': ai_score,
            'Skill1': skill1,
            'Skill2': skill2,
            'Skill3': skill3
        }])

        # Predict using the CatBoost model
     
        prediction = model.predict(input_data)[0]
        
        
        if prediction == 0:
            prediction = 'Hire'
        else:
            prediction = 'Reject'
 
        return render_template('Resume_Analyzer.html', prediction=prediction, education_levels=education_levels, certifications=certifications, job_roles=job_roles, skills=skills)

@app.route('/jobs', methods=['GET'])
def get_google_jobs():
    # Get query parameters with defaults
    query = request.args.get('q', 'software engineer')
    location = request.args.get('location', 'new york')
    
    # SerpAPI parameters
    params = {
        "engine": "google_jobs",
        "q": f"{query} {location}",
        "hl": "en",
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    
    try:
        # Perform the search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Filter and format job results
        filtered_jobs = []
        for job in results.get('jobs_results', [])[:9]:  # Limit to 9 jobs
            filtered_job = {
                'job_title': job.get('title', 'N/A'),
                'company_name': job.get('company_name', 'N/A'),
                'location': job.get('location', 'N/A'),
                'source': job.get('via', 'N/A'),
                'posting_age': job.get('extensions', [None])[0] if job.get('extensions') else 'N/A',
                'job_type': job.get('detected_extensions', {}).get('schedule_type', 'N/A'),
                'benefits': {
                    'paid_time_off': job.get('detected_extensions', {}).get('paid_time_off', False),
                    'health_insurance': job.get('detected_extensions', {}).get('health_insurance', False),
                    'dental_coverage': job.get('detected_extensions', {}).get('dental_coverage', False)
                },
                'description_preview': job.get('description', 'N/A')[:200] + '...' if job.get('description') else 'N/A',
                'apply_options': [
                    {
                        'platform': option.get('title', 'N/A'),
                        'apply_link': option.get('link', 'N/A')
                    } for option in job.get('apply_options', [])
                ]
            }
            filtered_jobs.append(filtered_job)
        
        return jsonify({
            'total_jobs': len(filtered_jobs),
            'jobs': filtered_jobs
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to fetch job results'
        }), 500
    

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    target_company = request.form.get('target_company', '')
    target_role = request.form.get('target_role', '')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            resume_text = extract_text_from_pdf(file_path)
            
            if not resume_text.strip():
                return jsonify({"error": "Extracted text is empty. Ensure the PDF contains readable text."}), 400
            
            prompt = f"""
            Perform a comprehensive analysis of the resume with specific focus on the target company {target_company} and role {target_role}.
            Resume Content:
            {resume_text}
            
            Provide a detailed JSON response with the following:
            1. **Detailed Resume Extraction**:
            - ExperienceYears: Total years of professional experience
            - Education: Highest educational qualification
            - Certifications: Relevant professional certifications
            - JobRole: Current or target professional role
            - SalaryExpectation: Expected compensation
            - ProjectsCount: Number of significant projects
            - AIScore: Comprehensive resume quality score (0-100)
            - Skill1, Skill2, Skill3: Top technical and soft skills
            
            2. **Company-Specific Recommendations**:
            Generate a comprehensive, structured roadmap for improving the candidate's profile specifically for {target_role} at {target_company}
            
            3. **Structured Recommendations Format**:
            Provide recommendations as a nested JSON object with the following structure:
            - TechnicalSkillGaps: List of missing technical skills
            - DSARecommendations: 
                * ImportanceLevel: Numeric score of DSA importance
                * ProblemDifficulty: Recommended problem-solving levels
                * StudyResources: Top recommended learning platforms
            - ProjectEnhancement: 
                * MissingProjectTypes: Types of projects to build
                * PortfolioStrategy: Specific project recommendations
            - SoftSkillDevelopment: Key areas of improvement
            - CareerRoadmap: 
                * ShortTermGoals: 6-month development plan
                * LongTermGoals: 2-year career progression
            - AdditionalCredentials: 
                * Certifications: Recommended certifications
                * Courses: Online learning suggestions
            
            Ensure recommendations are:
            - Specific to {target_role} at {target_company}
            - Actionable and motivational
            - Aligned with industry best practices
            
            Return ONLY a valid, comprehensive JSON object capturing all requested details.



            give me all the possible recommendations for the resume 

            4. Improve the resume quality by providing the following details:
            - Then inside this provide me all the possible recommendations for the resume that has provided that how can it be improved and whad extra can also be done
            """
            
            # Prepare API payload
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            # Send request to AI API
            headers = {"Content-Type": "application/json"}
            try:
                response = requests.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                
                # Extract and parse response
                response_data = response.json()
                generated_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                
                # Clean and parse JSON
                generated_text = generated_text.strip("```json").strip("```")
                parsed_json = json.loads(generated_text)
                
                # Validate and structure the response
                result = {
                    # Basic Resume Details
                    "ExperienceYears": parsed_json.get("ExperienceYears", 0),
                    "Education": parsed_json.get("Education", "Not provided"),
                    "Certifications": parsed_json.get("Certifications", []),
                    "JobRole": parsed_json.get("JobRole", "Not provided"),
                    "SalaryExpectation": parsed_json.get("SalaryExpectation", "Not provided"),
                    "ProjectsCount": parsed_json.get("ProjectsCount", 0),
                    "AIScore": parsed_json.get("AIScore", 0),
                    
                    # Top Skills
                    "Skill1": parsed_json.get("Skill1", "N/A"),
                    "Skill2": parsed_json.get("Skill2", "N/A"),
                    "Skill3": parsed_json.get("Skill3", "N/A"),
                    
                    # Detailed Recommendations
                    "TechnicalSkillGaps": parsed_json.get("TechnicalSkillGaps", []),
                    "DSARecommendations": parsed_json.get("DSARecommendations", {}),
                    "ProjectEnhancement": parsed_json.get("ProjectEnhancement", {}),
                    "SoftSkillDevelopment": parsed_json.get("SoftSkillDevelopment", []),
                    "CareerRoadmap": parsed_json.get("CareerRoadmap", {}),
                    "AdditionalCredentials": parsed_json.get("AdditionalCredentials", {})
                }
                
                return jsonify(result)
            
            except json.JSONDecodeError as json_err:
                print(f"JSON Decode Error: {json_err}")
                return jsonify({
                    "error": "Failed to parse AI-generated response",
                    "details": str(json_err)
                }), 500
            
            except Exception as api_err:
                print(f"API Request Error: {api_err}")
                return jsonify({
                    "error": "Error processing AI recommendations",
                    "details": str(api_err)
                }), 500
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400


# Routes For Interview Components
@app.route('/setup-interview', methods=['POST'])
def setup_interview():
    data = request.json
    if not data or 'name' not in data or 'role' not in data:
        return jsonify({"error": "Name and role are required"}), 400
    
    name = data['name']
    role = data['role']
    
    # Reset the chat with new system prompt
    system_prompt = create_system_prompt(name, role)
    with open('database.json', 'w') as f:
        json.dump([system_prompt], f)
    
    return jsonify({"status": "success", "message": f"Interview setup for {name} as {role}"})

@app.route('/start-interview', methods=['POST'])
def start_interview():
    messages = load_messages()
    if not messages:
        return jsonify({"error": "Please setup interview first with name and role"}), 400
        
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=messages,
    )
    response_text = gemini_response.candidates[0].content.parts[0].text
    response_message = {"role": "model", "parts": [{"text": response_text}]}
    save_message(response_message)
    
    audio_filename = f"response_{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static/audio", audio_filename)
    text_to_speech(response_text, audio_path)
    
    return jsonify({
        "response": response_text,
        "audio_url": url_for('static', filename=f'audio/{audio_filename}')
    })

# [Rest of your existing routes remain largely the same, just updating load_messages]
@app.route('/talk', methods=['POST'])
def post_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    transcript_text = transcribe_audio(file)
    user_message = {"role": "user", "parts": [{"text": transcript_text}]}
    chat_response = get_chat_response(user_message) 
    text_content = chat_response["parts"][0]["text"]
    
    audio_filename = f"response_{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static/audio", audio_filename)
    text_to_speech(text_content, audio_path)
    
    return jsonify({
        "transcription": transcript_text,
        "response": text_content,
        "audio_url": url_for('static', filename=f'audio/{audio_filename}')
    })

@app.route('/text-input', methods=['POST'])
def text_input():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = {"role": "user", "parts": [{"text": data['message']}]}
    chat_response = get_chat_response(user_message)
    text_content = chat_response["parts"][0]["text"]
    
    audio_filename = f"response_{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static/audio", audio_filename)
    text_to_speech(text_content, audio_path)
    
    return jsonify({
        "response": text_content,
        "audio_url": url_for('static', filename=f'audio/{audio_filename}')
    })

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    messages = load_messages()
    if messages:
        return jsonify(messages[1:])  # Skip system prompt
    return jsonify([])

@app.route('/reset-chat', methods=['POST'])
def reset_chat():
    return jsonify({"status": "success", "message": "Please setup new interview with name and role"})

# Functions
def transcribe_audio(file):
    filepath = f"./{file.filename}"
    file.save(filepath)
    transcript = Whisper_Model.transcribe(filepath)
    os.remove(filepath)
    return transcript["text"]

def get_chat_response(user_message):
    messages = load_messages()
    if not messages:
        return {"role": "model", "parts": [{"text": "Please setup interview first with name and role"}]}
    
    messages.append(user_message)
    
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=messages,
    )
    
    response_message = {
        "role": "model",
        "parts": [{"text": gemini_response.text}]
    }
    
    save_message(user_message)
    save_message(response_message)
    
    return response_message  

def load_messages():
    file = 'database.json'
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        return []
    
    with open(file) as db_file:
        messages = json.load(db_file)
    return messages

def save_message(message):
    messages = load_messages()
    messages.append(message)
    
    with open('database.json', 'w') as f:
        json.dump(messages, f)

def text_to_speech(text, output_path=None):
    try:
        tts = gTTS(text=text, lang="en")
        
        if output_path:
            tts.save(output_path)
            return output_path
        else:
            filename = "output.mp3"
            tts.save(filename)
            
            with open(filename, "rb") as audio_file:
                audio_content = audio_file.read()
            
            os.remove(filename)
            return audio_content
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    app.run(debug=True)
