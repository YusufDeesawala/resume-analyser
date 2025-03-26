from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from serpapi import GoogleSearch
from catboost import CatBoostClassifier

app = Flask(__name__)

# Load the trained CatBoost model
model_path = "models/catboost_model_1.pkl"  # Ensure this file is in your project folder
model = pickle.load(open(model_path, "rb"))

# Define categories for categorical features (ensure same order as training)
education_levels = ['B.Sc', 'B.Tech', 'MBA', 'PhD', 'M.Tech']
certifications = ['None', 'Deep Learning Specialization', 'AWS Certified', 'Google ML']
job_roles = ['AI Researcher', 'Cybersecurity Analyst', 'Data Scientist', 'Software Engineer']
skills = ['sql', 'python', 'tensorflow', 'pytorch', 'cybersecurity', 'ethical hacking', 'networking', 'java', 'react', 'deep learning', 'c', 'machine learning', 'linux', 'nlp']

# Function to encode categorical values the same way as training
def encode_category(value, category_list):
    return category_list.index(value) if value in category_list else -1  # -1 for unseen values

@app.route('/')
def home():
    return render_template('index.html', education_levels=education_levels, certifications=certifications, job_roles=job_roles, skills=skills)

@app.route('/job-search')
def job():
    return render_template('Job.html')

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
        
        
        if prediction == 1:
            prediction = 'Hire'
        else:
            prediction = 'Reject'
 
        return render_template('index.html', prediction=prediction, education_levels=education_levels, certifications=certifications, job_roles=job_roles, skills=skills)

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
        "api_key": "cc27bf84cc4bbe79fbf83665e2858f342355264165984777af0b8b009d9943dd"
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

if __name__ == '__main__':
    app.run(debug=True)
