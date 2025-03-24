from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

app = Flask(__name__)

# Load the trained CatBoost model
model_path = "best_resume_model.pkl"  # Ensure this file is in your project folder
model = pickle.load(open(model_path, "rb"))

# Define categories for categorical features
education_levels = ['B.Sc', 'B.Tech', 'MBA', 'PhD', 'M.Tech']
certifications = ['None', 'Deep Learning Specialization', 'AWS Certified', 'Google ML']
job_roles = ['AI Researcher', 'Cybersecurity Analyst', 'Data Scientist', 'Software Engineer']
skills = ['sql', 'python', 'tensorflow', 'pytorch', 'cybersecurity', 'ethical hacking', 'networking', 'java', 'react', 'deep learning', 'c', 'machine learning', 'linux', 'nlp']

@app.route('/')
def home():
    return render_template('index.html', education_levels=education_levels, certifications=certifications, job_roles=job_roles, skills=skills)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        experience = int(request.form['experience'])
        education = request.form['education']
        certification = request.form['certification']
        job_role = request.form['job_role']
        salary_expectation = int(request.form['salary_expectation'])
        projects_count = int(request.form['projects_count'])
        ai_score = int(request.form['ai_score'])
        skill1 = request.form['skill1']
        skill2 = request.form['skill2']
        skill3 = request.form['skill3']

        # Prepare input data in model format (matching the updated column names)
        input_data = pd.DataFrame([{
            'Experience (Years)': experience,
            'Education': education,
            'Certifications': certification,
            'Job Role': job_role,
            'Salary Expectation ($)': salary_expectation,
            'Projects Count': projects_count,
            'AI Score (0-100)': ai_score,
            'Skill1': skill1,
            'Skill2': skill2,
            'Skill3': skill3
        }])

        # Predict using the CatBoost model
        prediction = model.predict(input_data)
        print(prediction)  # 'Hire' or 'Reject'

        return render_template('index.html', prediction=prediction, education_levels=education_levels, certifications=certifications, job_roles=job_roles, skills=skills)

if __name__ == '__main__':
    app.run(debug=True)
