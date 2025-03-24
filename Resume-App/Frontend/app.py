from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
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
        prediction = model.predict(input_data)[0]
        
        
        if prediction == 1:
            prediction = 'Hire'
        else:
            prediction = 'Reject'
 
        return render_template('index.html', prediction=prediction, education_levels=education_levels, certifications=certifications, job_roles=job_roles, skills=skills)

if __name__ == '__main__':
    app.run(debug=True)
