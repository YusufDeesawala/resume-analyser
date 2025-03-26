from flask import Flask, render_template, request, jsonify
from serpapi import GoogleSearch
import os


app = Flask(__name__)


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
    app.run(port=8000,debug=True)