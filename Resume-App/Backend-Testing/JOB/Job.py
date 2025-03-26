# Serp API
from serpapi import GoogleSearch

params = {
  "engine": "google_jobs",
  "q": "barista new york",
  "hl": "en",
  "api_key": "cc27bf84cc4bbe79fbf83665e2858f342355264165984777af0b8b009d9943dd"
}

search = GoogleSearch(params)
results = search.get_dict()
jobs_results = results["jobs_results"]
print(jobs_results)

# Rapid API
"""import requests

url = "https://upwork-jobs-api2.p.rapidapi.com/active-freelance-7d"

querystring = {"search":"\"Data Engineer\"","location_filter":"\"United States\""}

headers = {
	"x-rapidapi-key": "cdb738cfdemsh0e73ab99eafa4a0p1e14bfjsnc767eb470c88",
	"x-rapidapi-host": "upwork-jobs-api2.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())"""