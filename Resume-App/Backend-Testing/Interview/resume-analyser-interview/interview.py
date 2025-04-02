from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from dotenv import load_dotenv
from google import genai
import whisper
import os
import numpy as np
import json

load_dotenv()

# Load the Whisper model
model = whisper.load_model("base")
client = genai.Client(api_key="AIzaSyCwFzKMYFOQG__sr86s2bDJ9ZmIdzlUTVw")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware
)

@app.post('/talk')
async def post_audio(file: UploadFile):
    user_message = transcribe_audio(file)
    chat_response = get_chat_response(user_message)
    print(chat_response)

    return {"transcription": user_message}

#Functions
def transcribe_audio(file):
    with open(f"./{file.filename}", "wb") as f:
        f.write(file.file.read())

    with open(f"./{file.filename}", "rb") as f:
        data = f.read()
    
    transcript = model.transcribe(file.filename)

    return {"role" : "user" , "parts": [{"text" : transcript["text"]}]}

def get_chat_response(user_message):
    messages = load_messages()
    messages.append(user_message)
    print(messages)
    
    gemini_response = client.models.generate_content(
        model = "gemini-2.0-flash",
        contents = messages,
    )
    
    # Create a proper response message from the Gemini response
    response_message = {
        "role": "model",
        "parts": [{"text": gemini_response.text}]  # Use the .text property
    }
    
    save_messages(user_message, response_message)
    return response_message  # Return the formatted response


def load_messages():
    messages = []
    file = 'database.json'

    empty = os.stat(file).st_size == 0 

    if not empty:
        with open(file) as db_file:
            data = json.load(db_file)
            for item in data:
                messages.append(item)
    else:
        messages.append(
            {'role': 'model', "parts": [{"text":"You are interviewing the user for the Machine Learning Developer position. Ask relevant questions, if the user is able to answer that question correctly then proceed to increase the difficulty of the questions. Your name is Alex and you will refer to the user with the name resource."}]}
        )
    return messages

def save_messages(user_message, model_response):
    file = 'database.json'
    messages = load_messages()
    messages.append(user_message)
    messages.append(model_response)  # Add the properly formatted response
    
    # Write the entire conversation history (not append)
    with open(file, 'w') as f:  # Note: 'w' instead of 'a'
        json.dump(messages, f)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
    