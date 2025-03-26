from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from dotenv import load_dotenv
import whisper
import os
import numpy as np
import json

load_dotenv()

# Load the Whisper model
model = whisper.load_model("base")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
)

@app.get('/')
async def root():
    return {"message": "Hello, World!"}

@app.post('/talk')
async def post_audio(file: UploadFile):
    user_message = transcribe_audio(file)
    return {"transcription": user_message}

#Functions
def transcribe_audio(file):
    print(file.file.name)
    with open(f"./{file.filename}", "wb") as f:
        f.write(file.file.read())

    with open(f"./{file.filename}", "rb") as f:
        data = f.read()
    
    transcript = model.transcribe(file.filename)

    return {"transcript": transcript}


if __name__ == "__main__":
    app.run(host="0.0.0.0")
    