from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from gtts import gTTS
from dotenv import load_dotenv
from google import genai
import whisper
import os
import json

load_dotenv()

# Load the Whisper model
model = whisper.load_model("base")
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
elevenlabs_key = os.getenv("ELEVENLABS_KEY")

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/talk', methods=['POST'])
def post_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    user_message = transcribe_audio(file)
    chat_response = get_chat_response(user_message) 
    text_content = chat_response["parts"][0]["text"]
    audio_output = text_to_speech(text_content)
    
    return Response(audio_output, mimetype="audio/mp3")

# Functions
def transcribe_audio(file):
    filepath = f"./{file.filename}"
    file.save(filepath)
    transcript = model.transcribe(filepath)
    os.remove(filepath)  # Clean up the saved file
    return {"role": "user", "parts": [{"text": transcript["text"]}]}

def get_chat_response(user_message):
    messages = load_messages()
    messages.append(user_message)
    
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=messages,
    )
    
    response_message = {
        "role": "model",
        "parts": [{"text": gemini_response.text}]
    }
    
    save_messages(user_message, response_message)
    return response_message  

def load_messages():
    file = 'database.json'
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        return [{'role': 'model', "parts": [{"text":"You are interviewing the user for the Machine Learning Developer position. Ask relevant questions, if the user is able to answer that question correctly then proceed to increase the difficulty of the questions. Your name is Alex and you will refer to the user with the name resource."}]}]
    
    with open(file) as db_file:
        return json.load(db_file)

def save_messages(user_message, model_response):
    messages = load_messages()
    messages.append(user_message)
    messages.append(model_response)  
    
    with open('database.json', 'w') as f:
        json.dump(messages, f)

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en")
        filename = "output.mp3"
        tts.save(filename)
        
        with open(filename, "rb") as audio_file:
            audio_content = audio_file.read()
        
        os.remove(filename)
        return audio_content
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
