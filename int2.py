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
    transcript_text = transcribe_audio(file)
    user_message = {"role": "user", "parts": [{"text": transcript_text}]}
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
    return transcript["text"]

def get_chat_response(user_message):
    messages = load_messages()
    
    # Add the user message to conversation history
    messages.append(user_message)
    
    # Get response from Gemini
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=messages,
    )
    
    # Create response message
    response_message = {
        "role": "model",
        "parts": [{"text": gemini_response.text}]
    }
    
    # Save both messages to database
    save_message(user_message)
    save_message(response_message)
    
    return response_message  

def load_messages():
    file = 'database.json'
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        initial_message = {'role': 'model', "parts": [{"text":"You are interviewing the user for the Machine Learning Developer position. Ask relevant questions, if the user is able to answer that question correctly then proceed to increase the difficulty of the questions. Your name is Alex and you will refer to the user with the name resource."}]}
        with open(file, 'w') as f:
            json.dump([initial_message], f)
        return [initial_message]
    
    with open(file) as db_file:
        return json.load(db_file)

def save_message(message):
    messages = load_messages()
    messages.append(message)
    
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