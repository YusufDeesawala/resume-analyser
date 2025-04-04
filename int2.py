from flask import Flask, request, jsonify, Response, render_template, send_file, url_for
from flask_cors import CORS
from gtts import gTTS
from dotenv import load_dotenv
from google import genai
import whisper
import os
import json
import uuid

load_dotenv()

# Load the Whisper model
model = whisper.load_model("base")
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

app = Flask(__name__, template_folder="templates")
CORS(app)

# Create directory for audio files if it doesn't exist
os.makedirs("static/audio", exist_ok=True)

# Function to create system prompt with dynamic name and role
def create_system_prompt(name, role):
    return {
        'role': 'model',
        "parts": [{
            "text": f"You are interviewing {name} for the {role} position. Ask relevant questions, if the user is able to answer that question correctly then proceed to increase the difficulty of the questions. Your name is Alex and you will refer to the user as {name}."
        }]
    }

@app.route('/')
def index():
    return render_template("index.html")

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
    transcript = model.transcribe(filepath)
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)