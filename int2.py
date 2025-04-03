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

# System prompt definition
SYSTEM_PROMPT = {
    'role': 'model', 
    "parts": [{"text":"You are interviewing the user for the Machine Learning Developer position. Ask relevant questions, if the user is able to answer that question correctly then proceed to increase the difficulty of the questions. Your name is Alex and you will refer to the user with the name resource."}]
}

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
    
    # Generate audio file
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
    
    # Generate audio file
    audio_filename = f"response_{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static/audio", audio_filename)
    text_to_speech(text_content, audio_path)
    
    return jsonify({
        "response": text_content,
        "audio_url": url_for('static', filename=f'audio/{audio_filename}')
    })

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    # Get all messages except the system prompt
    messages = load_messages()
    if messages and messages[0]['role'] == 'model' and "You are interviewing the user" in messages[0]['parts'][0]['text']:
        # Skip the system message in the displayed history
        return jsonify(messages[1:])
    return jsonify(messages)

@app.route('/reset-chat', methods=['POST'])
def reset_chat():
    # Reset to just the system prompt
    with open('database.json', 'w') as f:
        json.dump([SYSTEM_PROMPT], f)
        
    return jsonify({"status": "success"})

@app.route('/start-interview', methods=['POST'])
def start_interview():
    messages = load_messages()
    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=messages,  # Just system prompt initially
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
        # Initialize with system prompt
        with open(file, 'w') as f:
            json.dump([SYSTEM_PROMPT], f)
        return [SYSTEM_PROMPT]
    
    with open(file) as db_file:
        messages = json.load(db_file)
        
    # If first message is not the system prompt, add it
    if not messages or messages[0]['role'] != 'model' or "You are interviewing the user" not in messages[0]['parts'][0]['text']:
        messages.insert(0, SYSTEM_PROMPT)
        # Save the updated messages
        with open(file, 'w') as f:
            json.dump(messages, f)
    
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