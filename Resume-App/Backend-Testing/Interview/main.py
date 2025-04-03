from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from gtts import gTTS
from dotenv import load_dotenv
from google import genai
import whisper
import os
import json
import base64
import io
import edge_tts
import asyncio

load_dotenv()

# Load the Whisper model
model = whisper.load_model("base")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__, template_folder="templates")
CORS(app)

# Session configuration
user_profile = {
    "name": "",
    "position": "",
    "experience": ""
}

@app.route('/')
def index():
    # Reset user profile when loading the page
    global user_profile
    user_profile = {"name": "", "position": "", "experience": ""}
    
    # Clear previous conversation
    if os.path.exists('database.json'):
        os.remove('database.json')
    
    return render_template("index.html")

@app.route('/initialize', methods=['POST'])
def initialize_interview():
    global user_profile
    data = request.json
    
    # Update user profile
    user_profile["name"] = data.get('name', '')
    user_profile["position"] = data.get('position', '')
    user_profile["experience"] = data.get('experience', '')
    
    # Create initial system prompt based on the position
    system_prompt = create_system_prompt(
        user_profile["name"], 
        user_profile["position"], 
        user_profile["experience"]
    )
    
    # Store the system prompt as first message
    init_message = {'role': 'model', "parts": [{"text": system_prompt}]}
    with open('database.json', 'w') as f:
        json.dump([init_message], f)
    
    # Generate welcome message
    welcome_message = f"Hello {user_profile['name']}! I'm Alex, your AI interviewer. I'll be asking you questions about {user_profile['position']} at the {user_profile['experience']} level. Let's get started with the first question."
    
    # Add this welcome message to the chat history
    user_message = {"role": "user", "parts": [{"text": "Let's start the interview"}]}
    response_message = {"role": "model", "parts": [{"text": welcome_message}]}
    save_messages(user_message, response_message)
    
    # Generate audio for the welcome message
    audio_output = text_to_speech(welcome_message)
    audio_base64 = base64.b64encode(audio_output).decode('utf-8')
    
    # Store this as the last response
    store_last_response(welcome_message)
    
    # Return both text and audio
    return jsonify({
        "text": welcome_message,
        "audio": audio_base64
    })

@app.route('/talk', methods=['POST'])
def post_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    user_message = transcribe_audio(file)
    chat_response = get_chat_response(user_message) 
    text_content = chat_response["parts"][0]["text"]
    audio_output = text_to_speech(text_content)
    
    # Store the last response for transcript retrieval
    store_last_response(text_content)
    
    return Response(audio_output, mimetype="audio/mp3")

@app.route('/chat', methods=['POST'])
def post_chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = {"role": "user", "parts": [{"text": data['message']}]}
    chat_response = get_chat_response(user_message)
    text_content = chat_response["parts"][0]["text"]
    
    # Generate audio response
    audio_output = text_to_speech(text_content)
    
    # Encode audio as base64 to include in JSON response
    audio_base64 = base64.b64encode(audio_output).decode('utf-8')
    
    # Store the last response
    store_last_response(text_content)
    
    return jsonify({
        "text": text_content,
        "audio": audio_base64
    })

@app.route('/last_response', methods=['GET'])
def get_last_response():
    if os.path.exists('last_response.txt'):
        with open('last_response.txt', 'r') as f:
            text = f.read()
        return jsonify({"text": text})
    else:
        return jsonify({"text": "No response available"})

# Functions
def create_system_prompt(name, position, experience):
    """Create a tailored system prompt based on the position and experience level"""
    
    # Base prompt template
    base_prompt = f"You are interviewing {name} for the {position} position at the {experience} level. Ask relevant questions that match this experience level, if they answer correctly, gradually increase the difficulty. Your name is Alex and you will refer to the user by their name ({name})."
    
    # Position-specific additions
    position_prompts = {
        "Machine Learning Engineer": "Focus on machine learning algorithms, model deployment, feature engineering, and ML frameworks like TensorFlow and PyTorch. For entry level, focus on fundamentals; for senior level, include MLOps questions.",
        
        "Software Engineer": "Focus on data structures, algorithms, system design, and coding practices. Include language-specific questions about Python, Java, or C++ depending on responses. For senior levels, include architecture discussions.",
        
        "Data Scientist": "Focus on statistical methods, data analysis, visualization techniques, and predictive modeling. For entry level, focus on basic statistics; for senior level, include causal inference and experimental design.",
        
        "Frontend Developer": "Focus on HTML, CSS, JavaScript, frontend frameworks (React, Vue, Angular), responsive design, and web performance. For senior levels, include advanced state management and architecture patterns.",
        
        "Backend Developer": "Focus on API design, database optimization, server architecture, caching strategies, and security. For senior levels, include system design and microservices architecture.",
        
        "DevOps Engineer": "Focus on CI/CD pipelines, infrastructure as code, containerization, monitoring, and cloud services. For senior levels, include multi-cloud strategies and security practices.",
        
        "Mobile Developer": "Focus on mobile app architecture, platform-specific guidelines (iOS/Android), performance optimization, and user experience. For senior levels, include cross-platform considerations.",
        
        "Full Stack Developer": "Cover both frontend and backend topics, API integration, full-stack architecture patterns, and deployment strategies. For senior levels, include system design challenges."
    }
    
    # Experience level modifications
    experience_modifications = {
        "Entry Level": "Start with fundamental concepts and gradually move to basic application of those concepts. Focus on theoretical knowledge and simple practical problems.",
        
        "Mid Level": "Begin with practical implementation questions and move to design considerations. Include questions about debugging, optimization, and working within teams.",
        
        "Senior": "Start with complex design problems, architectural decisions, and team leadership scenarios. Include questions about mentoring, technical decision-making, and handling trade-offs."
    }
    
    # Compile the final prompt
    final_prompt = base_prompt
    
    if position in position_prompts:
        final_prompt += " " + position_prompts[position]
    
    if experience in experience_modifications:
        final_prompt += " " + experience_modifications[experience]
    
    final_prompt += " Conduct a professional interview with thoughtful follow-up questions. If the candidate struggles, provide gentle guidance without giving away the answers."
    
    return final_prompt

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
        # Default system prompt if no prior conversation exists
        return [{'role': 'model', "parts": [{"text":"You are interviewing the user for a technical position. Ask relevant questions based on their skill level, increasing difficulty as appropriate. Your name is Alex."}]}]
    
    with open(file) as db_file:
        return json.load(db_file)

def save_messages(user_message, model_response):
    messages = load_messages()
    messages.append(user_message)
    messages.append(model_response)  
    
    with open('database.json', 'w') as f:
        json.dump(messages, f)

def store_last_response(text):
    with open('last_response.txt', 'w') as f:
        f.write(text)

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en")
        output = io.BytesIO()
        tts.write_to_fp(output)
        output.seek(0)
        return output.read()
    except Exception as e:
        print(f"Error: {e}")
        return

def text_to_speech_gemini(text):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=text,
            config=genai.types.GenerateContentConfig(
                voice="en-US-JennyNeural",
                audio_format="mp3"
            )
        )
        audio_data = response.audio
        return audio_data
    except Exception as e:
        print(f"Error: {e}")
        return None

def text_to_speech_edgetts(text, voice="en-GB-SoniaNeural"):
    try:
        # Creating an asynchronous wrapper to handle the edge_tts logic
        async def wrapper():
            communicator = et.Communicate(text, voice)
            audio_data = await communicator.stream()  # This will return audio data as a stream

            # Convert the audio stream to BytesIO
            output = io.BytesIO()
            output.write(audio_data)
            output.seek(0)  # Rewind to the beginning of the BytesIO object

            return output

        # Run the asynchronous code in an event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(wrapper())
        loop.close()

        return result
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def text_to_speech_edge(text):
    try:
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        output = loop.run_until_complete(communicate.save("output.mp3"))
        with open("output.mp3", "rb") as audio_file:
            audio_data = audio_file.read()
        os.remove("output.mp3")  # Clean up the saved file
        return audio_data
    except Exception as e:
        print(f"Error: {e}")
        return None
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)