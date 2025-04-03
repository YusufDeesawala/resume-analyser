import io
import os

# Assuming the text_to_speech function is already implemented as shown in the previous response
from main import text_to_speech_edgetts
from main import sample


def test_text_to_speech():
    # Sample text to convert to speech
    sample_text = "Hello, welcome to this text-to-speech conversion test using edge_tts!"

    # Call the text_to_speech function
    audio_data = text_to_speech_edgetts(sample_text)

    # Check if the returned audio data is a valid BytesIO object
    if isinstance(audio_data, io.BytesIO):
        print("Test Passed: Audio data is in a BytesIO format.")

        # Optionally, save the audio to a file for manual verification
        with open("test_output.mp3", "wb") as file:
            file.write(audio_data.read())
        print("Test Passed: Audio saved as 'test_output.mp3'.")
    else:
        print("Test Failed: The returned data is not in the expected BytesIO format.")

def sample2(text):
    sample_text = text
    print(sample(sample_text))

if __name__ == "__main__":
    sample_text = "Hello, welcome to this text-to-speech conversion test using edge_tts!"
    sample2(sample_text)
    
