import requests
import sounddevice as sd
import soundfile as sf
from io import BytesIO

def chat_and_play_audio(message, language_code='a'):
    # Make request to chat-tts endpoint
    response = requests.post(
        "http://localhost:8000/chat-tts",
        json={
            "message": message,
            "language_code": language_code,
            "max_new_tokens": 500
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("LLM Response:", data["response_text"])
        
        # Get the audio file
        audio_response = requests.get(f"http://localhost:8000{data['audio_url']}")
        
        if audio_response.status_code == 200:
            # Load audio data
            audio_data, sample_rate = sf.read(BytesIO(audio_response.content))
            
            # Play the audio
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait until audio is finished playing
        else:
            print(f"Error getting audio: {audio_response.status_code}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test the API
    message = input("Enter your message: ")
    chat_and_play_audio(message) 