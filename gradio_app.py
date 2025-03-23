import gradio as gr
import requests
import numpy as np
import soundfile as sf
from io import BytesIO
import os

# FastAPI backend URL from environment variable, fallback to localhost if not set
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def chat_and_play(message, language_code="a", voice="af_heart"):
    """Send request to FastAPI backend and process response"""
    try:
        # Make request to chat-tts endpoint
        response = requests.post(
            f"{BACKEND_URL}/chat-tts",
            json={
                "message": message,
                "language_code": language_code,
                "voice": voice,
                "max_new_tokens": 500
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            text_response = data["response_text"]
            
            # Get the audio file
            audio_response = requests.get(f"{BACKEND_URL}{data['audio_url']}")
            
            if audio_response.status_code == 200:
                # Convert audio bytes to numpy array
                audio_data, sample_rate = sf.read(BytesIO(audio_response.content))
                return text_response, (sample_rate, audio_data)
            else:
                return f"Error getting audio: {audio_response.status_code}", None
        else:
            return f"Error: {response.status_code}\n{response.text}", None
            
    except Exception as e:
        return f"Error: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="Kokoro Chat TTS Demo") as demo:
    gr.Markdown("# 🎙️ Kokoro Chat TTS Demo")
    gr.Markdown("Chat with an AI and hear its responses!")
    
    with gr.Row():
        with gr.Column():
            # Input components
            message_input = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                lines=3
            )
            
            language_input = gr.Dropdown(
                choices=["a", "b"],
                value="a",
                label="Language"
            )
            
            voice_input = gr.Dropdown(
                choices=["af_heart", "af_bella"],
                value="af_heart",
                label="Voice"
            )
            
            submit_btn = gr.Button("Send", variant="primary")
        
        with gr.Column():
            # Output components
            text_output = gr.Textbox(
                label="AI Response",
                lines=5,
                interactive=False
            )
            audio_output = gr.Audio(
                label="Generated Speech",
                interactive=False,
                autoplay=True  # Enable autoplay
            )
    
    # Handle form submission
    submit_btn.click(
        fn=chat_and_play,
        inputs=[message_input, language_input, voice_input],
        outputs=[text_output, audio_output]
    )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["Tell me about artificial intelligence", "a", "af_heart"],
            ["What's the weather like today?", "a", "af_heart"],
            ["Tell me a short story", "a", "af_heart"]
        ],
        inputs=[message_input, language_input, voice_input]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Make accessible from other machines
        server_port=7860,       # Default Gradio port
        share=True             # Create a public link
    ) 