from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import soundfile as sf
import tempfile
import os
import logging
import numpy as np
from kokoro import KPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from datetime import datetime
import uuid
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure audio storage
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_audio")
MAX_AUDIO_FILES = 100  # Maximum number of audio files to keep

# Create audio directory if it doesn't exist
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI(title="Kokoro Chat TTS API", description="API for chat and text-to-speech generation")

# Initialize TTS pipelines for different languages
try:
    pipelines = {
        'a': KPipeline(lang_code='a'),
        'b': KPipeline(lang_code='b')
    }
    logger.debug("Successfully initialized Kokoro pipelines")
except Exception as e:
    logger.error(f"Failed to initialize TTS pipelines: {str(e)}")
    raise

# Initialize LLM model and tokenizer
try:
    logger.debug("Loading LLM model and tokenizer...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "nicoboss/Llama-3.2-1B-Instruct-Uncensored",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("nicoboss/Llama-3.2-1B-Instruct-Uncensored")
    
    # Initialize the pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    logger.debug("Successfully loaded LLM model and pipeline")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

class ChatRequest(BaseModel):
    message: str
    language_code: str = 'a'  
    max_new_tokens: int = 500
    voice: str = 'af_heart'  

def cleanup_old_files():
    """Remove old audio files if we exceed MAX_AUDIO_FILES"""
    try:
        files = [(f, os.path.getmtime(os.path.join(AUDIO_DIR, f))) 
                 for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
        
        if len(files) > MAX_AUDIO_FILES:
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])
            
            # Remove oldest files
            for f, _ in files[:(len(files) - MAX_AUDIO_FILES)]:
                os.remove(os.path.join(AUDIO_DIR, f))
                logger.debug(f"Cleaned up old audio file: {f}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def chat_with_llm(message: str, max_new_tokens: int = 500) -> str:
    """Generate response using the LLM model."""
    try:
        logger.debug(f"Generating LLM response for message: {message}")
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": message}
        ]

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        output = llm_pipeline(messages, **generation_args)
        response = output[0]['generated_text']
        
        logger.debug(f"Generated response: {response}")
        return response
    
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise

@app.get("/")
async def root():
    logger.debug("Root endpoint accessed")
    return {
        "message": "Welcome to Kokoro Chat TTS API",
        "available_languages": list(pipelines.keys())
    }

@app.post("/chat-tts")
async def chat_and_generate_speech(request: ChatRequest):
    try:
        logger.debug(f"Received chat request: {request.message}")
        
        # Generate LLM response
        llm_response = chat_with_llm(request.message, request.max_new_tokens)
        
        # Generate speech from LLM response
        generator = pipelines[request.language_code](llm_response, request.voice, speed=1, split_pattern=r'\n+')
        
        # Process the generator output
        try:
            # Collect all audio segments
            all_segments = []
            for i, (graphemes, phonemes, audio_data) in enumerate(generator):
                logger.debug(f"Processing segment {i}: graphemes={graphemes}, phonemes={phonemes}")
                all_segments.append(audio_data)
            
            # Concatenate all audio segments
            combined_audio = np.concatenate(all_segments)
            logger.debug(f"Combined {len(all_segments)} audio segments")
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"audio_{timestamp}_{unique_id}.wav"
            output_path = os.path.join(AUDIO_DIR, filename)
            
            # Save the combined audio
            logger.debug(f"Saving combined audio to file: {output_path}")
            sf.write(output_path, combined_audio, 24000)
            logger.debug("Audio saved successfully")
            
            # Clean up old files if needed
            cleanup_old_files()
        
        except Exception as e:
            logger.error(f"Error processing audio data: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing audio data: {str(e)}")
        
        # Return both the text response and audio file
        return JSONResponse({
            "response_text": llm_response,
            "audio_url": f"/audio/{filename}"
        })
        
    except Exception as e:
        logger.error(f"Error in chat_and_generate_speech: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve the generated audio files."""
    try:
        file_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(
            file_path,
            media_type="audio/wav",
            filename="generated_speech.wav"
        )
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Create audio directory on startup"""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    logger.info(f"Audio directory initialized at: {AUDIO_DIR}")

if __name__ == "__main__":
    import uvicorn
    
    # Run with debug mode enabled
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="error"
    ) 