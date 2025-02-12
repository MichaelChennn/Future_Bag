import sounddevice as sd
import numpy as np
import time as t
import whisper
from scipy.io.wavfile import write
import openai
import requests
import torch
from google.cloud import storage
from google.oauth2 import service_account     
import time
import re
import os
import shutil
from backgroundremover.bg import remove 

# Constants 
# The threshold of voice volume for detecting sound in the audio input
SILENCE_THRESHOLD = 2
# Minimum duration for valid voice input in seconds, if shorter than this, ignore the audio 
MIN_VOICE_DURATION = 2.5  
# Silence duration to trigger processing, if no sound detected in this duration, process the audio
SILENCE_GAP = 3.5    
# Sample rate for audio input and output
SAMPLE_RATE = 16000 
# Counter
counter = 1 
# lasted image 
lasted_image = None

# Google Cloud Storage credentials  
credentials = service_account.Credentials.from_service_account_file(
    "C:\Workspace\Future Bag\dalle_3\credentials.json"  
)
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # # The ID of your GCS bucket
    # bucket_name = "future_bag"
    # # The path to your file to upload
    # source_file_name = "generated_image.png"        
    # # The ID of your GCS object
    # destination_blob_name = "lasted_generated_image.png"    

    storage_client = storage.Client(credentials=credentials)    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    blob.cache_control = "no-cache, max-age=0"
    blob.patch()
    
upload_blob("future_bag", "advice_images/background.png", "lasted_generated_image.png")

# Create a directory to store images    
if not os.path.exists("images"):
    os.mkdir("images") 
if not os.path.exists("images_without_background"):
    os.mkdir("images_without_background")   
    
# choose device 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
# Whisper model loading
whisper_model = whisper.load_model("turbo", device=device, download_root="models") 
# Whisper model warming up  
whisper_model.transcribe("warmup.m4a")
print("Whisper model is ready!")            

def record_audio():
    """Record audio, process segments, and send transcriptions to the queue."""
    print("Demo version future bag project")    
    print("Listening for audio...")
    
    audio_buffer = []
    recording = False
    start_time = None
    last_sound_time = None

    def callback(indata, frames, time, status):
        nonlocal audio_buffer, recording, start_time, last_sound_time
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > SILENCE_THRESHOLD:
            if not recording:
                recording = True
                start_time = t.time()
                print("Heard sound, start recording...")
            last_sound_time = t.time()  # Update last sound timestamp
            audio_buffer.append(indata.copy())
        elif recording:
            current_time = t.time()
            silence_duration = current_time - last_sound_time if last_sound_time else 0
            if silence_duration > SILENCE_GAP:
                duration = current_time - start_time
                if duration >= MIN_VOICE_DURATION:
                    print("Stop recording...")  
                    process_audio(audio_buffer)  
                else:
                    print(f"Audio duration {duration} is too short, please try again.") 
                audio_buffer = []  # Clear buffer after processing
                recording = False
            else:
                audio_buffer.append(indata.copy())

    with sd.InputStream(callback=callback, samplerate= SAMPLE_RATE, channels=1):
        while True:
            t.sleep(0.1)
            
def process_audio(audio_buffer): 
    """Process buffered audio and save as voice.mp3."""
    global counter 
    global lasted_image
    audio_data = np.concatenate(audio_buffer, axis=0).astype(np.float32)
    # print("Processing audio...")
    try:
        # Save raw audio data to a WAV file
        wav_file = "tmp.wav"
        write(wav_file, SAMPLE_RATE, audio_data)
        # Transcribe audio using Whisper    
        result = whisper_model.transcribe(wav_file)
        transcription = result.get("text", "").strip()
        # if transcription:              
        if transcription and any(triggerword in transcription for triggerword in ["mystery bag", "mystery back", "mystery pack", "mystery box", "artefact"]):    
            upload_blob("future_bag", "advice_images/loading.png", "lasted_generated_image.png")           
            pattern = r'.*?\b(is|contains|future bag is|mystery bag is|mystery back is|mystery back contains|mystery bag contains)\b'               
            transcription = re.sub(pattern, '', transcription, count=1).strip().replace(".", "")    
            if counter == 1:
                upload_blob("future_bag", "demo_images/pink radio.png", "lasted_generated_image.png")  
                print("Success upload image pink radio!")   
                t.sleep(5)
                counter = 2
                lasted_image = "pink radio.png"     
            elif counter == 2: 
                upload_blob("future_bag", "demo_images/holly radio.png", "lasted_generated_image.png")
                print("Success upload image holly radio!")  
                t.sleep(5)
                counter = 3
                lasted_image = "holly radio.png"    
            elif counter == 3:  
                upload_blob("future_bag", "demo_images/3D printer.png", "lasted_generated_image.png")  
                print("Success upload image 3D printer!")     
                t.sleep(5)  
                counter = 4
                lasted_image = "3D printer.png" 
        else:
            print(f"Invalid transcription \"{transcription}\". Please try again.")
            # upload_blob("future_bag", "advice_images/try again.png", "lasted_generated_image.png")
            # t.sleep(2.5)    
            # upload_blob("future_bag", f"advice_images/{lasted_image}", "lasted_generated_image.png")     
    except Exception as e:
        print(f"Error during audio processing: {e}")
        
if __name__ == "__main__":
    record_audio()  
    # The mystery bag is a pink and blue fade radio with an antenna from the future
    # I think it is very light, easy to carry. It is hollow. Maybe it has a solar panel. I can use it for two-way communication from anywhere around the world to keep in touch with my friends and family.
    # This is a portable lightweight 3D printer that also can act as a radio. It has an antenna and I can print food with it