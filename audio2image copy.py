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
from openai import OpenAI   

# ======================================================================================
# Constants
# ======================================================================================

# The volume threshold used to decide if there is speaking in the audio.
# If the volume is below this number, it will be considered silence.
SILENCE_THRESHOLD = 2

# The minimum length of time (in seconds) someone must be speaking for the audio to count.
# If speaking is shorter than this, we ignore the audio segment.
# Lower this number if you expect only very short responses.
MIN_VOICE_DURATION = 2.5

# The length of silence (in seconds) that will trigger the audio processing to stop.
# In other words, if the user stops speaking for this many seconds, we finalize and process.
# Increase this if you want to allow more pauses when the user is thinking.
SILENCE_GAP = 8

# The sample rate for recording and processing audio. 16000 is standard for the Whisper model.
SAMPLE_RATE = 16000

# ======================================================================================
# OpenAI and Google Cloud Setup
# ======================================================================================

# Create an OpenAI API client. Replace the API key with your own.
client = OpenAI(api_key="") 

# Load Google Cloud credentials from a local JSON file to access Google Cloud Storage
credentials = service_account.Credentials.from_service_account_file(
    "C:\Workspace\Future Bag\dalle_3\credentials.json"  
)

# This list will keep track of all the image descriptions provided by the user
image_descriptions = [] 

# ======================================================================================
# Function to Upload Files to Google Cloud Storage
# ======================================================================================
def upload_blob(bucket_name, source_file_path, destination_blob_name):
    """
    Uploads a file to a specified Google Cloud Storage (GCS) bucket.
    
    Parameters: 
    - bucket_name: The ID of your GCS bucket
    - source_file_name: The path to your file to upload   
    - destination_blob_name: The name of file to be stored in the bucket           
    """
    
    # Create a client to interact with Google Cloud Storage
    storage_client = storage.Client(credentials=credentials)    
    
    # Access the specified bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Prepare a blob (an object in GCS) for the destination file name
    blob = bucket.blob(destination_blob_name)
    
    # Upload the local file to the blob
    blob.upload_from_filename(source_file_path)
    
    # Set cache control so the file is not cached (ensures fresh file on each upload)           
    blob.cache_control = "no-cache, max-age=0"
    blob.patch()
    
# ======================================================================================
# Initial Setup: Upload a Transparent Image
# ======================================================================================    
# Upload a transparent image so that initially, users see an empty/transparent background
upload_blob("future_bag", "advice_images/background.png", "lasted_generated_image.png")    

# Directory for storing images  
if not os.path.exists("images"):
    os.mkdir("images") 

# Directory for storing images without backgrounds
if not os.path.exists("images_without_background"):
    os.mkdir("images_without_background")   

# ======================================================================================
# Whisper Model Setup
# ======================================================================================    

# Decide whether to run on GPU or CPU based on availability     
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Download or load the "turbo" Whisper model to the "models" folder (faster variant).
whisper_model = whisper.load_model("turbo", device=device, download_root="models") 

# Run a quick test transcription to speed up the first real use.
# After this "warm up," the model will respond faster for the first user input.
whisper_model.transcribe("warmup.m4a")       
print("Whisper model is ready!")

def record_audio():
    """
    Continuously listen for incoming audio from the microphone.
    Once the user starts speaking above a certain volume threshold (SILENCE_THRESHOLD),
    we begin recording. We stop recording after a period of silence (SILENCE_GAP).
    Finally, if the recorded segment is long enough (MIN_VOICE_DURATION),
    we call the processing function for transcription and image generation.
    """
    
    print("Listening for audio...")
    # Update the cloud image to indicate we are in listening mode
    upload_blob("future_bag", "advice_images/listening.png", "lasted_generated_image.png") 
    
    # audio_buffer collects small chunks of audio data between silences.
    audio_buffer = []
    recording = False # Indicates whether we're currently in "recording" mode.
    start_time = None
    last_sound_time = None

    def callback(indata, frames, time, status):
        """
        A callback function that 'sounddevice' calls every time it has new audio data.
        - indata: The actual audio data as a NumPy array.
        - frames: The number of frames in this block of audio.
        - time: Timing information (not typically used here).
        - status: A status object used to notify about input overflows, etc.
        """
        nonlocal audio_buffer, recording, start_time, last_sound_time
        
        # Calculate the approximate "loudness" of this audio chunk.
        volume_norm = np.linalg.norm(indata) * 10

        if volume_norm > SILENCE_THRESHOLD:
            # If we're not already recording, this is the point where we start
            if not recording:
                recording = True
                start_time = t.time()
                print("Start speaking...")
                
            # Update the time we last detected any sound above the threshold
            last_sound_time = t.time() 
            
            # Store the current chunk of audio data
            audio_buffer.append(indata.copy())
            
        elif recording:
            # If we are in the middle of a recording but the volume dropped below the threshold:
            current_time = t.time()
            # Check how long it has been since we last heard any sound
            silence_duration = current_time - last_sound_time if last_sound_time else 0
            
            # If we've had no sound for longer than SILENCE_GAP, consider this a finished segment
            if silence_duration > SILENCE_GAP:
                duration = current_time - start_time
                
                # Check if the total recorded time is at least MIN_VOICE_DURATION
                if duration >= MIN_VOICE_DURATION:
                    print("Capturing audio...")
                    # Process the recorded audio data for transcription and further tasks
                    process_audio(audio_buffer)
                else:
                    # If too short, we ignore and tell the user to try again
                    print(f"Audio too short ({duration:.2f} seconds). Please try again.")
                    upload_blob("future_bag", "advice_images/try again.png", "lasted_generated_image.png")        
                
                # Reset everything for the next possible recording
                audio_buffer = []  # Clear buffer after processing
                recording = False
            else:
                # If it's not been quiet long enough, we keep adding audio data to the buffer
                audio_buffer.append(indata.copy())
                
    # This opens a continuous audio InputStream using sounddevice. 
    # The callback function is triggered for each chunk of audio.
    with sd.InputStream(callback=callback, samplerate= SAMPLE_RATE, channels=1):
        while True:
            t.sleep(0.1)

def process_audio(audio_buffer):
    """
    Process the collected audio chunks, save them to a WAV file,
    transcribe using Whisper, then trigger image generation if transcription is valid.
    """
    # Combine the chunks in the buffer into one continuous NumPy array (float32).
    audio_data = np.concatenate(audio_buffer, axis=0).astype(np.float32)
    
    
    try:
        # Save the raw audio data to a temporary WAV file before transcribing.
        wav_file = "tmp.wav"
        write(wav_file, SAMPLE_RATE, audio_data)
        
        # Use Whisper to transcribe the audio.   
        result = whisper_model.transcribe(wav_file)
        transcription = result.get("text", "").strip()
        
        # ======================================================================================
        # Trigger words Setup
        # ======================================================================================
        if transcription: 
        # If you want user to say "mystery bag is" or "mystery bag contains" before the description, uncomment the following code and comment the line before       
        # if transcription and any(triggerword in transcription for triggerword in ["mystery bag", "mystery back", "mystery pack", "mystery box", "future bag", "future back", "artefact"]):               
            # pattern = r'.*?\b(is|contains)\b'               
            # transcription = re.sub(pattern, '', transcription, count=1).strip().replace(".", "")
        
        # ======================================================================================
        # GPT enhanced prompt Setup
        # ======================================================================================        
            
            # TODO Prompt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print(f"User Transcription: {transcription}")  
            enhanced_transcription = enhance_description(transcription) 
            print(f"GPT Enhanced Transcription: {enhanced_transcription}")
            generate_images(enhanced_transcription)
        else:
            # If the transcription is empty or invalid, prompt the user to try again.
            print(f"Invalid transcription \"{transcription}\". Please try again.")
            upload_blob("future_bag", "advice_images/try again.png", "lasted_generated_image.png")
            # Brief pause before reverting to a default or previously generated image.
            t.sleep(2)
            upload_blob("future_bag", "advice_images/generated_image.png", "lasted_generated_image.png")      
            
    except Exception as e:
        print(f"Error during audio processing: {e}")
        
def enhance_description(transcription):
    """
    Enhance the user's transcription to make it a more vivid image prompt for DALL·E 3.
    Uses GPT to provide details in a cyberpunk/retro-futuristic style.
    """
    upload_blob("future_bag", "advice_images/loading.png", "lasted_generated_image.png")
    global image_descriptions
    base_description = "Enhance this transcription into a detailed image description for delle3: "  

    # if image_descriptions:
    #     prompt = f"{base_description}{transcription}. Modify it based on the previous description: {image_descriptions[-1]}"
    # else:
    #     prompt = f"{base_description}{transcription}"
    
    # Create a prompt for GPT using the user's transcription.
    # You could also incorporate previous descriptions for iterative modifications.
    prompt = f"{base_description}{transcription}"
    
    try:    
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant who generates clear image prompts "
                        "for dalle3. The images are cyberpunk and retro-futuristic. Make old user decriptions more clear and"
                        "concise, keep the user's original intent, and add few details to make the image more vivid. The propt"  
                        "you added should not be more than 20 words."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
    except Exception as e:  
        print(f"Error during GPT-4o completion: {e}")   
        return transcription
    
    # Extract the text from the GPT response.
    enhanced_description = response.choices[0].message.content.strip()
    # Save this in our global list to track or reuse later.
    image_descriptions.append(enhanced_description)
    return enhanced_description         
        
def generate_images(prompt):
    """
    Generate an image from a prompt using DALL·E 3, then remove its background
    and upload the final image to Google Cloud Storage.
    """
    # Use the OpenAI image generation API.
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        quality="standard",
        size="1024x1024"
    )
    # Extract the URL of the generated image from the API response.
    image_url = response.data[0].url
    
    # Download the image from the URL to a temporary file.
    img_data = requests.get(image_url).content
    download_path = "tmp.png"   
    with open(download_path, "wb") as handler:
        handler.write(img_data)
    
    # Create a unique name for the output image file using the current timestamp.
    # We also add the last 20 characters of the prompt to help identify the image.
    file_name = time.strftime('%Y%m%d%H%M%S') + prompt[-20:] 
    output_path = f"images/{file_name}.png" 
    
    # Save a copy of the downloaded image into the 'images' folder.
    shutil.copy(download_path, output_path)   
    
    # Remove the background of the generated image and save as "generated_image.png".
    remove_background(download_path, "generated_image.png")   
    
    # Copy the background-removed image to 'images_without_background' folder as well.
    shutil.copy("generated_image.png", f"images_without_background/{file_name}.png")      
    
    # Finally, upload the newly created image to the cloud (e.g., to replace the displayed image).
    upload_blob("future_bag", "generated_image.png", "lasted_generated_image.png") 
    print("Success upload image!")
    
def remove_background(src_img_path, out_img_path):
    """
    Removes the background from an image using the 'backgroundremover' library.
    If an error occurs, it keeps the original file.
    """
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]
    f = open(src_img_path, "rb")
    data = f.read()
    
    try:
        # Perform background removal with specific alpha matting settings for better edges.
        img = remove(data, model_name=model_choices[0],
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_structure_size=10,
                    alpha_matting_base_size=1000)
        
    except Exception as e:      
        # If removal fails, just copy the original data to output path.
        print(f"Error during background removal: {e}")  
        f.close()
        f = open(out_img_path, "wb")
        f.write(data)
        f.close()
        return
    
    # Save the newly processed (background-removed) image.
    f.close()
    f = open(out_img_path, "wb")
    f.write(img)
    f.close()

if __name__ == "__main__":
    record_audio()  
    