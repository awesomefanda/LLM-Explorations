import os
import gc
import torch
import soundfile as sf
from kokoro import KPipeline
from diffusers import AutoPipelineForText2Image
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips

# --- 1. SETUP & HARDWARE CHECK ---
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

device = "cuda" if torch.cuda.is_available() else "cpu"
# Tiny model for Low-RAM systems: "segmind/tiny-sd"
model_id = "segmind/tiny-sd" 

print(f"🚀 Studio starting on {device.upper()} using {model_id}...")

# --- 2. INITIALIZE ENGINES ---
# Load Image Engine
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id, 
    torch_dtype=torch.float32 if device == "cpu" else torch.float16
).to(device)

# Load Voice Engine
voice_pipeline = KPipeline(lang_code='a')

# --- 3. PRODUCTION TOOLS ---

def generate_assets(scene_idx, visual_p, narration):
    img_path = f"scene_{scene_idx}.jpg"
    aud_path = f"scene_{scene_idx}.wav"

    # Generate Image if missing
    if not os.path.exists(img_path):
        print(f"🎨 Drawing Scene {scene_idx}...")
        image = pipe(prompt=visual_p, num_inference_steps=20).images[0]
        image.save(img_path)
        gc.collect() # Clear RAM

    # Generate Voice if missing
    if not os.path.exists(aud_path):
        print(f"🎙️ Recording Voice {scene_idx}...")
        generator = voice_pipeline(narration, voice='af_heart', speed=1.0)
        for _, (_, _, audio) in enumerate(generator):
            sf.write(aud_path, audio, 24000)
            break
    
    return img_path, aud_path

# --- 4. MAIN MOVIE BUILDER ---

def make_movie(input_file):
    with open(input_file, "r") as f:
        scenes = f.read().split("SCENE")[1:]

    clips = []
    for i, scene_data in enumerate(scenes):
        try:
            # Parse text
            v_prompt = scene_data.split("VISUAL:")[1].split("NARRATION:")[0].strip()
            narr = scene_data.split("NARRATION:")[1].split("SCENE")[0].strip()
            
            # Generate local files
            img_p, aud_p = generate_assets(i+1, v_prompt, narr)
            
            # Create Clip (MoviePy 2.0 Syntax)
            audio = AudioFileClip(aud_p)
            img_clip = (ImageClip(img_p)
                        .with_duration(audio.duration)
                        .resized(height=1080)
                        .with_audio(audio))
            
            txt_clip = (TextClip(
                            text=narr, 
                            font_size=40, 
                            color='white', 
                            font='C:\\Windows\\Fonts\\arial.ttf',
                            method='caption',
                            size=(1600, None)
                        ).with_duration(audio.duration).with_position(('center', 850)))

            clips.append(CompositeVideoClip([img_clip, txt_clip]))
        except Exception as e:
            print(f"⚠️ Skipping scene {i+1} due to error: {e}")

    print("🎞️ Stitching final movie...")
    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile("final_ai_movie.mp4", fps=24)

if __name__ == "__main__":
    make_movie("moon_landing.txt")