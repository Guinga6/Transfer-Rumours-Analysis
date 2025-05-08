import os
import json
import whisper
from youtube_data_mining import download_audio
from transformation.transformation import audio_to_text
import difflib

rerun_folder = "video_data/rerun"
source_key = "title"  
target_key = "transcription"  
model = whisper.load_model("small")






files_in_directory = os.listdir()

for filename in os.listdir(rerun_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(rerun_folder, filename)
        try:
            with open(file_path, "r+", encoding="utf-8") as f:
                data = json.load(f)
                url = data.get(source_key)
                closest_match = difflib.get_close_matches(url, files_in_directory, n=1)
                text = audio_to_text(audio_path=closest_match[0],model=model)
                data[target_key] = text
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to update {filename}: {e}")
