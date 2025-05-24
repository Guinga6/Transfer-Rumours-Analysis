import os
import json
import logging
from datetime import datetime
from time import time
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.progress import show_progress

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_modified_date(path):
    return datetime.fromtimestamp(os.path.getmtime(path)).isoformat()


def extract_metadata(directory="./"):
    start_time = time()
    mp3_files = [f for f in os.listdir(directory) if f.lower().endswith(".mp3")]
    total = len(mp3_files)
    songs = []

    for index, filename in enumerate(mp3_files, start=1):
        filepath = os.path.join(directory, filename)
        try:
            audio = EasyID3(filepath)
            title = audio.get('title', [os.path.splitext(filename)[0]])[0]
        except ID3NoHeaderError:
            title = os.path.splitext(filename)[0]

        creation_date = get_modified_date(filepath)
        songs.append({
            "title": title,
            "creation_date": creation_date
        })

        show_progress(index, total, start_time, description="Processing MP3s")

    os.makedirs("date", exist_ok=True)
    output_path = os.path.join("date", "songs_metadata.json")
    with open(output_path, "w", encoding="utf-8",) as f:
        json.dump(songs, f, indent=4,ensure_ascii= False)

    logging.info(f"Metadata for {total} songs saved to '{output_path}'")

# Run
extract_metadata()
