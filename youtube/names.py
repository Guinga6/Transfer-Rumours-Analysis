import os
import json
import logging
from collections import Counter
from time import time
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.progress import show_progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_player_names(directory="./video_data/mistral"):
    player_counter = Counter()
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    total = len(files)
    start_time = time()

    for i, filename in enumerate(files, 1):
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            players = data.get("result", {})
            for name in players.keys():
                player_counter[name] += 1

        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")

        show_progress(i, total, start_time, description="Processing JSON files")

    return dict(sorted(player_counter.items()))

def save_to_txt(data, output_file="names.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for name, count in data.items():
            if count == 1:
                f.write(f"{name}: {count}\n")
            else:
                continue
    logging.info(f"Saved sorted names and counts to {output_file}")

# Run
player_data = extract_player_names()
save_to_txt(player_data)
