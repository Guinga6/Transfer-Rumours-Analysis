import os
import json
import logging
from time import time
from difflib import get_close_matches

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_song_metadata(path="./date/songs_metadata.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {item["title"]: item["creation_date"] for item in json.load(f)}
    except FileNotFoundError:
        logging.error(f"Metadata file not found at {path}")
        return {}

def update_video_data_with_fuzzy_match(metadata, directory="./video_data", cutoff=0.8):
    files = [f for f in os.listdir(directory) if f.lower().endswith(".json")]
    total = len(files)
    start_time = time()

    updated = 0
    not_matched = 0
    matched_titles = set()

    for i, filename in enumerate(files, 1):
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "").strip()
            match = get_close_matches(title, metadata.keys(), n=1, cutoff=cutoff)

            if match:
                matched_title = match[0]
                data["time"] = metadata[matched_title]
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                matched_titles.add(matched_title)
                updated += 1
            else:
                not_matched += 1
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

        percent = int((i / total) * 100)
        print(f"Progress: [{percent}%] ({i}/{total})", end="\r")

    unused_titles = set(metadata.keys()) - matched_titles

    logging.info("\n===== SUMMARY =====")
    logging.info(f"✔️  JSON files updated: {updated}")
    logging.info(f"❌  JSON files not matched: {not_matched}")
    logging.info(f"🟡  Titles in songs_metadata.json with no matching .json: {len(unused_titles)}")

# Run
metadata = load_song_metadata()
update_video_data_with_fuzzy_match(metadata)
