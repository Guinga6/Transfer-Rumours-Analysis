import os
import json
import shutil
import time

folder_path = "video_data"
key_to_check = "transcription"
rerun_folder = os.path.join(folder_path, "rerun")

os.makedirs(rerun_folder, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)

        # Retry up to 3 times
        for attempt in range(3):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get(key_to_check) is None:
                    shutil.move(file_path, os.path.join(rerun_folder, filename))
                break  # success, break retry loop

            except PermissionError as e:
                print(f"Retrying {filename} due to permission error...")
                time.sleep(1)

            except (json.JSONDecodeError, IOError) as e:
                print(f"Skipping {filename}: {e}")
                break
