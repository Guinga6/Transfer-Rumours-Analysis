import os
import json

def save_or_update_video_data(video_url, new_data, folder_path="video_data"):
    os.makedirs(folder_path, exist_ok=True)
    file_name = video_url.replace("https://", "").replace("/", "_") + ".json"
    file_path = os.path.join(folder_path, file_name)

    data = {"url": video_url}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass  # If file is corrupt or empty, start fresh

    data.update(new_data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
