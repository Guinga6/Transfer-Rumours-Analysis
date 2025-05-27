import os
import json

folder_path = "video_data"

def update_emojis_in_json(filename):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):  # Check if the file exists
        try:
            # Open the file in read/write mode
            with open(file_path, "r+", encoding="utf-8") as f:
                data = json.load(f)  # Load existing data

                # Move the pointer to the start and update the file
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)  # Save data with ensure_ascii=False to keep emojis
                f.truncate()  # Ensure the file is properly truncated after modifying content

            print(f"Successfully updated {filename} with emojis.")  # Success message
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error updating {filename}: {e}")
    else:
        print(f"File {filename} not found in {folder_path}")

# Loop through all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        update_emojis_in_json(filename)
