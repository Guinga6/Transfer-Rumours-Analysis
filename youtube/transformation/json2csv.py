import os
import json
import csv

input_dir = "video_data/result"
output_file = "data/rumors_fabrizio.csv"

rows = []

# Load existing data if file exists
if os.path.exists(output_file):
    with open(output_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows.extend(list(reader))

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {filename}")
                continue
            context = data.get("player_context_V2", {})
            player_dict = context.get("players", {})

            for player_key, player_val in player_dict.items():
                mentions = player_val.get("mentions", [])
                for mention in mentions:
                    row = {
                        'player_key': player_key,
                        'player_value': json.dumps(player_val.get("variants", []))
                    }
                    for mk, mv in mention.items():
                        if isinstance(mv, dict):
                            for subk, subv in mv.items():
                                row[f'{mk}_{subk}'] = subv
                        else:
                            row[mk] = mv
                    rows.append(row)

if rows:
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = list(fieldnames)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
