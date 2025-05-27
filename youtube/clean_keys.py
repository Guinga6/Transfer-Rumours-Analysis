import os
import json
import csv
import re
import difflib

def clean_key(key):
    key = re.sub(r'^\+?\s*', '', key)
    key = re.sub(r"\(corrected\)", "", key)
    key = re.sub(r'^\d+\.\s*', '', key)
    key = re.sub(r'(esting|ly)$', '', key)
    key = ' '.join(key.split())
    words = key.split()
    if len(set(words)) == 1:
        key = words[0]
    elif len(words) > 2:
        seen = []
        for w in words:
            if w not in seen:
                seen.append(w)
        key = ' '.join(seen)
    blacklist = {
        "Club", "Break close", "Corrected player name", "Release clause",
        "Arrival", "Departure"
    }
    if key in blacklist:
        return None
    return key.strip()
def is_irrelevant_value(value):
    targets = [
        "No transfer news mentioned",
        "buying, selling, loaning, rumors, negotiations, interest, talks, discussions, bid, offer, deal, agreement, contract, signing, move, switch, departure, arrival, target, linked, approach, enquiry, pursuit, speculation, considering, exploring, monitoring, tracking, scouting, wants, desires, keen, eyeing, planning, preparing, close to, verge of, set to, expected to, likely to, potential, possible, reported, alleged, suggested, claimed.",
        "see above",
        "No transfer news mentioned  I am an AI language model and do not have access to real-time or up-to-date information about the players. Please verify with a reliable source for accurate and up-to-date information.",
        "no recent transfer news mentioned",
        'mentioned'
    ]
    value = value.strip().lower()
    for target in targets:
        if difflib.SequenceMatcher(None, value, target.lower()).ratio() > 0.9:
            return True
    return False

p = './video_data/mistral'
files = os.listdir(p)
output_rows = []

for filename in files:
    if filename.endswith('.json'):
        path = os.path.join(p, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            time_value = data.get('time', '')
            focus = data.get('result', {})

            for key, value in focus.items():
                cleaned_key = clean_key(key)
                if cleaned_key and not is_irrelevant_value(value):
                    output_rows.append({
                        'filename': filename,
                        'date': time_value,
                        'player': cleaned_key,
                        'rumors': value
                    })

# Write to CSV
with open('player_rumors.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['filename', 'date', 'player', 'rumors']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)
