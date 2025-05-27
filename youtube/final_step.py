import os 
import sys
import json
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Add parent directory to path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.progress import show_progress

def parse_text_to_dict(text):
    entries = text.split('\n')
    result = {}
    current_key = None
    current_value = []

    for line in entries:
        if ':' in line:
            if current_key:
                result[current_key] = ' '.join(current_value).strip()
            key, value = line.split(':', 1)
            current_key = key.strip()
            current_value = [value.strip()]
        else:
            current_value.append(line.strip())

    if current_key:
        result[current_key] = ' '.join(current_value).strip()

    return result

p = './video_data/'
files = os.listdir(p)
total = len(files)

for idx, filename in enumerate(files, 1):
    result = {'result': {}}
    if filename.endswith(".json"):
        path = os.path.join(p, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            focus = data['clean_output']
            show_progress(idx, total)
            for key, value in focus.items():
                result['result'].update(parse_text_to_dict(value))
        data['result'] = result['result']
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
