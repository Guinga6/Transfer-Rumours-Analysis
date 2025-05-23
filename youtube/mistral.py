import os 
import sys
import json
import spacy
from transformers import pipeline
import time
from typing import List
from fuzzywuzzy import fuzz
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def snake_progress(current, total):
    snake = '■' * current
    sys.stdout.write(f'\rProcessing groups: {current}/{total} ' + snake)
    sys.stdout.flush()


def deduplicate_names(names: List[str], threshold: int = 90) -> List[str]:
    unique = []
    for name in names:
        if all(fuzz.ratio(name.lower(), u.lower()) < threshold for u in unique):
            unique.append(name)
    return unique

# Add parent directory to path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformation.ollama import build_system_user_prompt,run_ollama_prompt,extract_json,build_user_prompt,build_json_system_user_prompt
from transformation.rag_pipline import build_rag_input
# Load spaCy transformer-based model
nlp = spacy.load("en_core_web_trf") 

# # Load Hugging Face NER pipeline
# ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# Iterate over JSON files in directory
files = os.listdir('video_data')

for filename in files:
    final_output = {"players": {}}
    if filename.endswith('.json'):
        path = os.path.join('video_data', filename)

        with open(path, 'r', encoding='utf-8') as f:
            video = json.load(f)
            transcription = video['transcription']
            
            doc_spacy = nlp(transcription)
            spacy_names = list(set(
                ent.text.strip() for ent in doc_spacy.ents 
                if ent.label_ in ["PERSON", "ORG"]
            ))
            deduped_names = deduplicate_names(spacy_names)
            chunks = [deduped_names[i:i+7] for i in range(0, len(deduped_names), 7)]
            total_groups = len(chunks)
            start = time.time()
            total_groups = len(chunks)
            for idx, group in enumerate(chunks, start=1):
                snake_progress(idx, total_groups)
                max_retries = 5
                attempt = 0
                is_true = True
                used_chunk = build_rag_input(names=group, text=transcription)
                logging.info(f'Used chunks:\n{group}')
                prompt = build_json_system_user_prompt(group, used_chunk)
                while is_true and attempt < max_retries:
                    response = run_ollama_prompt(model_name="mistral:7b-instruct-q4_K_M", messages=prompt)
                    data = extract_json(response)
                    if isinstance(data, dict):
                        is_true = False
                    attempt += 1
                if is_true:
                    logging.warning(f"Failed to extract valid JSON after {max_retries} attempts for chunk: {group}")
                    continue
                if "extracted_players_info" not in video:
                    video["extracted_players_info"] = {}
                video["extracted_players_info"].update(data)
            end = time.time()
            logging.info(f'Time is {end - start:.2f} seconds')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(video, f, ensure_ascii=False, indent=2)
