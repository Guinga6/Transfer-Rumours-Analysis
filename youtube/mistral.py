import os 
import sys
import json
import spacy
from transformers import pipeline
import time

# Add parent directory to path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformation.ollama import build_system_user_prompt,run_ollama_prompt,extract_json
# Load spaCy transformer-based model
nlp = spacy.load("en_core_web_trf") 

# Load Hugging Face NER pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

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
            chunks = [spacy_names[i:i+4] for i in range(0, len(spacy_names), 4)]
            start = time.time()
            for group in chunks:
                prompt = build_system_user_prompt(group, transcription)
                response = run_ollama_prompt(model_name="mistral:7b-instruct", messages=prompt)
                print(response)
                data = extract_json(response)
                print(type(data))
                if "extracted_players_info" not in video:
                    video["extracted_players_info"] = {}
                video["extracted_players_info"].update(data)
            end = time.time()
            print(f'time is {end - start:.2f} seconds')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(video, f, ensure_ascii=False, indent=2)
        break