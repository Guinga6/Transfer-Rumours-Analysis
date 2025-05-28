import os 
import sys
import json
import spacy
from transformers import pipeline
import time
from typing import List
from fuzzywuzzy import fuzz
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Add parent directory to path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformation.ollama import build_system_user_prompt,run_ollama_prompt,extract_json,build_user_prompt,build_json_system_user_prompt
from utils.progress import show_progress
# Load spaCy transformer-based model
nlp = spacy.load("en_core_web_trf")

def filter_player_names(keys):
    """
    Filter keys list to keep only those recognized as PERSON by spaCy NER.

    Args:
      keys (list[str]): mixed player and club names

    Returns:
      list[str]: keys recognized as PERSON (players)
    """
    players = []
    for key in keys:
        doc = nlp(key)
        if any(ent.label_ == "PERSON" for ent in doc.ents):
            players.append(key)
    return players

from typing import Tuple, Dict, Any, List

def prepare_input_batch(merged_contexts: Dict[str, Any], batch_size: int = 6) -> Tuple[List[str], str, Dict[str, Any], bool]:
    """
    Processes player contexts in batches and tracks remaining data.
    
    Args:
        merged_contexts: Dictionary containing player context data
        batch_size: Number of players to process per batch (default: 9)
    
    Returns:
        tuple: (player_names, formatted_contexts, remaining_data, done_flag)
        where:
        - player_names: List of player names in current batch
        - formatted_contexts: String of formatted contexts for the batch
        - remaining_data: Dictionary with unprocessed players
        - done_flag: True when no players remain
    """
    # Create a copy to avoid modifying original
    remaining_data = merged_contexts.copy()
    player_names = filter_player_names(list(remaining_data.keys()))
    
    # Get batch of players
    batch_players = player_names[:batch_size]
    remaining_players = player_names[batch_size:]
    # print(f'those are the remaining player:{remaining_players}')
    # Prepare contexts for current batch
    formatted_entries = []
    for player in batch_players:
        contexts = []
        for mention in remaining_data[player].get('mentions', []):
            ctx = mention.get('context', '').strip().replace('\n', ' ')
            topic = mention.get('topic', 'General')
            if ctx:
                contexts.append(f'"{ctx} (topic: {topic})"')
        
        if contexts:
            formatted_entries.append(f"{player}=[{', '.join(contexts)}]")
        
        # Remove processed player from remaining data
        remaining_data.pop(player, None)
    
    # Prepare return values
    formatted_contexts = ', '.join(formatted_entries)
    done = len(remaining_players) == 0
    
    return batch_players, formatted_contexts, remaining_data, done



# Iterate over JSON files in directory
files = os.listdir('video_data')

files_10 = files[:10]
total = len(files_10)
for (idx,filename) in enumerate(files_10,1):

    final_output = {}
    if filename.endswith('.json'):
        i = 0
        logging.info(f'processing file {filename}')
        done =False
        path = os.path.join('video_data', filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                video = json.load(f)
                contexte = video['clean_player']
                video['clean_output']={}
                start = time.time()
                show_progress(idx, total, start, description="Mistal is working ...")
                while not done: 
                    i +=1
                    name_list, meged_contexte,contexte,done = prepare_input_batch(contexte)
                    prompt= build_system_user_prompt(player_names=name_list,contexts=meged_contexte)

                    response = run_ollama_prompt(model_name="mistral:7b-instruct-q4_K_M", messages=prompt)

                    
                    # data = extract_json(response)
                    video['clean_output'][str(i)] = response
                end = time.time()
                gc.collect()
                logging.info(f'Time is {end - start:.2f} seconds')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(video, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f'skiping file {filename} because of an error {e}')
