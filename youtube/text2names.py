import os 
import sys
import json
import spacy
import re
import time
from datetime import timedelta
from typing import List, Dict, Tuple, Optional
from fuzzywuzzy import fuzz, process
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.progress import show_progress
# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def deduplicate_names(names: List[str], threshold: int = 90) -> Dict[str, List[str]]:
    """
    Group similar names together and return canonical names with variants
    
    Args:
        names: List of entity names to deduplicate
        threshold: Similarity threshold for grouping (0-100)
        
    Returns:
        Dictionary of canonical names mapping to their variants
    """
    if not names:
        return {}
        
    name_groups = {}
    processed = set()
    
    for i, name in enumerate(names):
        if name in processed:
            continue
            
        show_progress(i+1, len(names))
        
        # Make this name the canonical one for its group
        canonical = name
        variants = [name]
        processed.add(name)
        
        # Find all similar names
        for other in names:
            if other != name and other not in processed:
                similarity = fuzz.ratio(name.lower(), other.lower())
                if similarity >= threshold:
                    variants.append(other)
                    processed.add(other)
        
        # Choose the most common variant as canonical
        if len(variants) > 1:
            canonical = max(variants, key=variants.count)
            
        name_groups[canonical] = variants
    
    print()  # Newline after progress bar
    return name_groups


def extract_context_for_entity(text: str, entity: str, context_window: int = 15) -> List[Dict[str, str]]:
    """
    Extract targeted context around each mention of the entity
    
    Args:
        text: The full text to search in
        entity: The entity to find
        context_window: Number of words to include before and after the entity
        
    Returns:
        List of dictionaries with context, position, and original sentence
    """
    results = []
    
    # Create regex pattern with word boundaries to find exact entity mentions
    pattern = r'\b' + re.escape(entity) + r'\b'
    
    # First split text into sentences for reference
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Find all occurrences of the entity in the text
    for match in re.finditer(pattern, text, re.IGNORECASE):
        start_pos, end_pos = match.span()
        
        # Find the sentence containing this mention
        sentence_for_mention = None
        sentence_start_pos = 0
        
        for sentence in sentences:
            sentence_end_pos = sentence_start_pos + len(sentence)
            if sentence_start_pos <= start_pos < sentence_end_pos:
                sentence_for_mention = sentence.strip()
                break
            # Update position for next sentence (+1 for space after sentence)
            sentence_start_pos = sentence_end_pos + 1
        
        # Get words before and after the entity
        words = text.split()
        entity_words = entity.split()
        entity_word_count = len(entity_words)
        
        # Find which word position corresponds to our entity
        word_pos = 0
        current_pos = 0
        
        for i, word in enumerate(words):
            word_length = len(word)
            if current_pos <= start_pos < current_pos + word_length:
                word_pos = i
                break
            # Move to next word position (+1 for space after word)
            current_pos += word_length + 1
        
        # Extract context window
        start_word_idx = max(0, word_pos - context_window)
        end_word_idx = min(len(words), word_pos + entity_word_count + context_window)
        
        context_words = words[start_word_idx:end_word_idx]
        context_text = " ".join(context_words)
        
        # Highlight the entity in the context
        entity_in_context = re.compile(pattern, re.IGNORECASE).sub(f"**{entity}**", context_text)
        
        results.append({
            "context": entity_in_context,
            "position": start_pos,
            "full_sentence": sentence_for_mention
        })
    
    return results


def clean_entity_name(name: str) -> str:
    """
    Clean entity names by removing common artifacts and normalizing
    
    Args:
        name: The entity name to clean
        
    Returns:
        Cleaned entity name
    """
    # Remove titles, extra spaces, punctuation at edges
    name = name.strip()
    name = re.sub(r'^(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+', '', name)
    name = re.sub(r'[\.,;:\'"]$', '', name)
    
    # Remove possessive 's at the end
    name = re.sub(r"'s$", "", name)
    
    return name


def filter_valid_entities(entities: List[str], min_length: int = 2) -> List[str]:
    """
    Filter valid entities removing ones that are too short or likely noise
    
    Args:
        entities: List of entity names to filter
        min_length: Minimum length of entity name
        
    Returns:
        Filtered list of entity names
    """
    filtered = []
    
    # Common words that are often misidentified as entities
    stopwords = {"the", "a", "an", "this", "that", "these", "those", "he", "she", "it", "they", "you"}
    
    for entity in entities:
        # Skip entities that are too short
        if len(entity) < min_length:
            continue
            
        # Skip entities that are just stopwords
        if entity.lower() in stopwords:
            continue
            
        # Keep only entities with at least one alphabetic character
        if not any(c.isalpha() for c in entity):
            continue
            
        filtered.append(clean_entity_name(entity))
    
    return filtered


def is_valid_entity(entity: str, doc: spacy.tokens.doc.Doc) -> bool:
    """
    Check if an entity is valid by examining its mentions in the document
    
    Args:
        entity: Entity name to validate
        doc: Spacy document
        
    Returns:
        Boolean indicating if entity is valid
    """
    # Check if entity appears in grammatical contexts typical for named entities
    entity_lower = entity.lower()
    
    # Count how many times the entity appears after specific indicators
    indicator_count = 0
    
    for token in doc:
        # Check if next tokens form our entity
        if token.i < len(doc) - 1:
            next_tokens = doc[token.i + 1:token.i + 6]  # Look at next few tokens
            next_text = "".join([t.text_with_ws for t in next_tokens])
            
            if re.search(r'\b' + re.escape(entity) + r'\b', next_text, re.IGNORECASE):
                # Check if current token is an indicator
                if token.text.lower() in ["said", "says", "by", "from", "with", "and", "or", "called"]:
                    indicator_count += 1
    
    # If entity appears after indicators multiple times, it's likely valid
    if indicator_count >= 2:
        return True
        
    # Entity is valid if it appears as part of named entities multiple times
    entity_in_ner_count = sum(1 for ent in doc.ents if entity_lower in ent.text.lower())
    
    return entity_in_ner_count >= 2


def process_transcription(transcription: str) -> Dict[str, Dict[str, any]]:
    """
    Process the transcription text to extract entities and their contexts
    
    Args:
        transcription: The text to process for entity extraction
        
    Returns:
        Structured dictionary with players, name variants, and relevant contexts
    """
    # Start timing
    proc_start_time = time.time()
    
    doc_spacy = nlp(transcription)
    
    # Extract person and organization entities
    spacy_names = list(set(
        ent.text.strip() for ent in doc_spacy.ents 
        if ent.label_ in ["PERSON", "ORG"] and len(ent.text.strip()) > 1
    ))
    
    # Filter out invalid entities and clean names
    logging.info(f"Found {len(spacy_names)} raw entities")
    valid_names = filter_valid_entities(spacy_names)
    logging.info(f"After filtering: {len(valid_names)} entities")
    
    # Further validate entities by checking context
    valid_context_names = [name for name in valid_names if is_valid_entity(name, doc_spacy)]
    logging.info(f"After context validation: {len(valid_context_names)} entities")
    
    # Group similar names (handles misspelled names)
    logging.info("Deduplicating entities and handling name variations...")
    name_groups = deduplicate_names(valid_context_names)
    
    logging.info(f"Grouped into {len(name_groups)} unique entities")
    
    # Process each canonical name to find its contexts
    result = {"players": {}}
    
    logging.info("Extracting context for each entity mention...")
    total_entities = len(name_groups)
    
    for idx, (canonical_name, variants) in enumerate(name_groups.items()):
        show_progress(idx+1, total_entities, proc_start_time)
        
        # Get all context mentions for this entity and its variants
        all_contexts = []
        for variant in variants:
            contexts = extract_context_for_entity(transcription, variant)
            all_contexts.extend(contexts)
        
        # Sort contexts by position in text
        all_contexts.sort(key=lambda x: x["position"])
        
        result["players"][canonical_name] = {
            "variants": variants,
            "mentions": all_contexts
        }
    
    print()  # Newline after progress bar
    
    proc_time = time.time() - proc_start_time
    logging.info(f"Processing completed in {timedelta(seconds=int(proc_time))}")
    
    return result


# Add parent directory to path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
nlp = spacy.load("en_core_web_trf") 

# Iterate over JSON files in directory
files = os.listdir('video_data')

for filename in files:
    if filename.endswith('.json'):
        path = os.path.join('video_data', filename)
        logging.info(f"Processing {filename}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                video = json.load(f)
                transcription = video['transcription']
                
                start_time = time.time()
                
                # Reset extracted_players_info to empty
                video["extracted_players_info"] = {}
                
                # Process the transcription to extract entities and contexts
                logging.info("Starting entity extraction and context processing...")
                extracted_data = process_transcription(transcription)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                video["extracted_players_info"] = extracted_data
                
                # Summary of results
                total_entities = len(extracted_data["players"])
                total_mentions = sum(len(data["mentions"]) for data in extracted_data["players"].values())
                
                logging.info(f'Processing completed in {timedelta(seconds=int(processing_time))}')
                logging.info(f'Extracted {total_entities} unique entities with {total_mentions} mentions')
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(video, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")