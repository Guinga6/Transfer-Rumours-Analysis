import os 
import sys
import json
import spacy
import re
import time
import numpy as np
from collections import defaultdict
from datetime import timedelta
from typing import List, Dict, Tuple, Optional, Set
from fuzzywuzzy import fuzz, process
import logging
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

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


def extract_smart_context(text: str, entity: str, doc_spacy: spacy.tokens.doc.Doc, 
                         min_context_words: int = 15, max_context_words: int = 30) -> List[Dict[str, str]]:
    """
    Extract targeted context around each mention of football entities with smart window sizing
    
    Args:
        text: The full text to search in
        entity: The football entity to find (player, club, agent)
        doc_spacy: Spacy-processed document
        min_context_words: Minimum number of words to include before and after the entity
        max_context_words: Maximum number of words to include before and after the entity
        
    Returns:
        List of dictionaries with context, position, and original sentence
    """
    results = []
    
    # Create regex pattern with word boundaries to find exact entity mentions
    pattern = r'\b' + re.escape(entity) + r'\b'
    
    # Transfer-related terms (defined once to avoid duplication)
    transfer_terms = ["transfer", "sign", "signed", "signing", "move", "deal", "bid", "fee", 
                     "price", "wages", "contract", "interest", "target", "negotiation", 
                     "medical", "rumor", "rumour", "join", "offer", "talks"]
    
    # Find all occurrences of the entity in the text
    for match in re.finditer(pattern, text, re.IGNORECASE):
        start_pos, end_pos = match.span()
        
        # Find which spaCy sentence contains this mention
        containing_sentence = None
        for sent in doc_spacy.sents:
            if start_pos >= sent.start_char and start_pos < sent.end_char:
                containing_sentence = sent
                break
        
        # If no sentence found (unlikely), create a fallback
        if not containing_sentence:
            # Find closest sentence boundaries
            prev_boundary = text.rfind('.', 0, start_pos)
            next_boundary = text.find('.', end_pos)
            if prev_boundary == -1:
                prev_boundary = 0
            if next_boundary == -1:
                next_boundary = len(text)
            
            sentence_text = text[prev_boundary:next_boundary].strip()
        else:
            sentence_text = containing_sentence.text.strip()
        
        # Determine context size based on transfer relevance and information density
        context_size = min_context_words  # Default
        
        if containing_sentence:
            # Check for transfer-related terms in the sentence
            has_transfer_context = any(term in containing_sentence.text.lower() for term in transfer_terms)
            
            if has_transfer_context:
                context_size = max_context_words  # Maximum context for transfer mentions
            else:
                # Calculate information density using POS tags
                if len(containing_sentence) > 0:  # Avoid division by zero
                    info_tokens = [token for token in containing_sentence if token.pos_ in ["NOUN", "VERB", "PROPN", "NUM"]]
                    info_density = len(info_tokens) / len(containing_sentence)
                    
                    # Scale context size based on information density
                    context_size = min(
                        max_context_words, 
                        max(min_context_words, int(min_context_words + (info_density * 15)))
                    )
        
        # Get token index of the entity mention
        entity_token_idx = None
        entity_tokens = []
        
        for i, token in enumerate(doc_spacy):
            if token.idx <= start_pos < token.idx + len(token.text):
                entity_token_idx = i
                # Collect all tokens that are part of the entity
                token_end = entity_token_idx
                while token_end < len(doc_spacy) and token_end - entity_token_idx < 10:  # Limit to 10 tokens max
                    if token_end * entity_token_idx > 0 and doc_spacy[token_end].idx + len(doc_spacy[token_end].text) <= end_pos:
                        entity_tokens.append(doc_spacy[token_end])
                        token_end += 1
                    else:
                        break
                break
        
        if entity_token_idx is not None:
            # Calculate context window boundaries
            start_token_idx = max(0, entity_token_idx - context_size)
            end_token_idx = min(len(doc_spacy), entity_token_idx + len(entity_tokens) + context_size)
            
            # Extract context
            context_tokens = doc_spacy[start_token_idx:end_token_idx]
            context_text = context_tokens.text
            
            # Check for transfer keywords in this specific context
            has_transfer_keywords = any(keyword in context_text.lower() for keyword in transfer_terms)
            
            # Highlight the entity in the context using regex to maintain case
            entity_in_context = re.compile(pattern, re.IGNORECASE).sub(f"**{entity}**", context_text)
            
            results.append({
                "context": entity_in_context,
                "position": start_pos,
                "full_sentence": sentence_text,
                "context_size": context_size,
                "transfer_related": has_transfer_keywords
            })
    
    return results


def cluster_mentions(entity: str, mentions: List[Dict]) -> List[Dict[str, any]]:
    """
    Cluster similar mentions of an entity to identify distinct discussion topics
    
    Args:
        entity: The entity name
        mentions: List of mention contexts
        
    Returns:
        List of clustered mentions with topic labels
    """
    import re
    import numpy as np
    import logging
    from collections import defaultdict
    from sklearn.metrics.pairwise import cosine_similarity
    
    if len(mentions) <= 1:
        # No clustering needed for single mention
        for mention in mentions:
            mention["cluster"] = 0
            mention["topic"] = "General"
        return mentions
    
    # Extract just text content from mentions for clustering
    mention_texts = [m["context"] for m in mentions]
    
    # Pre-process texts to remove entity highlighting markers for better vectorization
    clean_texts = [re.sub(r'\*\*' + re.escape(entity) + r'\*\*', entity, text) for text in mention_texts]
    
    # Try alternative vectorization methods if spaCy vectors fail
    valid_vectors = False
    
    
    try:
        
        import spacy
        nlp = spacy.load("en_core_web_trf")  
        
        # Process with spaCy
        mention_docs = list(nlp.pipe(clean_texts))
        
        # Check if vectors are available and non-zero
        if all(doc.has_vector for doc in mention_docs) and any(doc.vector.any() for doc in mention_docs):
            mention_vectors = np.array([doc.vector for doc in mention_docs])
            
            # Verify vectors are valid
            if mention_vectors.shape[1] > 0 and not np.isnan(mention_vectors).any():
                logging.info(f"Using spaCy vectors for entity '{entity}' clustering")
                valid_vectors = True
            else:
                logging.warning(f"Invalid spaCy vectors for entity '{entity}' - trying alternative method")
        else:
            logging.warning(f"spaCy vectors not available for entity '{entity}' - trying alternative method")
            
    except Exception as e:
        logging.warning(f"Error generating spaCy vectors for entity '{entity}': {str(e)} - trying alternative method")
    
    # APPROACH 2: If spaCy vectors fail, create simple bag-of-words vectors
    if not valid_vectors:
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Create a simple bag-of-words vectorizer - more stable than TF-IDF
            vectorizer = CountVectorizer(
                min_df=1, 
                max_df=0.9,
                stop_words='english', 
                lowercase=True,
                binary=True  # Binary features (presence/absence) are more robust
            )
            
            # Generate BOW vectors
            mention_vectors = vectorizer.fit_transform(clean_texts).toarray()
            
            # Check if we have valid vectors now
            if mention_vectors.shape[1] > 0:
                logging.info(f"Using Bag-of-Words vectors for entity '{entity}' clustering")
                valid_vectors = True
            else:
                logging.warning(f"Bag-of-Words vectorization failed for entity '{entity}' - falling back to simple clustering")
                
        except Exception as e:
            logging.warning(f"Vectorization error for entity '{entity}': {str(e)} - falling back to simple clustering")
    
    # If we still don't have valid vectors, use simple clustering
    if not valid_vectors or len(mentions) < 3:
        for mention in mentions:
            mention["cluster"] = 0
            mention["topic"] = "General"
        return mentions
    
    # For small datasets, use simple distance-based clustering
    if len(mentions) < 5:
        try:  
            # For very small sets, use simple approach
            similarity_matrix = cosine_similarity(mention_vectors)
            
            # Simple thresholding - if similarity > 0.5, put in same cluster
            threshold = 0.5
            clusters = []
            assigned = set()
            
            for i in range(len(mentions)):
                if i in assigned:
                    continue
                    
                current_cluster = [i]
                assigned.add(i)
                
                for j in range(i+1, len(mentions)):
                    if j not in assigned and similarity_matrix[i, j] > threshold:
                        current_cluster.append(j)
                        assigned.add(j)
                
                clusters.append(current_cluster)
            
            # Create cluster labels
            cluster_labels = np.zeros(len(mentions), dtype=int)
            for cluster_id, cluster in enumerate(clusters):
                for idx in cluster:
                    cluster_labels[idx] = cluster_id
                    
        except Exception as e:
            logging.warning(f"Simple clustering failed for entity '{entity}': {str(e)} - using single cluster")
            for mention in mentions:
                mention["cluster"] = 0
                mention["topic"] = "General"
            return mentions
    else:
        # For larger datasets, try hierarchical clustering instead of DBSCAN
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(mention_vectors)
            
            # Apply hierarchical clustering - more stable than DBSCAN
            num_clusters = min(3, len(mentions) // 2)  # Reasonable number of clusters
            
            # Fix: Remove unsupported parameters
            # Check sklearn version to use appropriate parameters
            import sklearn
            from packaging import version
            
            if version.parse(sklearn.__version__) >= version.parse('0.24.0'):
                # For newer versions of sklearn
                clusterer = AgglomerativeClustering(
                    n_clusters=num_clusters,
                    metric='precomputed',
                    linkage='average'
                )
            else:
                # For older versions of sklearn
                clusterer = AgglomerativeClustering(
                    n_clusters=num_clusters,
                    linkage='average'
                )
                
            # AgglomerativeClustering requires a distance matrix, not similarity
            distance_matrix = 1 - similarity_matrix  
            
            # For older versions that don't support precomputed distances directly
            if version.parse(sklearn.__version__) < version.parse('0.24.0'):
                from sklearn.metrics import pairwise_distances
                # Convert vectors to pairwise distances using cosine metric
                distance_matrix = pairwise_distances(mention_vectors, metric='cosine')
                
            cluster_labels = clusterer.fit_predict(distance_matrix)
            
        except Exception as e:
            logging.warning(f"Hierarchical clustering failed for entity '{entity}': {str(e)} - using simple clustering")
            # Fall back to single cluster
            for mention in mentions:
                mention["cluster"] = 0
                mention["topic"] = "General"
            return mentions
    
    # Generate topic labels for each cluster
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
    
    # Extract keywords for each cluster
    cluster_topics = {}
    for cluster_id, indices in clusters.items():
        # Combine texts from this cluster
        cluster_texts = [mention_texts[i] for i in indices]
        combined_text = " ".join(cluster_texts)
        
        try:
            # Extract key words using simpler approach - more robust
            words = re.findall(r'\b\w+\b', combined_text.lower())
            stopwords = {"the", "a", "an", "and", "in", "on", "at", "for", "with", "to", "of", "is", "are", 
                         "was", "were", "by", "that", "this", "it", "as", "be", "will", "can", "has", "have",
                         entity.lower()}
                
            # Filter short words and entity name
            word_counts = defaultdict(int)
            for word in words:
                if word not in stopwords and len(word) > 3:
                    word_counts[word] += 1
            
            if word_counts:
                # Use most frequent meaningful word
                most_common_word = max(word_counts.items(), key=lambda x: x[1])[0]
                cluster_topics[cluster_id] = most_common_word.title()
            else:
                cluster_topics[cluster_id] = f"Topic {cluster_id + 1}"
        except Exception as e:
            logging.warning(f"Topic extraction failed for cluster {cluster_id}: {str(e)}")
            cluster_topics[cluster_id] = f"Topic {cluster_id + 1}"
    
    # Assign cluster and topic to each mention
    for i, mention in enumerate(mentions):
        cluster_id = cluster_labels[i]
        mention["cluster"] = int(cluster_id)
        mention["topic"] = cluster_topics.get(cluster_id, f"Topic {cluster_id + 1}")
    
    return mentions


def extract_relationships(entities: Dict[str, Dict], doc_spacy: spacy.tokens.doc.Doc) -> Dict[str, List[Dict]]:
    """
    Extract relationships between entities by analyzing co-occurrences and linguistic patterns
    
    Args:
        entities: Dictionary of entities with their variants and mentions
        doc_spacy: Spacy-processed document
        
    Returns:
        Dictionary of relationships between entities
    """
    relationships = {}
    
    # Create a mapping of entity variants to canonical names
    variant_to_canonical = {}
    for canonical, data in entities.items():
        for variant in data["variants"]:
            variant_to_canonical[variant.lower()] = canonical
    
    # Define relationship extraction patterns for football transfers and rumors
    relationship_patterns = [
        # A signed/joining B (player to club)
        (r'\b({0})\s+(?:signed|joining|joins|joined|to join|will join|agreed|moved)\s+(?:for|with)?\s+({1})\b', 'signed for'),
        # A interested in B (club interested in player)
        (r'\b({0})\s+(?:interested in|pursuing|targeting|monitoring|watching|scouting|considering)\s+({1})\b', 'interested in'),
        # A bid for B (club bid for player)
        (r'\b({0})\s+(?:bid|bidding|offered|tabled an offer|made an offer|submitted a bid)\s+(?:for|on)\s+({1})\b', 'bid for'),
        # A rejected B (club or player rejected offer)
        (r'\b({0})\s+(?:rejected|declined|turned down|refused)\s+(?:an offer from|a bid from|a move to)?\s+({1})\b', 'rejected'),
        # A wants to leave/join B
        (r'\b({0})\s+(?:wants|hoping|looking|eager|keen)\s+to\s+(?:leave|join|transfer to|sign for|move to)\s+({1})\b', 'wants to join'),
        # A valued at/set asking price
        (r'\b({0})\s+(?:valued|priced|available|asking|demanding|quoted a price)\s+(?:at|for|of)\s+[£€$\d].*?\s+(?:from|by|to)\s+({1})\b', 'transfer valuation'),
        # A released/let go by B
        (r'\b({0})\s+(?:released|let go|offloaded|sold|transferred)\s+(?:by|from)\s+({1})\b', 'released by'),
        # A loan/loaned to B
        (r'\b({0})\s+(?:loan|loaned|on loan|going on loan)\s+(?:to|at|with)\s+({1})\b', 'loaned to'),
        # A in talks with B
        (r'\b({0})\s+(?:in talks|negotiating|discussing terms|in negotiations|discussing|negotiating terms)\s+(?:with)\s+({1})\b', 'in talks with'),
        # A manager/coach at B
        (r'\b({0})\s+(?:manager|coach|head coach|coaching|manages|managing)\s+(?:at|of|for)\s+({1})\b', 'manages'),
        # A and B (generic co-occurrence for rumors)
        (r'\b({0})\s+and\s+({1})\b', 'mentioned together')
    ]
    
    # Find entity pairs that frequently co-occur in sentences
    logging.info("Extracting relationships between entities...")
    
    entity_pairs = list(combinations(entities.keys(), 2))
    
    for i, (entity1, entity2) in enumerate(entity_pairs):
        show_progress(i+1, len(entity_pairs))
        
        relationships_found = []
        variants1 = [v.lower() for v in entities[entity1]["variants"]]
        variants2 = [v.lower() for v in entities[entity2]["variants"]]
        
        # Check for co-occurrence in sentences
        co_occurrences = []
        for sent in doc_spacy.sents:
            sent_text = sent.text.lower()
            
            if any(v1 in sent_text for v1 in variants1) and any(v2 in sent_text for v2 in variants2):
                co_occurrences.append(sent.text)
        
        # If we have co-occurrences, analyze them for relationship patterns
        if co_occurrences:
            # Check for specific relationship patterns
            for co_text in co_occurrences:
                for var1 in entities[entity1]["variants"]:
                    for var2 in entities[entity2]["variants"]:
                        for pattern_template, rel_type in relationship_patterns:
                            # Create pattern for both directions
                            pattern1 = pattern_template.format(re.escape(var1), re.escape(var2))
                            pattern2 = pattern_template.format(re.escape(var2), re.escape(var1))
                            
                            # Check first direction (entity1 -> entity2)
                            matches = re.search(pattern1, co_text, re.IGNORECASE)
                            if matches:
                                relationships_found.append({
                                    "type": rel_type,
                                    "direction": f"{entity1} -> {entity2}",
                                    "evidence": co_text
                                })
                            
                            # Check second direction (entity2 -> entity1)
                            matches = re.search(pattern2, co_text, re.IGNORECASE)
                            if matches:
                                relationships_found.append({
                                    "type": rel_type,
                                    "direction": f"{entity2} -> {entity1}",
                                    "evidence": co_text
                                })
        
        # If no specific relationships found but co-occurrences exist, add generic association
        if not relationships_found and len(co_occurrences) >= 2:
            relationships_found.append({
                "type": "co-mentioned",
                "direction": "bidirectional",
                "evidence": f"Co-mentioned in {len(co_occurrences)} sentences",
                "co_occurrence_count": len(co_occurrences)
            })
        
        # Add relationships if found
        if relationships_found:
            rel_key = f"{entity1}|{entity2}"
            relationships[rel_key] = relationships_found
    
    print()  # Newline after progress bar
    return relationships


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
    Check if an entity is valid by examining its mentions in the football context
    
    Args:
        entity: Entity name to validate
        doc: Spacy document
        
    Returns:
        Boolean indicating if entity is valid
    """
    # Check if entity appears in grammatical contexts typical for football entities
    entity_lower = entity.lower()
    
    # Football-specific indicators that might precede players, clubs, or agents
    football_indicators = [
        "player", "striker", "forward", "midfielder", "defender", "goalkeeper", "keeper", 
        "coach", "manager", "club", "team", "signed", "transfer", "bid", "offer",
        "from", "to", "contract", "deal", "agent", "representative", "sources",
        "league", "championship", "cup", "said", "according"
    ]
    
    # Count how many times the entity appears after specific indicators
    indicator_count = 0
    
    for token in doc:
        # Check if next tokens form our entity
        if token.i < len(doc) - 1:
            next_tokens = doc[token.i + 1:token.i + 6]  # Look at next few tokens
            next_text = "".join([t.text_with_ws for t in next_tokens])
            
            if re.search(r'\b' + re.escape(entity) + r'\b', next_text, re.IGNORECASE):
                # Check if current token is a football indicator
                if token.text.lower() in football_indicators:
                    indicator_count += 1
    
    # If entity appears after football indicators multiple times, it's likely valid
    if indicator_count >= 2:
        return True
        
    # Entity is valid if it appears as part of named entities multiple times
    entity_in_ner_count = sum(1 for ent in doc.ents if entity_lower in ent.text.lower())
    
    return entity_in_ner_count >= 2


def process_transcription(transcription: str) -> Dict[str, Dict[str, any]]:
    """
    Process the football transcription text to extract entities (players, clubs, agents), 
    their transfer contexts, and transfer/rumor relationships
    
    Args:
        transcription: The text to process for football entity extraction
        
    Returns:
        Structured dictionary with players, clubs, name variants, relevant contexts, and transfer relationships
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
    
    logging.info("Extracting smart context for each entity mention...")
    total_entities = len(name_groups)
    
    for idx, (canonical_name, variants) in enumerate(name_groups.items()):
        show_progress(idx+1, total_entities, proc_start_time)
        
        # Get all context mentions for this entity and its variants
        all_contexts = []
        for variant in variants:
            # Use enhanced smart context extraction
            contexts = extract_smart_context(transcription, variant, doc_spacy)
            all_contexts.extend(contexts)
        
        # Sort contexts by position in text
        all_contexts.sort(key=lambda x: x["position"])
        
        # Apply mention clustering to identify topics
        clustered_mentions = cluster_mentions(canonical_name, all_contexts)
        
        result["players"][canonical_name] = {
            "variants": variants,
            "mentions": clustered_mentions
        }
    
    print()  # Newline after progress bar
    
    # Extract relationships between entities
    logging.info("Extracting relationships between entities...")
    relationships = extract_relationships(result["players"], doc_spacy)
    result["relationships"] = relationships
    
    # Calculate relationship summary stats
    num_relationships = len(relationships)
    num_entities_with_relationships = len(set([entity for rel_key in relationships.keys() for entity in rel_key.split('|')]))
    
    logging.info(f"Extracted {num_relationships} relationships involving {num_entities_with_relationships} entities")
    
    proc_time = time.time() - proc_start_time
    logging.info(f"Processing completed in {timedelta(seconds=int(proc_time))}")
    
    return result


# Add parent directory to path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
nlp = spacy.load("en_core_web_trf") 

# Iterate over JSON files in directory
files = os.listdir('video_data\\result')

for filename in files:
    if filename.endswith('.json'):
        path = os.path.join('video_data\\result', filename)
        logging.info(f"Processing {filename}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                video = json.load(f)
                transcription = video['transcription_ponctuer']
                
                start_time = time.time()
                
                # Reset extracted_players_info to empty
                video["player_context"] = {}
                
                # Process the transcription to extract entities, contexts, and relationships
                logging.info("Starting entity extraction, context processing, and relationship analysis...")
                extracted_data = process_transcription(transcription)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                video["player_context"] = extracted_data
                
                # Summary of results
                total_entities = len(extracted_data["players"])
                total_mentions = sum(len(data["mentions"]) for data in extracted_data["players"].values())
                total_relationships = len(extracted_data.get("relationships", {}))
                
                logging.info(f'Processing completed in {timedelta(seconds=int(processing_time))}')
                logging.info(f'Extracted {total_entities} unique entities with {total_mentions} mentions')
                logging.info(f'Identified {total_relationships} relationships between entities')
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(video, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")