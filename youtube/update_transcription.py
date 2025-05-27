import json
import os
from typing import List, Dict
from fuzzywuzzy import fuzz

# Helper function for progress
def show_progress(current, total):
    print(f"\rProcessing: {current}/{total}", end='')

# Deduplication function applied to words in transcription
def deduplicate_names(names: List[str], threshold: int = 90) -> Dict[str, List[str]]:
    if not names:
        return {}
    name_groups = {}
    processed = set()
    for i, name in enumerate(names):
        if name in processed:
            continue
        show_progress(i + 1, len(names))
        canonical = name
        variants = [name]
        processed.add(name)
        for other in names:
            if other != name and other not in processed:
                similarity = fuzz.ratio(name.lower(), other.lower())
                if similarity >= threshold:
                    variants.append(other)
                    processed.add(other)
        if len(variants) > 1:
            canonical = max(variants, key=variants.count)
        name_groups[canonical] = variants
    print()
    return name_groups

# Ensure directory exists
os.makedirs("video_data", exist_ok=True)

# Load corrections
with open('corrections.json', 'r', encoding='utf-8') as f:
    corrections = json.load(f)

video_data_folder = 'video_data'
not_found_keys = set(corrections.keys())
all_transcription_words = set()

# Step 1: Gather all words in transcriptions
for filename in os.listdir(video_data_folder):
    if filename.endswith('.json'):
        path = os.path.join(video_data_folder, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'transcription' in data:
            words = data['transcription'].split()
            all_transcription_words.update(words)

# Step 2: Deduplicate words
deduplicated_groups = deduplicate_names(list(all_transcription_words), threshold=90)

# Step 3: Build variant-to-correction mapping
variant_to_correct = {}
for canonical, variants in deduplicated_groups.items():
    if canonical in corrections:
        corrected = corrections[canonical]
        for variant in variants:
            variant_to_correct[variant] = corrected
        not_found_keys.discard(canonical)

# Step 4: Apply corrections to transcriptions
for filename in os.listdir(video_data_folder):
    if filename.endswith('.json'):
        path = os.path.join(video_data_folder, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'transcription' in data:
            text = data['transcription']
            for wrong, correct in variant_to_correct.items():
                text = text.replace(wrong, correct)
            data['transcription'] = text
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

# ✅ Summary
print(f"\n✅ Completed.\n🔍 Corrections attempted: {len(corrections)}")
print(f"❌ Corrections not applied (no match found): {len(not_found_keys)}")
if not_found_keys:
    print(f"📝 Unmatched keys: {sorted(not_found_keys)}")
