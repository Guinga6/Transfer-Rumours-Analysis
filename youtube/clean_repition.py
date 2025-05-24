import json
import os
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.progress import show_progress

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_contexts(text1, text2):
    if text2 in text1:
        return text1
    if text1 in text2:
        return text2
    max_overlap = 0
    for i in range(1, min(len(text1), len(text2))):
        if text1[-i:] == text2[:i]:
            max_overlap = i
    return text1 + text2[max_overlap:]

def process_plt(plt):
    new_plt = {}
    total = 0
    merged_count = 0
    unmerged_count = 0

    for name, entry in plt.items():
        variants = entry.get("mentions", [])
        total += len(variants)
        merged = []
        used = set()

        for i, v1 in enumerate(variants):
            if i in used:
                continue
            merged_flag = False
            merged_variant = v1.copy()
            for j, v2 in enumerate(variants):
                if i != j and j not in used:
                    if v1["cluster"] == v2["cluster"] and v1["topic"] == v2["topic"]:
                        merged_variant["context"] = merge_contexts(merged_variant["context"], v2["context"])
                        used.add(j)
                        merged_flag = True
            if merged_flag:
                merged_count += 1
            else:
                unmerged_count += 1
            merged.append(merged_variant)

        new_plt[name] = {"variants": merged}

    logging.info(f"Summary:\nTotal contexts: {total}\nMerged: {merged_count}\nUnmerged: {unmerged_count}")
    return new_plt


# Iterate over JSON files in directory
files = os.listdir('video_data')
total = len(files)
for (idx,filename) in enumerate(files,1):
    show_progress(idx, total)
    if filename.endswith('.json'):
        print(filename)
        i = 0
        logging.info(f'processing file {filename}')
        done =False
        path = os.path.join('video_data', filename)
        with open(path, 'r', encoding='utf-8')  as f:
            data = json.load(f)

        data["clean_player"] = process_plt(data["extracted_players_info"]["players"])

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        