import os
import json
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO
from difflib import SequenceMatcher
import time

def print_progress_bar(current, total, prefix='Progress', suffix='Complete', length=50):
    """Print a progress bar to show processing status"""
    percent = f"{100 * (current / float(total)):.1f}"
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if current == total: 
        print()  # New line when complete
def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_closest_audio_file(title, audio_files):
    """Find the closest matching audio file based on title"""
    best_match = None
    best_score = 0
    
    # Clean the title for better matching
    title_clean = os.path.splitext(title)[0].lower()
    
    for audio_file in audio_files:
        # Clean the audio filename for comparison
        audio_clean = os.path.splitext(audio_file)[0].lower()
        
        # Calculate similarity
        score = similarity(title_clean, audio_clean)
        
        if score > best_score:
            best_score = score
            best_match = audio_file
    
    return best_match, best_score



def detect_segments(audio_path):
    """Detect speech and silence segments in audio"""
    try:
        [sampling_rate, signal] = audioBasicIO.read_audio_file(audio_path)

        flags = aS.silence_removal(signal,
                                   sampling_rate,
                                   st_win=0.02,
                                   st_step=0.02,
                                   smooth_window=0.5,
                                   weight=0.3,
                                   plot=False)

        speech_segments = flags
        silence_segments = []

        for i in range(len(speech_segments) - 1):
            silence_start = speech_segments[i][1]
            silence_end = speech_segments[i + 1][0]
            silence_segments.append([silence_start, silence_end])

        return speech_segments, silence_segments

    except Exception as e:
        print(f"Erreur lors de l'analyse audio: {e}")
        return [], []

def get_json_files_with_titles(json_folder):
    """Get all JSON files and extract their titles"""
    json_files_info = {}
    
    for file in os.listdir(json_folder):
        if file.lower().endswith(".json"):
            json_path = os.path.join(json_folder, file)
            try:
                with open(json_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    if "title" in data:
                        json_files_info[file] = {
                            "title": data["title"],
                            "path": json_path,
                            "data": data
                        }
                    else:
                        print(f"Pas de clé 'title' trouvée dans {file}")
            except Exception as e:
                print(f"Erreur lors de la lecture de {file}: {e}")
    
    return json_files_info

def get_audio_files(audio_folder):
    """Get all audio files from the folder"""
    audio_files = []
    for file in os.listdir(audio_folder):
        if file.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
            audio_files.append(file)
    return audio_files

def update_json_files(json_folder, audio_folder, similarity_threshold=0.3):
    """Update each JSON file with audio analysis data"""
    
    # Get all JSON files with their titles
    json_files_info = get_json_files_with_titles(json_folder)
    
    if not json_files_info:
        print("Aucun fichier JSON avec une clé 'title' trouvé!")
        return
    
    # Get all audio files
    audio_files = get_audio_files(audio_folder)
    
    if not audio_files:
        print("Aucun fichier audio trouvé!")
        return
    
    total_files = len(json_files_info)
    print(f"Trouvé {total_files} fichiers JSON et {len(audio_files)} fichiers audio")
    print(f"Début du traitement...\n")
    
    matched_count = 0
    processed_count = 0
    start_time = time.time()
    
    for json_file, info in json_files_info.items():
        title = info["title"]
        json_path = info["path"]
        data = info["data"]
        
        # Update progress
        processed_count += 1
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / processed_count if processed_count > 0 else 0
        remaining_files = total_files - processed_count
        eta = avg_time_per_file * remaining_files
        
        # Print progress bar with ETA
        eta_str = f"ETA: {eta:.1f}s" if eta > 0 else "ETA: --"
        print_progress_bar(
            processed_count, 
            total_files, 
            prefix=f'Traitement ({processed_count}/{total_files})', 
            suffix=f'{eta_str}'
        )
        
        # Find the closest matching audio file
        closest_audio, score = find_closest_audio_file(title, audio_files)
        
        if closest_audio and score >= similarity_threshold:
            audio_path = os.path.join(audio_folder, closest_audio)
            print(f"\n✓ Correspondance: '{title[:50]}...' -> '{closest_audio}' (score: {score:.2f})")
            
            # Analyze the audio
            print(f"  Analyse audio en cours...")
            speech_segments, silence_segments = detect_segments(audio_path)
            
            if speech_segments or silence_segments:
                # Update the JSON data
                data["audio_analysis"] = {
                    "matched_audio_file": closest_audio,
                    "matching_score": score,
                    "speech_segments": speech_segments,
                    "silence_segments": silence_segments
                }
                
                # Save the updated JSON file
                try:
                    with open(json_path, "w", encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"  ✓ JSON mis à jour: {json_file}")
                    matched_count += 1
                except Exception as e:
                    print(f"  ✗ Erreur sauvegarde {json_file}: {e}")
            else:
                print(f"  ✗ Échec analyse audio pour {closest_audio}")
        else:
            best_score_str = f"{score:.2f}" if closest_audio else "0.00"
            print(f"\n✗ Aucune correspondance pour '{title[:50]}...' (meilleur: {best_score_str})")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"RÉSUMÉ FINAL:")
    print(f"Fichiers traités: {processed_count}/{total_files}")
    print(f"Correspondances trouvées: {matched_count}")
    print(f"Temps total: {total_time:.1f}s")
    print(f"Temps moyen par fichier: {total_time/processed_count:.1f}s")
    print(f"Taux de réussite: {(matched_count/total_files)*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    # Dossiers à modifier selon vos besoins
    folder_audio = "video"      # Dossier contenant les fichiers audio
    folder_json = "video_data"  # Dossier contenant les fichiers JSON
    
    # Seuil de similarité (0.3 = 30% de similarité minimum)
    similarity_threshold = 0.3
    
    if not os.path.exists(folder_audio):
        print(f"Le dossier audio '{folder_audio}' n'existe pas!")
    elif not os.path.exists(folder_json):
        print(f"Le dossier JSON '{folder_json}' n'existe pas!")
    else:
        update_json_files(folder_json, folder_audio, similarity_threshold)