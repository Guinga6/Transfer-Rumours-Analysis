import os
import json
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.progress import show_progress

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Sample corrections mapping (add more as needed)
corrections ={
    # P
    "Petrović": "Đorđe Petrović",  # Chelsea goalkeeper
    "Pierre Kalulu": "Pierre Kalulu",  # Already correct (AC Milan)
    "Pininz Zahavi": "Pini Zahavi",  # Famous agent
    "Player": "Club",  # Placeholder
    "Player_name": "Club",  # Placeholder
    "Poubill": "Boubacar Kamara",  # Likely typo for Villa player
    "ROsimhen": "Victor Osimhen",
    
    # R
    "Rafa Ela Pimenta(Rafa Pimenta)": "Rafaela Pimenta",  # Agent
    "Rafa Rafael Leão": "Rafael Leão",
    "Rafael Varan": "Raphaël Varane",
    "Raoucho": "Rauch",  # Likely staff member
    "Rasmus": "Rasmus Højlund",  # Most likely
    "Rasmus Oylund": "Rasmus Højlund",
    "Raul Real Sociedad": "Takefusa Kubo",  # Likely referring to Sociedad player
    "Real Madridly": "Real Madrid",
    "Real Madridly, Trent Alexander-Arnold": "Trent Alexander-Arnold",  # Remove incorrect club
    "Reinders": "Tijani Reijnders",
    "Ren": "Renato Veiga",  # Shortened
    "Ricci Massara": "Ricky Massara",  # Milan director
    "Richard Dugues": "Richard Dunne",  # Likely former player
    "Riz James": "Reece James",
    "Riz Nelson": "Reiss Nelson",
    "Romelu (Romeo)": "Romelu Lukaku",
    "Romeo Luccardo (Romelu Lukaku)": "Romelu Lukaku",
    "Romero Lucaku": "Romelu Lukaku",
    "Romero Lukaku": "Romelu Lukaku",
    "Roni Araujo": "Ronald Araújo",
    "Ronny Araujo": "Ronald Araújo",
    "Roque": "Vitor Roque",
    "Rud van Nisterloy": "Ruud van Nistelrooy",
    "Ruinunias": "Rūdolfs Ruņģis",  # Unclear, guessing
    "Rushford": "Marcus Rashford",
    "Ryan Cherokee": "Ryan Cherki",  # Lyon player
    
    # S

    "Samo Omerodion": "Samu Omorodion",  # Atletico Madrid

    "Schesco": "Benjamin Šeško",
    "Sergim Ratcliffe": "Sir Jim Ratcliffe",  # INEOS owner
    "Sergio con Cessau": "Sergio Conceição",  # Porto manager
    "Serri Altemira": "Sergi Altimira",  # Barcelona youngster
    "Seydou": "Seydou Doumbia",  # Former player
    "Smith Rowe": "Emile Smith Rowe",
    "Sofjan Amrabat": "Sofyan Amrabat",
    "Stefan Bycetic": "Stefan Bajčetić",
    "Subimendi": "Martín Zubimendi",
    "Sverre Nipan": "Sverre Nypan",  # Rosenborg player
    
    # T
    "Tassari": "Mauro Tassotti",  # Likely referring to Mauro Tassotti

    
    # V/W
    "Vanderson": "Vanderson",  # Monaco (already correct)

    "Willy Cambuala": "Willyan Capemba",  # Angolan player
    
    # X/Z

    "Zerede": "Zerrouki",  # Feyenoord player
    
    # Miscellaneous

    "paul victor": "pau victor",
    "club": "Club"  # Placeholder
}

def correct_json_files(directory="./video_data/mistral"):
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    total = len(files)

    updated_files = 0
    skipped_files = 0
    start = 0

    for i, filename in enumerate(files, 1):
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            players = data.get("result", {})
            new_players = {}
            changed = False

            for key, value in players.items():
                corrected_key = corrections.get(key, key)
                if corrected_key != key:
                    logging.info(f"Renamed '{key}' -> '{corrected_key}' in {filename}")
                    changed = True
                new_players[corrected_key] = value

            if changed:
                data["result"] = new_players
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                updated_files += 1
            else:
                skipped_files += 1

        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}")

        show_progress(i, total, start, description="Updating JSON files")

    logging.info(f"Update complete: {updated_files} files updated, {skipped_files} skipped.")

correct_json_files()
