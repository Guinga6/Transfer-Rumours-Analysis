# Analyse des Rumeurs de Transferts et des Performances des Joueurs

## 🎯 Objectif
Analyser les rumeurs de transferts et les performances réelles des joueurs à partir de sources en ligne (YouTube, Transfermarkt, WhoScored) à l’aide de techniques de NLP, d’analyse de sentiment et de machine learning.

## 🔧 Technologies
- **Python** : scripting, scraping et traitement des données
- **Apache Airflow** : orchestration du pipeline ETL
- **Docker** : déploiement conteneurisé
- **MongoDB Atlas** : stockage des données structurées (JSON/CSV)
- **Whisper** : transcription audio
- **SpaCy, fuzzywuzzy** : extraction & nettoyage des entités
- **VaderSentiment** : analyse de sentiment
- **XGBoost** : classification des rumeurs
- **Power BI / Streamlit** : visualisation des résultats

## L'architecture de projet:
```bash
   Transfers_Rumors/
   ├── data/
   │   ├── transfer_market/
   │   │   ├── players.csv
   │   │   ├── rumours.csv
   │   │   ├── rumours_detail.csv
   │   │   ├── transfermarkt.csv
   │   │   └── transfermarkt_latest.csv
   │   ├── whoscored/
   │   │   ├── bundesliga_players_20250524_1545.xlsx
   │   │   ├── la_liga_players_20250524_1433.xlsx
   │   │   ├── mancity_data.xlsx
   │   │   ├── premier_league_players_20250521_1529.xlsx
   │   │   ├── serie_a_players_20250524_1538.xlsx
   │   │   └── westham_players_20250521_1515.xlsx
   │   ├── youtube/ # les donner collecter depuis youtube 
   │   │   ├── csv/
   │   │   │   ├── rumors.csv
   │   │   │   └── rumors_fabrizio_romano.csv
   │   │   └── json/
   │   │       ├── 9HmRqoTp6Yw.json
   │   │       └── Read me
   │   └── __init__.py
   ├── model/
   │   ├── classification_des_rumors.ipynb
   |   ├── sentence_boundary_model.pkl
   |   ├── model_without_sentiment.pkl
   |   ├── model_with_sentiment.pkl
   │   └── sentiment.ipynb
   ├── transfermarkt/
   │   ├── dags/
   │   │   ├── ExtractRumourDetail.py
   │   │   ├── ExtractRumours.py
   │   │   ├── RumourDetail.py
   │   │   ├── etltransfermarket.py
   │   │   ├── player.py
   │   │   └── rumour.py
   │   ├── Dockerfile
   │   ├── README.md
   │   ├── docker-compose.yml
   │   ├── file.py
   │   ├── packages.txt
   │   └── requirements.txt
   ├── whoscored/
   │   └── scripts/
   │       ├── bundesliga.py
   │       ├── dashboard.py
   │       ├── laliga.py
   │       ├── ligue1.py
   │       ├── premier_league.py
   │       └── serie_a.py
   ├── youtube/ # les scripts de youtube
   │   ├── date/ # les scripts qui ont etait utiliser pour ajouter la data pour chaque video
   │   │   ├── adding_date2files.py
   │   │   ├── finding_the_upload_data.py
   │   │   └── songs_metadata.json
   │   ├── evaluation/ 
   │   │   └── text_evaluation.py # evaluer deux transcription
   │   ├── transformation/
   │   │   ├── __init__.py
   │   │   ├── json2csv.py # transfromer les files json vers csv
   │   │   ├── ollama.py # utiliser ollama 
   │   │   ├── ponctuation.py # ajouter la ponctuation aux transcription
   │   │   ├── text2names.py # appliquer l extraction des contexte inteligent 
   │   │   └── transformation.py # transfromer un audio a un texte
   │   ├── utils/
   │   │   ├── __init__.py
   │   │   ├── mongoDB.py # utiliser mongoDB
   │   │   └── progress.py # un simple scripts pour voir les progress du code
   │   ├── verification/
   │   │   ├── __init__.py
   │   │   └── audio_2_text.py # un scripts pour verifier si tous les video on un transcription
   │   ├── audio_meta_data.py # extraction des meata donner des audio
   │   ├── clean_repition.py # supprimer les texte dupliquer
   │   ├── main.py # les main script pour crawler et scraper les audio puis les transfromer a un texte
   │   ├── mistral.py # le code qui utilise ollama (mistral-7b) dans la transformation
   │   ├── save_local.py # un simple code pour stocker les resultat dans local
   │   ├── transcription_is_Null.py # refaire la transformation des audio vers texte si la transcription est null
   │   └── youtube_data_mining.py #le scripts qui contient les main fonction utliser pour scraper et crawler les vdieo en utilisant yt-dlp et youtube-comment-downloader
   ├── README.md
```

## 📦 Pipeline du Projet
1. **Collecte des données**  
   - Vidéos et commentaires YouTube (Fabrizio Romano)
   - Statistiques de transfert (Transfermarkt)
   - Performances individuelles (WhoScored)

2. **Prétraitement**
   - Transcription audio (Whisper)
   - Détection des entités nommées (SpaCy)
   - Nettoyage et normalisation des noms

3. **Stockage**
   - Données stockées dans MongoDB Atlas (`TopTransfers`, `RumoursData`)

4. **Modélisation**
   - Analyse de sentiment (Vader)
   - Classification crédibilité rumeur (XGBoost)

5. **Visualisation**
   - Dashboard Power BI
   - Application interactive via Streamlit

## 🚀 Démarrage rapide
```bash
git clone Transfers_Rumors
cd Transfers_Rumors
```

## 👨‍💻 Auteurs
- Aly Guinga  
- Zayd El Ouaragli  
- Anas El Hassouni  

## 📜 Références
- https://github.com/jawadoch/Transfermarkt_ETL_using_Airflow  
- https://www.youtube.com/@FabrizioRomanoYT  
- https://www.whoscored.com/  
- https://www.transfermarkt.com/  
