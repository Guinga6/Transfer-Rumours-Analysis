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
