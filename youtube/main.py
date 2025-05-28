import sys
import os
import time
import whisper
from save_local import save_or_update_video_data
from datetime import timedelta
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from youtube_data_mining import get_youtube_videos, download_audio, fetch_and_store_youtube_comments
from utils.mongoDB import MongoDBHandler
from transformation.ollama import run_ollama_prompt, extract_json, build_system_user_prompt
from transformation.transformation import audio_to_text
import toml
from pymongo import MongoClient
import spacy
import json
from utils.progress import show_progress,clear_progress
# Load configuration
config = toml.load("config.toml")

# Set up MongoDB connection
client = MongoClient(config["secret"]["mongodb"])
db = client['Web_mining']
collection_name = 'fabrizio_youtube'
handler = MongoDBHandler(db)

# Load models
model = whisper.load_model("small")
nlp = spacy.load("en_core_web_trf")

# Track existing data
data_existant = []

# Get existing video IDs
logging.info("Gathering existing video IDs...")
for filename in os.listdir('video_data'):
    if filename.endswith('.json'):
        data_existant.append(filename[:-5])
for filename in os.listdir('video_data/done'):
    if filename.endswith('.json'):
        data_existant.append(filename[:-5])
for filename in os.listdir('video_data/rerun'):
    if filename.endswith('.json'):
        data_existant.append(filename[:-5])

# Get YouTube videos
logging.info("Fetching YouTube videos from playlist...")
videos = get_youtube_videos(
    playlist_url="https://www.youtube.com/playlist?list=PLpo-zNXD5plEYSSU2m-5t7kGRiWvJuH7H",
    start_results=0,
    end_results=400
)

# Log information about retrieved videos
logging.info("=" * 50)
logging.info(f"Found {len(set(data_existant))} already processed videos")
logging.info(f"Retrieved {len(videos)} videos from YouTube playlist")
logging.info(f"Latest video in list: {videos[-1]['title'] if videos else 'None'}")
logging.info("=" * 50)

# Process each video
total_videos = len(videos)
processed_count = 0
skipped_count = 0
error_count = 0
start_processing_time = time.time()

for i, video in enumerate(videos):
    show_progress(i+1, total_videos, start_processing_time,"Processing videos")
    
    if video['video_id'] in data_existant:
        skipped_count += 1
        continue
        
    try:
        start_time = time.time()
        logging.info(f"Processing video: {video['title']} ({video['video_id']})")
        
        # Step 1: Download audio of the video
        logging.info(f"Downloading audio for {video['title']}...")
        download_audio(video_url=video['url'])

        # Step 2: Fetch and store YouTube comments
        logging.info(f"Fetching comments for {video['title']}...")
        comments = fetch_and_store_youtube_comments(
            video_url=video['url'],
            db=db,
            collection_name=collection_name,
            video_title=video['title']
        )
        comments = list(comments)
        comments = [comment['text'] for comment in comments if 'text' in comment]
        logging.info(f"Retrieved {len(comments)} comments")

        # Step 3: Transcribe audio to text
        logging.info(f"Transcribing audio to text...")
        text = audio_to_text(audio_path=video['title']+'.mp3', model=model)
        logging.info(f"Transcription completed: {len(text)} characters")
        
        # doc = nlp(text)
        # names = list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON"))

        # Initialize final output
        final_output = {"players": {}}

        # # Group names into chunks of 3
        # chunks = [names[i:i+4] for i in range(0, len(names), 4)]

        # for group in chunks:
        #     prompt = build_system_user_prompt(group, text)
        #     response = run_ollama_prompt(model_name="mistral:7b-instruct", messages=prompt)
        # # response = chat_model.run_prompt(prompt)
        # try:
        #     print(response)
        #     data = extract_json(response)  
        #     for name in group:
        #         if data["players"].get(name):
        #             final_output["players"][name] = data["players"][name]
        # except Exception as e:
        #     print(f"Failed to parse output for: {group} | Error: {e}")

        # print(json.dumps(final_output, indent=2))
            
        # Create video data dictionary
        video_data = {
            "url": video['url'],
            "title": video['title'],
            "comments": comments,
            "transcription": text,
            "extracted_players_info": final_output
        }
                # # Step 8: Insert the data into MongoDB
        # handler.update_one(
        #     collection_name,
        #     {"url": video['url']},  # Assuming video URL is unique
        #     {"$set": video_data},
        #     upsert=True
        # )

        # Save the data
        logging.info(f"Saving data for {video['title']}...")
        save_or_update_video_data(
            video['video_id'],
            video_data
        )
        
        processed_count += 1
        logging.info(f"Successfully processed {video['title']}")

    except Exception as e:
        error_count += 1
        logging.error(f"Error processing video {video['title']}: {str(e)}")
        
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Video processing time: {timedelta(seconds=int(execution_time))}")


# Print final summary
total_time = time.time() - start_processing_time
logging.info("\n" + "=" * 50)
logging.info("Processing Summary:")
logging.info(f"Total videos: {total_videos}")
logging.info(f"Processed: {processed_count}")
logging.info(f"Skipped (already existed): {skipped_count}")
logging.info(f"Errors: {error_count}")
logging.info(f"Total execution time: {timedelta(seconds=int(total_time))}")
logging.info("=" * 50)

