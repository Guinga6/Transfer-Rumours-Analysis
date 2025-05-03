
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from youtube_data_mining import get_youtube_videos,download_audio,fetch_and_store_youtube_comments
from utils.mongoDB import MongoDBHandler
from transformation.ollama import run_ollama_prompt, from_str_to_dict
from transformation.transformation import audio_to_text
import toml
from pymongo import MongoClient

config =toml.load("config.toml")


client = MongoClient(config["secret"]["mongodb"])


db = client['your_database_name']


collection_name = 'fabrizio_youtube'

handler = MongoDBHandler(db)


prompt ="""You are an expert language evaluator and information extractor.
You will receive a transcript of spoken audio, usually from football-related content where players are discussed.

Your task:

Language Check: Evaluate how likely the text is written in English. Return a score between 0 and 100 (0 = not English, 100 = perfect English).

Connectivity Check: Evaluate how logically connected and grammatically coherent the sentences are. Return a score between 0 and 100.

Player Extraction: Identify full names of football players mentioned in the text. For each player, return the exact quote from the input where the player is mentioned — do not summarize or rephrase.

Important: You must return a response strictly in valid JSON format as follows. Do not include any explanation, comments, or extra output outside this structure:
{
  "is_english": <int between 0-100>,
  "is_connected": <int between 0-100>,
  "players": {
    "player_1": "exact quote from text",
    "player_2": "exact quote from text"
    }
  }
  here is the transcript: """

videos = get_youtube_videos(channel_url="https://www.youtube.com/@FabrizioRomanoYT",start_results=0,end_results=5)

for video in videos:
    try:
        # Step 1: Download audio of the video
        download_audio(video_url=video['url'])
        
        # Step 2: Fetch and store YouTube comments
        comments = fetch_and_store_youtube_comments(video_url=video['url'])
        
        # Step 3: Transcribe audio to text
        text = audio_to_text(audio_path=video['title'])
        
        # Step 4: Run Ollama model with prompt and transcribed text
        output = run_ollama_prompt(model_name="qwen2.5:3b-instruct-q4_K_M", prompt=prompt + text)
        
        # Step 5: Convert the model output from string to dictionary
        output_to_dict = from_str_to_dict(model_output=output)

        # Step 6: Check if the output meets the criteria (English > 0.85, Logical > 0.85)
        if float(output_to_dict.get('is_english', 0)) > 0.85 and float(output_to_dict.get('is_logical', 0)) > 0.85:
            
            # Step 7: Store the relevant data in the MongoDB collection
            video_data = {
                "url": video['url'],
                "timestamps": video['published_at'],
                "comments": comments,
                "transcription": text,
                "ollama_output": output_to_dict
            }

            # Step 8: Insert the data into MongoDB
            handler.update_one(
                collection_name,
                {"url": video['url']},  # Assuming video URL is unique
                {"$set": video_data},
                upsert=True
            )
            
            print(f"Data for {video['title']} stored in MongoDB.")

        else:
            print(f"Skipped {video['title']} due to low English or logical score.")

    except Exception as e:
        print(f"Error processing video {video['title']}: {e}")
