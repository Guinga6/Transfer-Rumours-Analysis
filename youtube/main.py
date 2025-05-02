from youtube_data_mining import get_youtube_videos,download_audio,fetch_and_store_youtube_comments
from utils.mongoDB import MongoDBHandler
from transformation.ollama import run_ollama_prompt, from_str_to_dict
from transformation.transformation import audio_to_text
import toml
from pymongo import MongoClient

config =toml.load("config.toml")

client = MongoClient[config['mongodb']['uri']]
db = client['your_database_name']
collection_name = 'video_data_collection'
handler=MongoDBHandler(db)

prompt =''

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
        output = run_ollama_prompt(model_name="", prompt=prompt + text)
        
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
