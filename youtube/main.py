
import sys
import os
import time
import whisper
from save_local import save_or_update_video_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from youtube_data_mining import get_youtube_videos,download_audio,fetch_and_store_youtube_comments
from utils.mongoDB import MongoDBHandler
from transformation.ollama import run_ollama_prompt,extract_json,build_system_user_prompt
from transformation.transformation import audio_to_text
import toml
from pymongo import MongoClient
import spacy
import json
# from transformation.deepseek import ChatModel

config =toml.load("config.toml")



client = MongoClient(config["secret"]["mongodb"])


db = client['Web_mining']

#model = whisper.load_model("small")

collection_name = 'fabrizio_youtube'

handler = MongoDBHandler(db)

nlp = spacy.load("en_core_web_trf") #34 # 59 +43 +3

data_existant= []

for filename in os.listdir('video_data'):
    if filename.endswith('.json'):
        data_existant.append(filename[:-5])
for filename in os.listdir('video_data/done'):
    if filename.endswith('.json'):
        data_existant.append(filename[:-5])
for filename in os.listdir('video_data/rerun'):
    if filename.endswith('.json'):
        data_existant.append(filename[:-5])

videos = get_youtube_videos(playlist_url="https://www.youtube.com/playlist?list=PLpo-zNXD5plEYSSU2m-5t7kGRiWvJuH7H",start_results=0,end_results=400)
print("*********************************\n******************************\n*************************")
print("*********************************\n******************************\n*************************")
print("*********************************\n******************************\n*************************")
print("*********************************\n******************************\n*************************")
print("*********************************\n******************************\n*************************")
print(f'there is the len() of the list: {len(set(data_existant))}')
print(f'the type of vidos is {type(videos)}')
print(f'{len(videos)}')
print(f'the firt item in the list:{videos[-1]}')
for video in videos:
    if video['video_id'] in data_existant:
        continue  
    try:
        start_time = time.time()
        # Step 1: Download audio of the video
        download_audio(video_url=video['url'])

        # Step 2: Fetch and store YouTube comments
        comments = fetch_and_store_youtube_comments(video_url=video['url'],db=db,collection_name=collection_name,video_title=video['title'])
        comments = list(comments)
        comments = [comment['text'] for comment in comments if 'text' in comment]

        # Step 3: Transcribe audio to text
        text = audio_to_text(audio_path=video['title']+'.mp3',model=model)
        
        # doc = nlp(text)
        # names = list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON"))

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
            
        # Step 7: Store the relevant data in the MongoDB collection
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
        save_or_update_video_data(
            video['video_id'],
            video_data
        )
        
        print(f"Data for {video['title']} stored in MongoDB.")


    except Exception as e:
        print(f"Error processing video {video['title']}: {e}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")

