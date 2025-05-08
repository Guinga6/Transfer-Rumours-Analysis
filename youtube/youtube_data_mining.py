import yt_dlp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from youtube_comment_downloader import YoutubeCommentDownloader
from utils.mongoDB import MongoDBHandler  


def get_youtube_videos(channel_url, start_results=0,end_results= 1):
    """
    Get the latest videos from a YouTube channel using yt-dlp
    """
    if start_results> end_results:
        raise IndexError
    if not isinstance(start_results, int) or not isinstance(end_results, int):
        raise TypeError("Both variables must be integers.")

    ydl_opts = {
        'ignoreerrors': True,
        'extract_flat': True,
        'force_generic_extractor': False,
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"{channel_url}/videos", download=False)
            
            if 'entries' not in result:
                return []
            
            videos = []
            for entry in list(result['entries'])[start_results:end_results]:
                upload_date = (
                    entry.get('upload_date') or
                    entry.get('release_date') or
                    entry.get('timestamp')
                )
                
                if isinstance(upload_date, str) and len(upload_date) == 8:
                    formatted_date = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                elif isinstance(upload_date, (int, float)):
                    from datetime import datetime
                    formatted_date = datetime.utcfromtimestamp(upload_date).strftime('%Y-%m-%d')
                else:
                    formatted_date = 'Unknown'
                
                videos.append({
                    'title': entry.get('title', 'No title'),
                    'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                    'video_id': entry.get('id', ''),
                    'published_at': formatted_date,
                    'channel': entry.get('channel', 'Fabrizio Romano')
                })
            print(videos)
            return videos
    except Exception as e:
        print(f"Error: {e}")
        return []




def download_audio(video_url,output_name='%(title)s'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',       # Correct key for audio extraction
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_name}.%(ext)s',
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# # Example usage
# download_audio('https://www.youtube.com/watch?v=hIuJySt7zAI')
def fetch_and_store_youtube_comments(video_url, db, collection_name,video_title):
    """
    Fetch and store YouTube video comments in the MongoDB database.

    :param video_url: The URL of the YouTube video
    :param db: MongoDB database object
    :param collection_name: MongoDB collection where comments will be stored
    """
    # Initialize the MongoDB handler
    handler = MongoDBHandler(db)

    # Fetch the comments from the video
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url)

    if not comments:
        print(f"No comments found or failed to fetch for: {video_url}")
        comments = []

    return comments

    # Prepare the comment data, grouped under the URL key
    # comment_data = {
    #     'url': video_url,
    #     'title': video_title,
    #     'comments': [{'comment_text': comment['text']} for comment in comments]  # Store only the comment text
    # }

    # try:
    #     # Insert or update the document based on the video URL
    #     handler.update_one(collection_name, {'url': video_url}, {'$set': comment_data}, upsert=True)
    #     print(f"Successfully stored comments for video: {video_url}")
    # except Exception as e:
    #     print(f"Error storing comments: {e}")

# Example usage
# Assuming 'db' is your MongoDB database instance and 'comments_collection' is the collection name
# fetch_and_store_youtube_comments('https://www.youtube.com/watch?v=MBTciBUnPvI', db, 'comments_collection')


# # Example usage
# fetch_youtube_comments('https://www.youtube.com/watch?v=MBTciBUnPvI')


def fetch_and_store_videos(channel_url, start_results, end_results, db, collection_name):
    try:
        # Initialize the MongoDBHandler with the provided db
        handler = MongoDBHandler(db)
        
        # Fetch the videos using the helper function
        videos = get_youtube_videos(channel_url, start_results, end_results)
        
        # Insert or update the videos into the specified collection
        for video in videos:
            handler.update_one(collection_name, {'video_id': video['video_id']}, {'$set': video}, upsert=True)

    except Exception as e:
        print(f"Error: {e}")