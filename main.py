# def main():
#     # Correct Fabrizio Romano's channel URL
#     channel_url = "https://www.youtube.com/@FabrizioRomanoYT"
    
#     # Generate a filename with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"fabrizio_romano_videos_{timestamp}.csv"
    
#     # Get the videos
#     print(f"Fetching the latest 5 videos from {channel_url}...")
#     videos = get_latest_youtube_videos(channel_url)
    
#     if videos:
#         print(f"Found {len(videos)} videos")
        
#         # Display videos in console
#         for i, video in enumerate(videos, 1):
#             print(f"{i}. {video['title']}")
#             print(f"   URL: {video['url']}")
#             print(f"   Published: {video['published_at']}")
#             print("---")
        
#         # Save to CSV
#         save_to_csv(videos, filename)
#     else:
#         print("No videos found")

# if __name__ == "__main__":
#     main()