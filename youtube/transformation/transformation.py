
from utils.mongoDB import MongoDBHandler

from faster_whisper import WhisperModel

def audio_to_text(audio_path,model):
    """
    Transcribes audio to text and returns the transcription text.

    :param audio_path: Path to the audio file (e.g., .mp3)
    :return: The transcribed text
    """
    

    try:
        # Transcribe the audio with language specified (e.g., English)
        result = model.transcribe(audio_path, language="en")

        # Return the transcription text
        return result["text"]

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


# Example usage:
# audio_to_text("your_audio_file.mp3", db, "transcriptions")
