import whisper

# Load the model
model = whisper.load_model("small")

# Transcribe the audio, specify language
result = model.transcribe("🚨 HAALAND CLAUSE, MAN UTD MIDFIELD DREAM, ARDA GULER’S FUTURE, NEW GEM… [VdPn6j2WrQg].mp3", language="en")  # Use "en" if English is more prominent

# Save the transcription
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
