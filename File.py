import torch
from faster_whisper import WhisperModel
import time

class AudioTranscriber:
    def __init__(self, model_size="large-v3"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compute_type = 'int8' if self.device == 'cuda' else 'float32'
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(audio_path)
        language = info.language
        probability = info.language_probability

        transcript = " ".join(segment.text for segment in segments)
        return transcript, language, probability

def main():
    start_time = time.time()
    audio_path = "data/english.mp3"
    transcriber = AudioTranscriber()
    transcript, language, probability = transcriber.transcribe(audio_path)
    print(f"Transcript: {transcript}, Language: {language}, Probability: {probability}")
    print(f"Time taken: {time.time() - start_time}")

if __name__ == "__main__":
    main()
