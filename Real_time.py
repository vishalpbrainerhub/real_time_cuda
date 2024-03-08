# this it normal code check the async_real_time.py for the async version of this code

import pyaudio
import wave
import os
import numpy as np
import time
import random
import sys
import torch
import logging
from faster_whisper import WhisperModel
from pytube import YouTube
import subprocess
from pydub import AudioSegment
import asyncio

# Setting up basic logging for debug messages and info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioTranscriber:
    def __init__(self, model_size="large-v3"):
        # Initializes the transcriber with a specific Whisper model size, sets up the device (GPU or CPU), and logging
        self.model_size = model_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Chooses CUDA if available, else CPU
        self.compute_type = 'int8_float16' if self.device == 'cuda' else 'float32'  # Optimizes model for speed/precision based on device
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)  # Loads the Whisper model
        self.p = pyaudio.PyAudio()  # Initiates the PyAudio class for audio recording

        logging.info(f"Device: {self.device}, Compute Type: {self.compute_type}")



    def record_chunk(self, stream, chunk_length=1):
        # Records a chunk of audio data from the microphone and saves it to a temporary file
        frames = []
        for _ in range(0, int(16000 / 1024 * chunk_length)):  # Calculates the number of frames needed based on the chunk length
            data = stream.read(1024, exception_on_overflow=False)  # Reads audio data from the stream
            frames.append(data)

        # chunk_file = f'temp_chunk_{random.randint(0, 1000000)}.wav'
        chunk_file = 'temp_chunk.wav'  # Temporary file name
        wf = wave.open(chunk_file, 'wb')
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)  # Standard sample rate for speech recognition
        wf.writeframes(b''.join(frames))
        wf.close()
        return chunk_file
    


    def transcribe_chunk(self, chunk_file):
        # Transcribes the audio in a chunk file using Whisper and logs the detected language and transcription
        segments, info = self.model.transcribe(chunk_file, beam_size=5)  # Transcribes the chunk
        transcription = " ".join([segment.text for segment in segments])  # Concatenates segments of the transcription
        logging.info(f"Detected language: {info.language} with probability {info.language_probability}")
        logging.info(f"Transcription: {transcription}")

        return transcription
    

    
    def download_youtube_audio(self, url, output_path='youtube_audio.wav'):
        
        yt = YouTube(url)
        audio_stream = yt.streams.get_audio_only()
        audio_stream.download(filename='temp_audio.mp4')
        subprocess.run(['ffmpeg', '-i', 'temp_audio.mp4', output_path])
        os.remove('temp_audio.mp4')
        return output_path
    

    async def transcribe_youtube_audio(self, url):
        # Downloads the audio from a YouTube video and transcribes it
        audio_path = self.download_youtube_audio(url)
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = 5000 
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        transcribed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            transcription = self.transcribe_chunk(chunk_path)
            transcribed_chunks.append(transcription)
            os.remove(chunk_path)
            
        os.remove(audio_path)
        return " ".join(transcribed_chunks)



    def transcribe_audio(self):
        # Main loop for recording audio in chunks and transcribing them in real-time
        stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        acc = ""  # Accumulates the transcription text

        try:
            while True:  # Continuously records and transcribes audio until manually interrupted
                logging.info("Recording chunk...")
                chunk_file = self.record_chunk(stream)
                logging.info("Recorded chunk, starting transcription...")
                
                transcription = self.transcribe_chunk(chunk_file)
                os.remove(chunk_file)  # Cleans up the temporary audio file

                acc += transcription + " "
        except KeyboardInterrupt:  # Allows the user to interrupt the loop (e.g., by pressing Ctrl+C)
            logging.info("Stopping transcription.")
            with open('transcription_log.txt', 'w') as f:  # Saves the accumulated transcription to a file
                f.write(acc)
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()  # Properly closes the audio stream and PyAudio to free system resources



if __name__ == '__main__':
    transcriber = AudioTranscriber()  # Instantiates the transcriber
    start_time = time.time()
    if len(sys.argv) > 1 and sys.argv[1].startswith('http'):
        youtube_url = sys.argv[1]
        # Use asyncio.run to run the asynchronous function
        transcription = asyncio.run(transcriber.transcribe_youtube_audio(youtube_url))
        print(transcription)
    else:
        transcriber.transcribe_audio()