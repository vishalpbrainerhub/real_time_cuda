import asyncio
import concurrent.futures
import pyaudio
import wave
import os
import logging
from faster_whisper import WhisperModel
import torch
from pydub import AudioSegment
from pytube import YouTube
import sys

# Setting up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioTranscriber:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compute_type = 'int8_float16' if self.device == 'cuda' else 'float32'
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        self.p = pyaudio.PyAudio()
        logging.info(f"Device: {self.device}, Compute Type: {self.compute_type}")

    def record_chunk(self, stream, chunk_length=2):
        frames = []
        for _ in range(0, int(16000 / 1024 * chunk_length)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        chunk_file = f'temp_chunk.wav'
        wf = wave.open(chunk_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()
        return chunk_file

    async def transcribe_chunk_async(self, chunk_file):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            transcription = await loop.run_in_executor(pool, self.transcribe_chunk, chunk_file)
            # os.remove(chunk_file)
            return transcription

    def transcribe_chunk(self, chunk_file):
        segments, info = self.model.transcribe(chunk_file, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        logging.info(f"Detected language: {info.language} with probability {info.language_probability}")
        logging.info(f"Transcription: {transcription}")
        return transcription

    async def download_youtube_audio(self, url, output_path='youtube_audio.wav'):
        yt = YouTube(url)
        audio_stream = yt.streams.get_audio_only()
        temp_filename = audio_stream.download(filename='temp_audio.mp4')
        logging.info(f"Downloaded audio file: {temp_filename}")

        # Use create_subprocess_exec for executing the ffmpeg command
        process = await asyncio.create_subprocess_exec('ffmpeg', '-i', temp_filename, output_path,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate() 
        
        if process.returncode == 0:
            logging.info(f"Converted audio file saved to: {output_path}")
        else:
            logging.error(f"Failed to convert audio file. FFmpeg error: {stderr.decode()}")
            raise Exception("FFmpeg conversion failed")

        os.remove(temp_filename)
        logging.info("Temporary file removed.")
        return os.path.abspath(output_path)  # Return the absolute path to the converted file


    async def transcribe_youtube_audio(self, url):
        audio_path = await self.download_youtube_audio(url)
        if not audio_path or not os.path.exists(audio_path):
            logging.error("Failed to download or convert YouTube audio.")
            return "Failed to download or convert YouTube audio."

        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = 5000
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        transcribed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            transcription = await self.transcribe_chunk_async(chunk_path)
            transcribed_chunks.append(transcription)

        os.remove(audio_path)
        return " ".join(transcribed_chunks)


async def transcribe_audio_continuous(transcriber):
    stream = transcriber.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    transcription_tasks = []

    try:
        while True:
            logging.info("Recording chunk...")
            chunk_file = transcriber.record_chunk(stream)
            logging.info("Recorded chunk, dispatching for transcription...")
            task = asyncio.create_task(transcriber.transcribe_chunk_async(chunk_file))
            transcription_tasks.append(task)

            completed, pending = await asyncio.wait(transcription_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0)
            for task in completed:
                logging.info(f"Completed Transcription: {task.result()}")
                transcription_tasks.remove(task)

    except KeyboardInterrupt:
        logging.info("Stopping transcription...")
        stream.stop_stream()
        stream.close()
        transcriber.p.terminate()
        if transcription_tasks:
            await asyncio.wait(transcription_tasks)
    finally:
        stream.stop_stream()
        stream.close()
        transcriber.p.terminate()

async def main():
    transcriber = AudioTranscriber(model_size="base")
    if len(sys.argv) > 1 and sys.argv[1].startswith('http'):
        youtube_url = sys.argv[1]
        transcription = await transcriber.transcribe_youtube_audio(youtube_url)
        print(transcription)
    else:
        await transcribe_audio_continuous(transcriber)

if __name__ == '__main__':
    asyncio.run(main())
