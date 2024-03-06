# Real-time Audio Transcription with Faster-Whisper

## Introduction
This project provides tools for real-time audio transcription using OpenAI's Whisper model. It includes scripts for both live audio capture and transcription from microphone input, as well as for transcribing pre-recorded audio files.

## Prerequisites
- Python 3.6 or higher

## Setup

### Virtual Environment (Recommended)
1. Create a virtual environment: `python3 -m venv venv`
2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

### Dependencies
Install the required Python packages:
 - pip install -r requirements.txt

 
## Usage

### Real-time Transcription
To transcribe the audio in real-time from your microphone:
- The model `large-v3` is used to transcribe audio. Alternatively we can use the `module-en` model.
- python real_time.py


Press `Ctrl+C` to stop recording and save the transcription to `transcription_log.txt`.


### Youtube Video real time transcription


### File Transcription
To transcribe a pre-recorded audio file:
1. Edit `file.py`, replacing `"data/english.mp3"` or  `"data/spanish.mp3"` with your audio file's path.
2. Run:
  - python file.py


## Files
- `real_time.py`: Captures and transcribes audio from microphone in real-time.
- `file.py`: Transcribes a specified audio file.
- `requirements.txt`: Lists all necessary Python packages.

## Features
- Supports CUDA for GPU acceleration.
- Automatic language detection.
- Efficient on-device processing.

## Note
Performance may vary based on system capabilities and GPU availability. Adjust the audio source or transcription settings as needed.

## Contributing
Contributions are welcome. Please submit pull requests or issues to enhance functionality or fix bugs.

## License
This project is licensed under the MIT License.

---
