# StethoSmart Flask Server

StethoSmart Server is a Flask-based API service that processes lung sound recordings from an embedded system using ESP32-S3 through a stethoscope and performs AI-based respiratory disease prediction.

The server receives Pulse Code Modulation (PCM), converts to a WAV audio recording, preprocesses them, runs inference using a trained deep learning model, and returns a prediction with a confidence score.

## Features

* Accepts Pulse Code Modulation (PCM)
* Performs lung sound segmentation
* Runs deep learning inference
* Returns prediction and confidence score
* Supports ESC/POS printing for diagnostic output
* Designed for integration with StethoSmart hardware devices

## Requirements

* Python 3.10+
* pip
* FFmpeg (optional if future audio conversion is added)

### Python libraries:

* Flask
* PyTorch
* librosa
* numpy
* soundfile
* pydub (optional)

## Installation

1. Clone the Repository
```bash
git clone https://github.com/Knu09/stethosmart-server.git
cd stethosmart-server
```

2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Server

Start the Flask API server:
``` bash
python server.py
```

The server will run on:

* http://localhost:5000


## API Endpoint
### POST /predict_pcm

Uploads a WAV lung sound recording and returns a prediction.

### Request

Multipart form upload:

* file: lung_sound.wav

Example using curl:
```bash
curl -X POST http://localhost:5000/predict_pcm \
  --data-binary @pcm_data.pcm
```

## Audio Requirements

The server expects audio in the following format:

* File type: WAV
* Sample rate: 16 kHz
* Channels: Mono
* Bit depth: 16-bit PCM

## License

This project is part of the StethoSmart research and development system.

## Authors
16kHz Labs
* [https://github.com/Knu09]
* [https://github.com/rafaeljacov/rafaeljacov]
