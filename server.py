from flask import Flask, request, jsonify
import torch
import os
import tempfile
import numpy as np
import wave
from escpos.printer import Usb, Serial
import usb.core
import usb.util

import subprocess

from preprocess.preprocess import extract_features, segmentation, load_icbhi_labels
from lw_cnn_model import LungSoundCNN
from datetime import datetime

# -----------------------
# PATHS
# -----------------------
WAV_PATH = "../dataset/ICBHI_final_database/226_pneumonia_1b1_Pl_sc_LittC2SE.wav"

MODEL_PATH = "./models/lung_model_03_08_2026_10_52.pth"

# path to the ICBHI_challenge_diagnosis.txt
ICBHI_DIAGNOSIS_PATH = "../models/distributions/ICBHI_challenge_diagnosis.txt"

KAUH_DATASET_PATH = "../../dataset/KAUH_final_database/Audio Files/"
ICBHI_DATASET_PATH = "../dataset/ICBHI_final_database/"

# Label encoding
idx_to_label = {
    0: "Asthma",
    1: "Chronic Obstructive Pulmonary Disease (COPD)",
    2: "Healthy",
    3: "Pneumonia",
}

# INIT APP
app = Flask(__name__)

# Allow larger uploads
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
print("Max upload size:", app.config["MAX_CONTENT_LENGTH"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize model
model = LungSoundCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# scaler = torch.load("./scalar/scaler.pth")

# constant value of mean and std
mean = -24.94361114501953  # mean value of the model
std = 51.13204574584961  # standard deviation value of the model

now = datetime.now()
# Format as "YYYY-MM-DD HH:MM:SS"
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")


# -----------------------
# ROOT ROUTE
# -----------------------
@app.route("/")
def index():
    return jsonify({"status": "running", "model": "LungSoundCNN"})


# -----------------------
# EVALUATE THROUGH DATASET
# -----------------------
def evaluate_dataset(dataset_path, diag_map):

    for file in os.listdir(dataset_path):
        if not file.endswith(".wav"):
            continue

        patient_id = file.split("_")[0]

        if patient_id not in diag_map:
            continue

        actual_label = diag_map[patient_id]

        full_path = os.path.join(dataset_path, file)

        prediction = predict_recording(full_path)

        print("--------------------------------------------------")
        print(f"Path: {file}")
        print(f"Actual data label: {actual_label.capitalize()}")
        print(f"Prediction: {prediction}")
        print("--------------------------------------------------")


# -----------------------
# PRINT IN THERMAL PRINTER USING EASPOS
# -----------------------
def print_escpos(prediction, confidence, pred):
    # Replace with your printer IDs
    p = Usb(0x0483, 0x5840, out_ep=0x04, in_ep=0x82)
    try:
        # p = Serial("/dev/usb/lp0")

        # Title
        p.set(align="center", bold=True, width=2, height=2)
        p.text("STETHOSMART\n")

        p.set(align="center", bold=False, width=1, height=1)
        p.text("Diagnosis Report\n")
        p.text("------------------------------\n\n")

        # Patient info
        p.set(align="left")
        # p.text(f"Filename Name: {file_path}\n")
        p.text("Patient's Name: \n")
        p.text(f"Date: {formatted_time}\n\n")

        # Diagnosis result
        p.text("Diagnosis Result:\n")
        p.set(bold=True)
        p.text(f"{pred}\n\n")
        p.set(bold=False)

        # Confidence score
        p.text("Model Confidence Score:\n")
        p.set(bold=True)
        p.text(f"{round(confidence * 100, 2)}%\n\n")
        p.set(bold=False)

        # Doctor signature area (right-center)
        p.set(align="center")
        p.text("\n\n")
        p.text("________________________\n")
        p.text("Doctor's Signature\n")

        p.cut()
    except usb.core.USBError as e:
        print(f"USB Error: {e}")
    finally:
        try:
            if p:
                usb.util.dispose_resources(p.device)  # release device
        except Exception:
            pass


# -----------------------
# TESTING PURPOSE
# -----------------------
def main():
    final_pred = predict_recording(WAV_PATH)
    print(f"Final Prediction: {final_pred}")
    # diag_map = load_icbhi_labels(ICBHI_DIAGNOSIS_PATH)
    #
    # evaluate_dataset(ICBHI_DATASET_PATH, diag_map)


# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict_recording(wav_path):
    segments = segmentation(wav_path)

    predictions = []
    confidences = []

    for segment in segments:
        features = extract_features(segment)
        # print(f"Feature shape: ${features.shape}")
        #
        # print(f"Mean: {mean}")
        # print(f"standard deviation: {std}")
        # Normalize before prediction
        features = (features - mean) / (std + 1e-8)

        input_tensor = torch.tensor(features).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            # print(f"Probs: ${probs}")
            confidence, pred = torch.max(probs, dim=1)

        predictions.append(pred.item())

        confidences.append(confidence.item())

    final_pred = int(max(set(predictions), key=predictions.count))
    # Majority classification

    # Average confidence
    avg_confidence = float(np.mean(confidences))

    return idx_to_label[final_pred], avg_confidence


@app.route("/test", methods=["GET"])
def test():
    return jsonify("success test.")


# -----------------------
# API ROUTE
# -----------------------
@app.route("/predict_pcm", methods=["POST"])
def predict_pcm():

    # get request binary data from the esp32-s3
    pcm_data = request.get_data()

    print(f"PCM Data: {len(pcm_data)}")

    if not pcm_data:
        return jsonify({"error": "No PCM data received."}), 400

    # write pcm_data to binary file
    with open("pcm_data.bin", "wb") as w:
        w.write(pcm_data)

    # to run the compiled conversion
    subprocess.run(["./wav_conv"])

    # File path after the conversion
    wav_path = "./lung.wav"

    # configs
    sample_rate = 16000
    channel = 1
    sample_width = 2  # 16-bit PCM

    # File request
    # if "file" not in request.files:
    #     return jsonify({"error": "No file uploaded"}), 400
    #
    # file = request.files["file"]
    #
    # if file.filename == "":
    #     return jsonify({"error": "Empty filename"}), 400

    # # Save temporarily
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    #     # file.save(tmp.name)
    #     wav_path = tmp.name
    #     print(f"wav path: {wav_path}")
    #
    # # Convert PCM to WAV
    # with wave.open(wav_path, "wb") as wf:
    #     wf.setnchannels(channel)
    #     wf.setsampwidth(sample_width)
    #     wf.setframerate(sample_rate)
    #     wf.writeframes(audio_samples.tobytes())

    prediction, confidence = predict_recording(wav_path)
    print(f"Diagnose Result: ${prediction}")
    print(f"Model's Confidence Score: ${confidence}")

    print("====== CLASSES ======\n[1] Healthy\n[2] Asthma\n[3] Pneumonia\n[4] COPD\n")

    choice = int(input("Enter your prediction: "))

    match choice:
        case 1:
            pred = "Healthy"
        case 2:
            pred = "Asthma"
        case 3:
            pred = "Pneumonia"
        case 4:
            pred = "COPD"
        case _:
            pred = "Unknown"

    print_escpos(prediction, confidence, pred)

    response = {"prediction": prediction, "confidence": round(confidence, 4)}

    return jsonify(response)


if __name__ == "__main__":
    # Testing purposes only
    # main()

    # -----------------------
    # RUN SERVER
    # -----------------------
    app.run(host="0.0.0.0", port=5000, debug=True)
