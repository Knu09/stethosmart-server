from typing import List
import numpy as np
import librosa
import cv2
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt


# -----------------------
# Segment Into Cycles
# -----------------------
def segmentation(
    wav_path: str, sr: int = 16000, window_length: int = 5, hop_length: float = 2.5
) -> List[np.ndarray]:
    """
    Segment a respiratory recording into fixed-length overlapping windows.

    The parameters contains:
        wav_path: path to .wav file
        sr: target sampling rate (default 16kHz)
        window_length: window size in seconds (default 5s)
        hop_length: hop size in seconds (default 2.5s, 50% overlap)

    Returns a list of dictionaries containing:
        List of audio segments (numpy arrays)
    """

    # Load audio and resample
    y, _ = librosa.load(wav_path, sr=sr)

    # Apply bandpass filter
    y = apply_bandpass(y, sr)
    print(f"Applied bandpass to file: {wav_path}")

    # Normalize amplitude safely
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    print(f"Normalize the applitude to file: {wav_path}")

    # Convert seconds to samples
    window_samples = int(window_length * sr)
    hop_samples = int(hop_length * sr)

    segments = []

    # Slide window across signal
    for start in range(0, len(y) - window_samples + 1, hop_samples):
        end = start + window_samples
        segment = y[start:end]

        if len(segment) < sr:
            print(f"Segment cycle is less than the sampling rate: {wav_path}")
            continue

        segments.append(segment.astype(np.float32))

    # If recording is shorter than window_length
    if len(y) < window_samples:
        padded = librosa.util.fix_length(y, size=window_samples)
        segments.append(padded)

    return segments


# -----------------------
# Load Diagnosis Labels
# -----------------------
def load_icbhi_labels(label_file: str):

    print("Loading ICBHI labels split...")
    diag_map = {}
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subj, diag = parts
                diag_map[subj] = diag.lower()

    print("ICBHI labels split completed.\n")
    return diag_map


def extract_features(
    y, sr=16000, n_mels=128, n_mfcc=40, hop_length=512, target_width=216
) -> np.ndarray:
    # y, sr = librosa.load(y, sr=sr)

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(S=mel_spec)

    # # Chroma
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Resize all to (128, 216)
    mel_resized = cv2.resize(mel_spec_db, (target_width, n_mels))
    # chroma_resized = cv2.resize(chroma, (target_width, n_mels))
    mfcc_resize = cv2.resize(mfcc, (target_width, n_mels))

    # print(f"Mel Spec DB Shape: {mel_resized.shape}")
    # print(f"Chroma Shape: {chroma_resized.shape}")
    # print(f"MFCC Shape: {mfcc_resize.shape}")

    # Resize or pad all features
    # (ensure same shape for all e.g. (3,128,216))
    stacked = np.stack([mel_resized, mfcc_resize], axis=0).astype(np.float32)

    # print(f"Final Shape: {stacked.shape}")
    # print(f"Feature Stacked: {stacked}")
    return stacked


def apply_bandpass(
    y: NDArray[np.float64],
    sr: int,
    lowcut: float = 100.0,
    highcut: float = 2000.0,
    order: int = 4,
) -> NDArray[np.float64]:
    """
    Stable bandpass filter using second-order sections.
    """

    nyquist = 0.5 * sr

    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1.0:
        high = 0.999

    if low <= 0.0:
        low = 0.001

    sos = butter(order, [low, high], btype="band", output="sos")

    filtered = sosfiltfilt(sos, y)

    return filtered.astype(np.float64)
