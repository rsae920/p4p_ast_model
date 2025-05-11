import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Data Partitioning
data_dir = "/Users/ryansaena/Documents/Part IV/RESEARCH PROJECT/P4P/p4p_ast_model/data"
files = os.listdir(data_dir)

# Spectrogram Initialisation
spectrogram_dir = "spectrograms"
os.makedirs(spectrogram_dir, exist_ok=True)

emotion_mapping = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "07": "disgust"
}

desired_emotions = {"01", "03", "04", "05", "07"}

# Populate the data list
data = []
for file in files:
    if file.endswith(".wav"):
        parts = file.split("-")
        if len(parts) > 2: 
            emotion_code = parts[2]
            if emotion_code in desired_emotions:
                label = emotion_mapping[emotion_code]
                filepath = os.path.join(data_dir, file)
                data.append((filepath, label))
                print(f"Added: {filepath}, Label: {label}")

# Spectrogram Generation Function
def save_spectrogram(filepath, save_dir, sr=22050, n_fft=2048, hop_length=512, win_length=None, n_mels=128, fmax=None):
    try:
        # Load audio file
        y, _ = librosa.load(filepath, sr=sr)
        
        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, fmax=fmax)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel_spec, sr=sr, hop_length=hop_length, cmap="grey")
        
        save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(filepath))[0] + ".png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()  # Close the plot to avoid overlapping
        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

for filepath, label in data:
    # Create label subdirectory
    label_dir = os.path.join(spectrogram_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # Generate and save spectrogram as image
    save_spectrogram(filepath, label_dir)


