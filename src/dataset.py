import os
import wave
import numpy as np
import torch
from torch.utils.data import Dataset

def load_wav(filepath):
    """Load mono 16-bit PCM WAV file into float32 numpy array normalized to [-1.0, 1.0]."""
    with wave.open(filepath, 'rb') as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        frames = w.readframes(n_frames)
        
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    elif sampwidth == 1:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
        
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if sampwidth == 2:
        audio /= 32768.0
    elif sampwidth == 4:
        audio /= 2147483648.0
    elif sampwidth == 1:
        audio = (audio - 128.0) / 128.0
        
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
        
    return audio, framerate

def preload_all_audio(filepaths, target_len=44100):
    """Preload all audio into a contiguous float32 numpy array (N, target_len)."""
    n_samples = len(filepaths)
    audio_matrix = np.zeros((n_samples, target_len), dtype=np.float32)
    for i, fp in enumerate(filepaths):
        aud, _ = load_wav(fp)
        if len(aud) < target_len:
            audio_matrix[i, :len(aud)] = aud
        else:
            audio_matrix[i] = aud[:target_len]
    return audio_matrix

class MemoryFaultDataset(Dataset):
    def __init__(self, audio_array, labels, augment=False):
        self.audio_array = audio_array
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.audio_array[idx]
        label = self.labels[idx]
        
        if self.augment:
            audio = audio.copy()
            # 1. Random noise
            if np.random.rand() < 0.35:
                audio = audio + np.random.normal(0, 0.003, size=audio.shape).astype(np.float32)
            # 2. Random gain
            if np.random.rand() < 0.35:
                gain = float(np.random.uniform(0.85, 1.15))
                audio = audio * gain
                
        return torch.from_numpy(audio).float(), torch.tensor(label, dtype=torch.long)
