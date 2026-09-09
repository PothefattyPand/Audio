import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from src.signal_statistics import zero_crossing_rate as compute_zero_crossing_rate

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_mel_filterbank(sr=44100, n_fft=1024, n_mels=64, f_min=20.0, f_max=22050.0):
    """Create a Mel filterbank matrix (n_mels, n_fft // 2 + 1) without external audio libraries."""
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max if f_max is not None else sr / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    
    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        
        for f in range(left, center):
            if f < n_freqs and center > left:
                weights[i - 1, f] = (f - left) / (center - left)
        for f in range(center, right):
            if f < n_freqs and right > center:
                weights[i - 1, f] = (right - f) / (right - center)
                
    # Normalize filter energy
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]
    return torch.from_numpy(weights).float()


class LogMelSpectrogramExtractor(nn.Module):
    """GPU-accelerated STFT and Log-Mel Spectrogram Extractor."""
    def __init__(self, sr=44100, n_fft=1024, hop_length=256, n_mels=64, f_min=20.0, f_max=22050.0):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.register_buffer("window", torch.hann_window(n_fft))
        mel_fb = get_mel_filterbank(sr, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_filterbank", mel_fb)
        
    def forward(self, x):
        """
        x: (batch_size, time_samples)
        Returns: (batch_size, 1, n_mels, time_steps)
        """
        # x is (B, T)
        # stft -> (B, Freq, Frames) complex
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        # Power spectrogram: |STFT|^2
        power = torch.abs(stft) ** 2 # (B, n_fft//2 + 1, Frames)
        # Mel projection: (n_mels, Freq) @ (B, Freq, Frames) -> (B, n_mels, Frames)
        mel_spec = torch.matmul(self.mel_filterbank, power)
        # Log scale (dB)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-6))
        # Instance normalization (zero mean, unit variance per spectrogram)
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        norm_mel = (log_mel - mean) / std
        # Add channel dimension: (B, 1, n_mels, Frames)
        return norm_mel.unsqueeze(1)


def extract_signal_statistical_features(audio, sr=44100):
    """
    Extract comprehensive time-domain and frequency-domain statistical features for ML models (XGBoost/LightGBM).
    """
    # 1. Time-domain statistical indicators
    abs_audio = np.abs(audio)
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    peak = np.max(abs_audio)
    peak_to_peak = np.ptp(audio)
    mean_val = np.mean(audio)
    std_val = np.std(audio) + 1e-10
    variance = np.var(audio)
    skewness = stats.skew(audio)
    kurt = stats.kurtosis(audio)
    crest_factor = peak / rms
    shape_factor = rms / (np.mean(abs_audio) + 1e-10)
    impulse_factor = peak / (np.mean(abs_audio) + 1e-10)
    margin_factor = peak / ((np.mean(np.sqrt(abs_audio)) ** 2) + 1e-10)
    energy = np.sum(audio ** 2)
    zero_crossing_rate = compute_zero_crossing_rate(audio)
    
    # 2. Frequency-domain statistical indicators (FFT spectrum)
    fft_vals = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    fft_power = fft_vals ** 2
    sum_power = np.sum(fft_power) + 1e-10
    
    spectral_centroid = np.sum(freqs * fft_power) / sum_power
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_power) / sum_power)
    spectral_skewness = np.sum(((freqs - spectral_centroid) ** 3) * fft_power) / (sum_power * (spectral_spread ** 3) + 1e-10)
    spectral_kurtosis = np.sum(((freqs - spectral_centroid) ** 4) * fft_power) / (sum_power * (spectral_spread ** 4) + 1e-10)
    
    # Energy in frequency sub-bands (e.g. 0-1k, 1k-2k, 2k-5k, 5k-10k, 10k-20k)
    bands = [(0, 1000), (1000, 2500), (2500, 5000), (5000, 10000), (10000, 15000), (15000, 22050)]
    band_energies = []
    for f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs < f_high)
        band_energies.append(np.sum(fft_power[mask]) / sum_power)
        
    features = [
        rms, peak, peak_to_peak, mean_val, std_val, variance,
        skewness, kurt, crest_factor, shape_factor, impulse_factor,
        margin_factor, energy, zero_crossing_rate,
        spectral_centroid, spectral_spread, spectral_skewness, spectral_kurtosis
    ] + band_energies
    
    return np.array(features, dtype=np.float32)
