import torch
import torch.nn as nn
import torch.nn.functional as F
from src.audio_features import LogMelSpectrogramExtractor

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + res, inplace=True)
        return out


class TFFaultNet(nn.Module):
    """
    Time-Frequency Attention ResNet for Acoustic Fault Diagnosis.
    End-to-end: Raw audio -> Log-Mel Spectrogram -> 2D ResNet + SE-Attention -> Linear Classifier.
    """
    def __init__(self, num_classes=12, sr=44100, n_mels=64, n_fft=1024, hop_length=256):
        super().__init__()
        self.spec_extractor = LogMelSpectrogramExtractor(
            sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer1 = ConvBlock2D(32, 64, stride=2)
        self.layer2 = ConvBlock2D(64, 128, stride=2)
        self.layer3 = ConvBlock2D(128, 256, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max = nn.AdaptiveMaxPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x is (B, Time_Samples)
        spec = self.spec_extractor(x)  # (B, 1, n_mels, frames)
        feat = self.init_conv(spec)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        
        avg_p = self.global_pool(feat).view(feat.size(0), -1)
        max_p = self.global_max(feat).view(feat.size(0), -1)
        pooled = torch.cat([avg_p, max_p], dim=1)
        
        logits = self.classifier(pooled)
        return logits


class WaveformCNN1D(nn.Module):
    """1D End-to-End Raw Audio Waveform CNN."""
    def __init__(self, num_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 4),
            
            nn.Conv1d(32, 64, kernel_size=16, stride=4, padding=6, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 4),
            
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 4),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x is (B, T) -> add channel dimension (B, 1, T)
        x = x.unsqueeze(1)
        feat = self.features(x).view(x.size(0), -1)
        return self.classifier(feat)


class AudioBiGRU(nn.Module):
    """Bidirectional GRU on Time-Frequency Frames."""
    def __init__(self, num_classes=12, sr=44100, n_mels=64, hidden_dim=128):
        super().__init__()
        self.spec_extractor = LogMelSpectrogramExtractor(sr=sr, n_mels=n_mels)
        self.gru = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x is (B, T) -> spec is (B, 1, n_mels, frames)
        spec = self.spec_extractor(x).squeeze(1) # (B, n_mels, frames)
        # transpose to (B, frames, n_mels)
        seq = spec.transpose(1, 2)
        out, _ = self.gru(seq) # (B, frames, hidden*2)
        # Mean pooling over frames
        pooled = torch.mean(out, dim=1)
        return self.classifier(pooled)
