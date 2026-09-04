import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F

from src.dataset import load_wav
from src.audio_features import LogMelSpectrogramExtractor
from src.models import TFFaultNet

class GradCAM:
    """
    Grad-CAM on 2D Spectrogram representations for acoustic fault diagnosis interpretability.
    Extracts gradients from layer3 feature maps and backprojects onto time-frequency bins.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate(self, audio_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        # audio_tensor: (1, 44100)
        logits = self.model(audio_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        score = logits[0, target_class]
        score.backward()
        
        # Gradients: (1, C, H, W), Activations: (1, C, H, W)
        grads = self.gradients[0] # (C, H, W)
        acts = self.activations[0] # (C, H, W)
        
        # Global average pool gradients to get importance weight per channel alpha_k
        weights = torch.mean(grads, dim=(1, 2)) # (C,)
        
        # Weighted sum of feature maps
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        cam = F.relu(cam) # Only features that positively contribute to class
        cam = cam.cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-9)
        else:
            cam = np.zeros_like(cam)
            
        return cam, logits[0].detach().cpu().numpy(), target_class


def analyze_se_attention(model, audio_tensor):
    """
    Extracts the channel gating weights w_c = sigmoid(W2 * relu(W1 * avg_pool))
    from the Squeeze-and-Excitation blocks in layer1, layer2, and layer3.
    """
    model.eval()
    se_weights = {}
    
    def get_se_hook(name):
        def hook(module, input, output):
            # output is x * w. Let's calculate w directly from module.fc
            x = input[0]
            b, c, _, _ = x.shape
            w = module.fc(x).view(c).detach().cpu().numpy()
            se_weights[name] = w
        return hook
    
    h1 = model.layer1.se.register_forward_hook(get_se_hook("layer1"))
    h2 = model.layer2.se.register_forward_hook(get_se_hook("layer2"))
    h3 = model.layer3.se.register_forward_hook(get_se_hook("layer3"))
    
    with torch.no_grad():
        _ = model(audio_tensor)
        
    h1.remove()
    h2.remove()
    h3.remove()
    
    return se_weights


def main():
    print("=" * 75)
    print("RUNNING EXPLAINABLE AI (XAI) & SPECTROGRAM ATTENTION SALIENCY ANALYSIS")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "results/checkpoints/best_tf_faultnet.pt"
    
    model = TFFaultNet(num_classes=12).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded trained checkpoint from {checkpoint_path}")
    else:
        print(f"Error: {checkpoint_path} not found.")
        return

    # Extract spectrogram extractor for visualization
    spec_extractor = LogMelSpectrogramExtractor(sr=44100, n_mels=64).to(device)
    
    # Class dictionary
    class_names = sorted([d for d in os.listdir("Base_de_Dados") if os.path.isdir(os.path.join("Base_de_Dados", d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    # Select representative samples:
    # 1. Normal (Baseline)
    # 2. Tooth Loss (Perda_concentrada - Impulsive Shock Transients)
    # 3. Belt Slip (Escorregamento - Continuous Friction / Shearing)
    samples_to_analyze = [
        ("Normal", "Normal", "Baseline / Healthy Rig"),
        ("Perda_concentrada", "Perda_concentrada", "Severe Tooth Loss (Impulsive Impact Shocks)"),
        ("Escorregamento", "Escorregamento", "Belt Slip (Continuous High-Frequency Friction)")
    ]
    
    gradcam = GradCAM(model, target_layer=model.layer3.conv2)
    
    # Setup Figure with GridSpec
    fig = plt.figure(figsize=(18, 11), dpi=300)
    gs = gridspec.GridSpec(3, 3, width_ratios=[1.2, 1.2, 0.8], height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)
    
    time_axis = np.linspace(0, 1.0, 173) # 173 time frames
    mel_freqs = np.linspace(20, 22050, 64) # 64 mel bins
    
    print("\nComputing Grad-CAM heatmaps and SE Attention activations...")
    
    for row_idx, (folder_name, label_name, desc_title) in enumerate(samples_to_analyze):
        files = sorted(glob.glob(os.path.join("Base_de_Dados", folder_name, "*.wav")))
        if not files:
            continue
        # Pick middle file for consistency
        sample_path = files[len(files) // 2]
        audio_np, sr = load_wav(sample_path)
        if len(audio_np) > 44100:
            audio_np = audio_np[:44100]
        elif len(audio_np) < 44100:
            pad = np.zeros(44100, dtype=np.float32)
            pad[:len(audio_np)] = audio_np
            audio_np = pad
            
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(device)
        
        # 1. Log-Mel Spectrogram
        with torch.no_grad():
            spec = spec_extractor(audio_tensor)[0, 0].cpu().numpy() # (64, 173)
            
        # 2. Grad-CAM
        target_c = class_to_idx[label_name]
        cam, logits, pred_c = gradcam.generate(audio_tensor, target_class=target_c)
        
        # Resize CAM to match spectrogram size (64, 173)
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(64, 173), mode='bilinear', align_corners=False)[0, 0].numpy()
        
        # 3. SE Attention
        se_w = analyze_se_attention(model, audio_tensor)
        layer3_w = se_w["layer3"]
        
        # Plot Column 1: Raw Log-Mel Spectrogram
        ax_spec = fig.add_subplot(gs[row_idx, 0])
        im1 = ax_spec.imshow(spec, aspect='auto', origin='lower', cmap='magma',
                             extent=[0, 1.0, 0, 64])
        ax_spec.set_title(f"Log-Mel Spectrogram: {label_name}\n({desc_title})", fontsize=11, fontweight='bold')
        ax_spec.set_ylabel("Mel Frequency Bins", fontsize=10, fontweight='bold')
        if row_idx == 2:
            ax_spec.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
            
        # Plot Column 2: Grad-CAM Saliency Overlay
        ax_cam = fig.add_subplot(gs[row_idx, 1])
        # Background: grayscale spectrogram
        ax_cam.imshow(spec, aspect='auto', origin='lower', cmap='gray', extent=[0, 1.0, 0, 64])
        # Foreground: jet CAM with alpha
        im2 = ax_cam.imshow(cam_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.55,
                            extent=[0, 1.0, 0, 64], vmin=0.0, vmax=1.0)
        pred_label_str = class_names[pred_c]
        conf = np.exp(logits[pred_c]) / np.sum(np.exp(logits)) * 100
        ax_cam.set_title(f"Grad-CAM Attribution (Layer 3)\nPred: {pred_label_str} ({conf:.1f}% conf)", fontsize=11, fontweight='bold')
        if row_idx == 2:
            ax_cam.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
            
        # Plot Column 3: Squeeze-and-Excitation Channel Attention Distribution
        ax_se = fig.add_subplot(gs[row_idx, 2])
        channel_indices = np.arange(len(layer3_w))
        # Highlight top 10% channels
        top_threshold = np.percentile(layer3_w, 90)
        colors = ['#d62728' if w >= top_threshold else '#1f77b4' for w in layer3_w]
        ax_se.bar(channel_indices, layer3_w, color=colors, width=1.0, alpha=0.85)
        ax_se.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_se.set_ylim(0, 1.05)
        ax_se.set_ylabel("SE Channel Weight ($w_c$)", fontsize=9.5, fontweight='bold')
        ax_se.set_title(f"SE Attention Weights (256 Ch)\nMax: {np.max(layer3_w):.2f}, Mean: {np.mean(layer3_w):.2f}", fontsize=10.5, fontweight='bold')
        if row_idx == 2:
            ax_se.set_xlabel("Feature Channel Index", fontsize=10, fontweight='bold')

    plt.suptitle("Explainable AI (XAI) Spectrogram Saliency & SE Attention Recalibration in TF-FaultNet",
                 fontsize=15, fontweight='bold', y=0.99)
    
    os.makedirs("results/visualizations", exist_ok=True)
    out_path = "results/visualizations/xai_attention_gradcam.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"\n>>> Saliency and explainability figure successfully saved to: {out_path}")


if __name__ == "__main__":
    main()
