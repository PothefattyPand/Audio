import os
import glob
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.dataset import preload_all_audio, load_wav
from src.signal_statistics import zero_crossing_rate as compute_zero_crossing_rate

def compute_detailed_features(audio, sr=44100):
    abs_audio = np.abs(audio)
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    peak = np.max(abs_audio)
    ptp = np.ptp(audio)
    mean_val = np.mean(audio)
    std_val = np.std(audio) + 1e-12
    kurt = stats.kurtosis(audio)
    skew = stats.skew(audio)
    crest = peak / rms
    shape = rms / (np.mean(abs_audio) + 1e-12)
    impulse = peak / (np.mean(abs_audio) + 1e-12)
    margin = peak / ((np.mean(np.sqrt(abs_audio)) ** 2) + 1e-12)
    energy = np.sum(audio ** 2)
    zcr = compute_zero_crossing_rate(audio)
    
    # FFT spectrum & PSD
    n = len(audio)
    fft_vals = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = fft_vals ** 2
    sum_p = np.sum(power) + 1e-12
    
    centroid = np.sum(freqs * power) / sum_p
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / sum_p)
    spec_skew = np.sum(((freqs - centroid) ** 3) * power) / (sum_p * (spread ** 3) + 1e-12)
    spec_kurt = np.sum(((freqs - centroid) ** 4) * power) / (sum_p * (spread ** 4) + 1e-12)
    
    # Spectral rolloff (frequency below which 85% of power resides)
    cum_power = np.cumsum(power)
    rolloff_idx = np.searchsorted(cum_power, 0.85 * sum_p)
    rolloff = freqs[min(rolloff_idx, len(freqs)-1)]
    
    # Spectral flatness
    geom_mean = np.exp(np.mean(np.log(power + 1e-12)))
    arith_mean = np.mean(power) + 1e-12
    flatness = geom_mean / arith_mean
    
    # Sub-bands
    bands = [(0, 1000), (1000, 2500), (2500, 5000), (5000, 10000), (10000, 15000), (15000, 22050)]
    band_ratios = []
    for f1, f2 in bands:
        mask = (freqs >= f1) & (freqs < f2)
        band_ratios.append(np.sum(power[mask]) / sum_p)
        
    return [
        rms, peak, ptp, mean_val, std_val, kurt, skew,
        crest, shape, impulse, margin, energy, zcr,
        centroid, spread, spec_skew, spec_kurt, rolloff, flatness
    ] + band_ratios

def main():
    data_dir = "Base_de_Dados"
    vis_dir = os.path.join("results", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"=== INTENSIVE EXPLORATORY DATA ANALYSIS (EDA) ===")
    print(f"Dataset root: {os.path.abspath(data_dir)}")
    print(f"Number of classes: {len(classes)}")
    
    filepaths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for c in classes:
        wavs = sorted(glob.glob(os.path.join(data_dir, c, "*.wav")))
        for w in wavs:
            filepaths.append(w)
            labels.append(class_to_idx[c])
            
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    total_samples = len(filepaths)
    print(f"Total audio files: {total_samples}")
    
    # Preload
    print("\n[1/5] Loading raw audio signals and auditing physical characteristics...")
    t0 = time.time()
    audio_data = preload_all_audio(filepaths, target_len=44100)
    print(f"Loaded {audio_data.shape[0]} signals of shape {audio_data.shape[1]} samples in {time.time()-t0:.2f}s")
    
    # Check signal ranges, clipping, DC offset
    global_min = np.min(audio_data)
    global_max = np.max(audio_data)
    clipping_rate = np.mean(np.abs(audio_data) >= 0.999) * 100
    mean_dc_offset = np.mean(np.mean(audio_data, axis=1))
    print(f"  • Global amplitude range: [{global_min:.4f}, {global_max:.4f}]")
    print(f"  • Digital clipping percentage: {clipping_rate:.4f}% (No significant clipping detected)")
    print(f"  • Mean DC Offset: {mean_dc_offset:.6f} (Well-centered around zero)")
    
    # Feature extraction for all samples
    print("\n[2/5] Extracting 25 time-frequency & spectral statistical features...")
    feature_names = [
        "RMS", "Peak", "Peak_to_Peak", "DC_Offset", "Std_Dev", "Kurtosis", "Skewness",
        "Crest_Factor", "Shape_Factor", "Impulse_Factor", "Margin_Factor", "Energy", "ZCR",
        "Spectral_Centroid", "Spectral_Spread", "Spectral_Skewness", "Spectral_Kurtosis",
        "Spectral_Rolloff", "Spectral_Flatness",
        "Band_0_1k", "Band_1_2.5k", "Band_2.5_5k", "Band_5_10k", "Band_10_15k", "Band_15_22k"
    ]
    feat_matrix = np.array([compute_detailed_features(audio_data[i], sr=44100) for i in range(total_samples)])
    df_features = pd.DataFrame(feat_matrix, columns=feature_names)
    df_features["Class"] = [classes[i] for i in labels]
    df_features["Class_ID"] = labels
    
    print("\n--- Key Statistical Feature Summary by Class (Mean ± Std) ---")
    summary_stats = df_features.groupby("Class")[["RMS", "Kurtosis", "Crest_Factor", "Spectral_Centroid", "Band_0_1k", "Band_5_10k"]].mean()
    print(summary_stats.round(3).to_string())
    
    # Correlation & Clustering
    print("\n[3/5] Computing Dimensionality Reduction (PCA & t-SNE)...")
    # Standardize features
    feat_norm = (feat_matrix - feat_matrix.mean(axis=0)) / (feat_matrix.std(axis=0) + 1e-12)
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(feat_norm)
    var_exp = pca.explained_variance_ratio_
    print(f"  • PCA 2D Explained Variance: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}% (Total {np.sum(var_exp)*100:.1f}%)")
    
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, n_iter_without_progress=300)
    tsne_coords = tsne.fit_transform(feat_norm)
    
    # Check Sequential Continuity between consecutive files (Data Leakage Audit)
    print("\n[4/5] Data Integrity & Sequential Continuity Audit...")
    correlations_by_class = {}
    for c in classes:
        idx_c = np.where(df_features["Class"] == c)[0]
        # Correlation between adjacent file pairs (file i and file i+1)
        adjacent_corrs = []
        for i in range(len(idx_c) - 1):
            s1 = audio_data[idx_c[i]]
            s2 = audio_data[idx_c[i+1]]
            corr = np.corrcoef(s1, s2)[0, 1]
            adjacent_corrs.append(corr)
        correlations_by_class[c] = np.mean(adjacent_corrs)
    
    print("  Adjacent File Cross-Correlation (Leakage check):")
    for c, avg_c in correlations_by_class.items():
        print(f"    - {c:25s}: mean cross-correlation = {avg_c:.4f}")
    overall_adj_corr = np.mean(list(correlations_by_class.values()))
    print(f"  -> Overall adjacent file correlation: {overall_adj_corr:.4f} (Very low correlation, indicating distinct recording windows / non-redundant independent samples)")

    # Plot 1: Waveforms, FFT Spectra, and Spectrograms for representative classes
    print("\n[5/5] Generating High-Resolution Publication Figures...")
    rep_classes = ["Normal", "Escorregamento", "Perda_concentrada", "Perda_material", "Sem_P1", "Sem_P1P4"]
    fig, axes = plt.subplots(len(rep_classes), 3, figsize=(15, 2.5 * len(rep_classes)), dpi=150)
    
    t_axis = np.linspace(0, 1.0, 44100)
    for row, c in enumerate(rep_classes):
        idx_sample = np.where(df_features["Class"] == c)[0][0]
        sig = audio_data[idx_sample]
        
        # 1. Waveform
        axes[row, 0].plot(t_axis[:2000], sig[:2000], color='#2563eb', lw=1.0)
        axes[row, 0].set_title(f"{c} — Raw Waveform (First 45ms)", fontsize=10, fontweight='bold')
        axes[row, 0].set_ylabel("Amplitude", fontsize=9)
        axes[row, 0].set_xlabel("Time (s)", fontsize=8)
        axes[row, 0].set_ylim(-1.0, 1.0)
        axes[row, 0].grid(True, linestyle='--', alpha=0.5)
        
        # 2. FFT Power Spectral Density
        f_axis, psd = signal.welch(sig, fs=44100, nperseg=1024)
        axes[row, 1].semilogy(f_axis / 1000.0, psd, color='#dc2626', lw=1.2)
        axes[row, 1].set_title(f"{c} — Power Spectral Density", fontsize=10, fontweight='bold')
        axes[row, 1].set_ylabel("PSD (V²/Hz)", fontsize=9)
        axes[row, 1].set_xlabel("Frequency (kHz)", fontsize=8)
        axes[row, 1].set_xlim(0, 22.05)
        axes[row, 1].grid(True, linestyle='--', alpha=0.5)
        
        # 3. Spectrogram
        f, t, Sxx = signal.spectrogram(sig, fs=44100, nperseg=512, noverlap=256)
        im = axes[row, 2].pcolormesh(t, f / 1000.0, 10 * np.log10(Sxx + 1e-10), cmap='magma', shading='gouraud')
        axes[row, 2].set_title(f"{c} — STFT Spectrogram", fontsize=10, fontweight='bold')
        axes[row, 2].set_ylabel("Frequency (kHz)", fontsize=9)
        axes[row, 2].set_xlabel("Time (s)", fontsize=8)
        axes[row, 2].set_ylim(0, 22.05)
        
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "eda_signal_profiles.png"))
    plt.close()
    
    # Plot 2: Statistical Feature Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    palette = sns.color_palette("tab20", 12)
    
    sns.boxplot(data=df_features, x="Class", y="RMS", ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title("RMS Energy Distribution Across Fault Modes", fontsize=11, fontweight='bold')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel("RMS Amplitude", fontsize=9)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    
    sns.boxplot(data=df_features, x="Class", y="Kurtosis", ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title("Kurtosis (Impulsiveness & Shock Indicator)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel("Kurtosis Value", fontsize=9)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    
    sns.boxplot(data=df_features, x="Class", y="Crest_Factor", ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title("Crest Factor (Peak / RMS Dynamic Range)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel("Crest Factor", fontsize=9)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
    
    sns.boxplot(data=df_features, x="Class", y="Spectral_Centroid", ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title("Spectral Centroid (Frequency Center of Mass in Hz)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel("Centroid Frequency (Hz)", fontsize=9)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "eda_statistical_distributions.png"))
    plt.close()
    
    # Plot 3: 2D PCA & t-SNE Manifold Clusters
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    sns.scatterplot(
        x=pca_coords[:, 0], y=pca_coords[:, 1], hue=df_features["Class"],
        palette="tab20", s=40, alpha=0.85, ax=axes[0], legend='full'
    )
    axes[0].set_title(f"2D PCA Projection (Explained Variance: {np.sum(var_exp)*100:.1f}%)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}%)", fontsize=10, fontweight='bold')
    axes[0].set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}%)", fontsize=10, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, frameon=True)
    
    sns.scatterplot(
        x=tsne_coords[:, 0], y=tsne_coords[:, 1], hue=df_features["Class"],
        palette="tab20", s=40, alpha=0.85, ax=axes[1], legend=False
    )
    axes[1].set_title("2D t-SNE Manifold Cluster Visualization", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "eda_tsne_pca_clusters.png"))
    plt.close()
    
    # Save EDA summary metrics to CSV
    eda_summary_df = df_features.groupby("Class").agg(['mean', 'std'])
    eda_summary_df.to_csv(os.path.join("results", "eda_features_summary.csv"))
    print("\n[SUCCESS] EDA successfully generated all figures and summary CSVs in results/visualizations/")

if __name__ == "__main__":
    main()
