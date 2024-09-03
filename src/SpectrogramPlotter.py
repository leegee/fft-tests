import os
import io
import sqlite3
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

class SpectrogramPlotter:
    @staticmethod
    def plot_mel_spectrogram(mel_spectrogram, filename):
        """Plot and save Mel spectrogram."""
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Frames')
        plt.ylabel('Mel Filter Index')
        plt.title(f'Mel Spectrogram of {filename}')
        plt.savefig(filename)
        # plt.close()
        return plt


