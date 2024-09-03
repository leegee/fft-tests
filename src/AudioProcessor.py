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

class AudioProcessor:
    def __init__(self, window_length=1024, step_size=512, n_filters=24):
        self.window_length = window_length
        self.step_size = step_size
        self.n_filters = n_filters

    def load_wav(self, filename):
        """Load WAV file."""
        data, samplerate = sf.read(filename)
        return data, samplerate

    def perform_fft(self, data, samplerate):
        """Perform FFT on data with specified window length and step size."""
        N = len(data)
        T = 1.0 / samplerate
        num_windows = (N - self.window_length) // self.step_size + 1
        xf = fftfreq(self.window_length, T)[:self.window_length // 2]

        fft_results = np.zeros((num_windows, self.window_length // 2))
        for i in range(num_windows):
            start = i * self.step_size
            end = start + self.window_length
            if end > N:
                break
            segment = data[start:end]
            windowed_segment = segment * np.hanning(self.window_length)
            fft_result = fft(windowed_segment)
            fft_results[i] = 2.0 / self.window_length * np.abs(fft_result[:self.window_length // 2])
        
        return xf, fft_results

    def mel_filterbank(self, samplerate):
        """Generate Mel filterbank."""
        mel_filters = librosa.filters.mel(sr=samplerate, n_fft=(self.window_length-1), n_mels=(self.n_filters-1))
        return mel_filters

    def apply_mel_filterbank(self, fft_results, mel_filters):
        """Apply Mel filterbank to FFT results."""
        fft_bins = fft_results.shape[1]
        mel_bins = mel_filters.shape[1]

        if fft_bins != mel_bins:
            raise ValueError(f"Mismatch between FFT results and Mel filterbank dimensions: {fft_bins} != {mel_bins}")

        return np.dot(fft_results, mel_filters.T)

    def process_file(self, filename):
        """Process a single WAV file to compute Mel spectrograms."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        data, samplerate = self.load_wav(filename)
        n_fft = self.window_length
        
        if len(data.shape) == 2:  # Stereo
            left_channel = data[:, 0]
            right_channel = data[:, 1]
            
            xf_left, fft_left = self.perform_fft(left_channel, samplerate)
            xf_right, fft_right = self.perform_fft(right_channel, samplerate)
            
            mel_filters = self.mel_filterbank(samplerate)
            
            mel_left = self.apply_mel_filterbank(fft_left, mel_filters)
            mel_right = self.apply_mel_filterbank(fft_right, mel_filters)

            return {'left': mel_left, 'right': mel_right}
        else:  # Mono
            xf, fft_data = self.perform_fft(data, samplerate)
            mel_filters = self.mel_filterbank(samplerate)
            mel_data = self.apply_mel_filterbank(fft_data, mel_filters)
            return {'mono': mel_data}


