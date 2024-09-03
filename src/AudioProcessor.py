import os
import numpy as np
import soundfile as sf
import librosa
from scipy.fft import fft, fftfreq

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

    def stereo_to_mono(self, input_signal):
        # Ensure the input is a numpy array
        input_signal = np.array(input_signal)
        
        # Check if the signal is stereo
        if input_signal.ndim != 2 or input_signal.shape[1] != 2:
            return input_signal
        
        # Average the two channels
        mono_signal = np.mean(input_signal, axis=1)
        
        return mono_signal

    def process_file(self, filename):
        """Process a single WAV file to compute Mel spectrograms."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        originalData, samplerate = self.load_wav(filename)

        data = self.stereo_to_mono(originalData)
        
        xf, fft_data = self.perform_fft(data, samplerate)
        mel_filters = self.mel_filterbank(samplerate)

        mel_data = self.apply_mel_filterbank(fft_data, mel_filters)

        return mel_data
