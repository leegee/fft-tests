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

from AudioProcessor import AudioProcessor
from SpectrogramPlotter import SpectrogramPlotter
from SpectrogramStorage import SpectrogramStorage
from DataClusterer import DataClusterer

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and cluster WAV files.")
    parser.add_argument("path", help="Path to a WAV file or a directory containing WAV files.")
    parser.add_argument("--window_length", type=int, default=256, help="FFT window length.")
    parser.add_argument("--step_size", type=int, default=512, help="Step size for FFT.")
    parser.add_argument("--n_filters", type=int, default=24, help="Number of Mel filters.")
    parser.add_argument("--db", default='ffts.sqlite3', help="SQLite database file to store data.")
    
    args = parser.parse_args()

    # Initialize the components
    audio_processor = AudioProcessor(args.window_length, args.step_size, args.n_filters)
    storage = SpectrogramStorage(args.db)
    plotter = SpectrogramPlotter()
    clusterer = DataClusterer()

    # Process files
    if os.path.isfile(args.path):
        spectrograms = audio_processor.process_file(args.path)
        for channel, mel_spectrogram in spectrograms.items():
            plotter.plot_mel_spectrogram(mel_spectrogram, args.path.replace('.wav', f'_{channel}_mel.png'))
            storage.save_data_to_sql(mel_spectrogram, args.path.replace('.wav', f'_{channel}_mel.npy'))
    elif os.path.isdir(args.path):
        for root, dirs, files in os.walk(args.path):
            for file in files:
                if file.lower().endswith('.wav'):
                    filepath = os.path.join(root, file)
                    print(f"Processing {filepath}")
                    spectrograms = audio_processor.process_file(filepath)
                    for channel, mel_spectrogram in spectrograms.items():
                        plotter.plot_mel_spectrogram(mel_spectrogram, filepath.replace('.wav', f'_{channel}_mel.png'))
                        storage.save_data_to_sql(mel_spectrogram, filepath.replace('.wav', f'_{channel}_mel.npy'))
    else:
        raise ValueError("The provided path is neither a file nor a directory.")

    # Clustering
    mel_spectrograms = storage.fetch_data_from_sql()
    if len(mel_spectrograms) > 0:
        max_len = max(mel.shape[0] for mel in mel_spectrograms)
        num_features = mel_spectrograms[0].shape[1]
        mel_spectrograms_padded = clusterer.pad_sequences(mel_spectrograms, max_len, num_features)
        scaled_data, clusters = clusterer.cluster_data(mel_spectrograms_padded)
        clusterer.plot_clusters(scaled_data, clusters)

    storage.close()

if __name__ == "__main__":
    main()
