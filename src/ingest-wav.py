import os
import io
import sqlite3
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def load_wav(filename):
    """Load WAV file."""
    data, samplerate = sf.read(filename)
    return data, samplerate

def perform_fft(data, samplerate, window_length, step_size):
    """Perform FFT on data with specified window length and step size."""
    N = len(data)
    T = 1.0 / samplerate
    num_windows = (N - window_length) // step_size + 1
    xf = fftfreq(window_length, T)[:window_length // 2]

    # Prepare to store FFT results
    fft_results = np.zeros((num_windows, window_length // 2))
    
    for i in range(num_windows):
        start = i * step_size
        end = start + window_length
        if end > N:
            break
        segment = data[start:end]
        windowed_segment = segment * np.hanning(window_length)
        fft_result = fft(windowed_segment)
        fft_results[i] = 2.0 / window_length * np.abs(fft_result[:window_length // 2])
    
    return xf, fft_results

def mel_filterbank(samplerate, n_filters, n_fft):
    """Generate Mel filterbank."""
    mel_filters = librosa.filters.mel(sr=samplerate, n_fft=(n_fft-1), n_mels=(n_filters-1))
    return mel_filters

def apply_mel_filterbank(fft_results, mel_filters):
    """Apply Mel filterbank to FFT results."""
    fft_bins = fft_results.shape[1]
    mel_bins = mel_filters.shape[1] 
    
    if fft_bins != mel_bins:
        raise ValueError(f"Mismatch between FFT results and Mel filterbank dimensions: {fft_bins} != {mel_bins}")
    
    return np.dot(fft_results, mel_filters.T)

def plot_mel_spectrogram(mel_spectrogram, filename):
    """Plot and save Mel spectrogram."""
    plt.figure(figsize=(12, 6))
    plt.imshow(mel_spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Frames')
    plt.ylabel('Mel Filter Index')
    plt.title(f'Mel Spectrogram of {filename}')
    plt.savefig(filename)
    plt.close()

def save_data(mel_spectrogram, filename):
    """Save Mel spectrogram data to a NumPy file."""
    np.save(filename, mel_spectrogram)

def save_data_to_sql(mel_spectrogram, filename, db_connection):
    """Save Mel spectrogram data to an SQLite database."""
    cursor = db_connection.cursor()
    
    # Create table if not exists
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {os.getenv('LEE_TABLE_SEPECTROGRAMS', 'mel_spectrograms')} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            spectrogram BLOB
        )
    ''')

    # Convert mel_spectrogram to binary format
    with io.BytesIO() as buffer:
        np.save(buffer, mel_spectrogram)
        blob = buffer.getvalue()
    
    # Insert data into the database
    cursor.execute(f'''
        INSERT INTO {os.getenv('LEE_TABLE_SEPECTROGRAMS', 'mel_spectrograms')} (filename, spectrogram)
        VALUES (?, ?)
    ''', (filename, blob))
    
    db_connection.commit()

def process_file(filename, window_length=1024, step_size=512, n_filters=24, db_connection=None):
    """Process a single WAV file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    data, samplerate = load_wav(filename)
    n_fft = window_length
    
    if len(data.shape) == 2:
        left_channel = data[:, 0]
        right_channel = data[:, 1]
        
        xf_left, fft_left = perform_fft(left_channel, samplerate, window_length, step_size)
        xf_right, fft_right = perform_fft(right_channel, samplerate, window_length, step_size)
        
        mel_filters = mel_filterbank(samplerate, n_filters, n_fft)
        
        mel_left = apply_mel_filterbank(fft_left, mel_filters)
        mel_right = apply_mel_filterbank(fft_right, mel_filters)
        
        plot_mel_spectrogram(mel_left, filename.replace('.wav', '_left_mel.png'))
        plot_mel_spectrogram(mel_right, filename.replace('.wav', '_right_mel.png'))

        save_data(mel_left, filename.replace('.wav', '_left_mel.npy'))
        save_data(mel_right, filename.replace('.wav', '_right_mel.npy'))

        if db_connection:
            save_data_to_sql(mel_left, filename.replace('.wav', '_left_mel.npy'), db_connection)
            save_data_to_sql(mel_right, filename.replace('.wav', '_right_mel.npy'), db_connection)

    else:
        xf, fft_data = perform_fft(data, samplerate, window_length, step_size)
        
        mel_filters = mel_filterbank(samplerate, n_filters, n_fft)
        mel_data = apply_mel_filterbank(fft_data, mel_filters)
        
        plot_mel_spectrogram(mel_data, filename.replace('.wav', '_mel.png'))
        save_data(mel_data, filename.replace('.wav', '_mel.npy'))

        if db_connection:
            save_data_to_sql(mel_data, filename.replace('.wav', '_mel.npy'), db_connection)

def process_directory(directory, window_length=1024, step_size=512, n_filters=24, db_connection=None):
    """Process all WAV files in a directory recursively."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                filepath = os.path.join(root, file)
                print(f"Processing {filepath}")
                process_file(filepath, window_length, step_size, n_filters, db_connection)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process WAV files to generate Mel spectrograms.")
    parser.add_argument("path", help="Path to a WAV file or a directory containing WAV files.")
    parser.add_argument("--window_length", type=int, default=256, help="FFT window length.")
    parser.add_argument("--step_size", type=int, default=512, help="Step size for FFT.")
    parser.add_argument("--n_filters", type=int, default=24, help="Number of Mel filters.")
    parser.add_argument("--db", default='ffts.sqlite3', help="SQLite database file to store data.")
    
    args = parser.parse_args()
    
    db_connection = None
    if args.db:
        db_connection = sqlite3.connect(args.db);

    if os.path.isfile(args.path):
        process_file(args.path, args.window_length, args.step_size, args.n_filters, db_connection)
    elif os.path.isdir(args.path):
        process_directory(args.path, args.window_length, args.step_size, args.n_filters, db_connection)
    else:
        raise ValueError("The provided path is neither a file nor a directory.")

    if db_connection:
        db_connection.close()

if __name__ == "__main__":
    main()
