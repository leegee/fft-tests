import os

def wav_file_to_mel_spectrogram(filepath, audio_processor, storage, plotter):
    """Process a single WAV file and save spectrograms and plots."""
    print(f"Processing file: {filepath}")
    spectrograms = audio_processor.wav_file_to_mel_spectrogram(filepath)

    storage.save_data_to_sql(spectrograms, filepath)
    
    plot_path = filepath.replace('.wav', f'.png')
    plt = plotter.plot_mel_spectrogram(spectrograms, plot_path)
    plt.close()
    print(f"Processed and saved spectrograms for {filepath}")

def process_directory(directory_path, audio_processor, storage, plotter):
    """Process all WAV files in a directory and its subdirectories."""
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.wav'):
                filepath = os.path.join(root, file)
                wav_file_to_mel_spectrogram(filepath, audio_processor, storage, plotter)
