import argparse
import os
import sys

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config import config
from AudioProcessor import AudioProcessor
from SpectrogramPlotter import SpectrogramPlotter
from SpectrogramStorage import SpectrogramStorage

def process_file(filepath, audio_processor, storage, plotter):
    """Process a single WAV file and save spectrograms and plots."""
    print(f"Processing file: {filepath}")
    spectrograms = audio_processor.process_file(filepath)

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
                process_file(filepath, audio_processor, storage, plotter)

def main():
    parser = argparse.ArgumentParser(description="Process and cluster WAV files.")
    parser.add_argument("path", help="Path to a WAV file or a directory containing WAV files.")
    parser.add_argument("--window_length", type=int, default=config.FFT_WINDOW_SIZE, help="FFT window length.")
    parser.add_argument("--step_size", type=int, default=config.FFT_STEP_SIZE, help="Step size for FFT.")
    parser.add_argument("--n_filters", type=int, default=config.FFT_N_FILTERS, help="Number of Mel filters.")
    parser.add_argument("--db", default=config.DB_FILE, help="SQLite database file to store data.")
    
    args = parser.parse_args()

    # Initialize components
    audio_processor = AudioProcessor(args.window_length, args.step_size, args.n_filters)
    storage = SpectrogramStorage(args.db)
    plotter = SpectrogramPlotter()

    # Process files or directories
    if os.path.isfile(args.path):
        process_file(args.path, audio_processor, storage, plotter)
    elif os.path.isdir(args.path):
        process_directory(args.path, audio_processor, storage, plotter)
    else:
        raise ValueError("The provided path is neither a file nor a directory.")

    storage.close()
    print("Done")

if __name__ == "__main__":
    main()
