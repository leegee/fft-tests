import argparse
import os
import sys

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config import config
from AudioProcessor import AudioProcessor
from SpectrogramPlotter import SpectrogramPlotter
from SpectrogramStorage import SpectrogramStorage
from Ingester import Ingester

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
    ingester = Ingester()

    # Process files or directories
    if os.path.isfile(args.path):
        ingester.wav_file_to_mel_spectrogram(args.path, audio_processor, storage, plotter)
    elif os.path.isdir(args.path):
        ingester.process_directory(args.path, audio_processor, storage, plotter)
    else:
        raise ValueError("The provided path is neither a file nor a directory.")

    storage.close()
    print("Done")

if __name__ == "__main__":
    main()
