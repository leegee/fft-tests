# find_closest_matches.py

import argparse
import os
import sys

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from AudioProcessor import AudioProcessor
from SpectrogramStorage import SpectrogramStorage
from SpectrogramPlotter import SpectrogramPlotter
from DataClusterer import DataClusterer

def main():
    parser = argparse.ArgumentParser(description="Find closest matches to a WAV file in the database.")
    parser.add_argument("wav_path", help="Path to a WAV file to find closest matches for.")
    parser.add_argument("--window_length", type=int, default=256, help="FFT window length.")
    parser.add_argument("--step_size", type=int, default=512, help="Step size for FFT.")
    parser.add_argument("--n_filters", type=int, default=24, help="Number of Mel filters.")
    parser.add_argument("--db", default='ffts.sqlite3', help="SQLite database file to store data.")
    parser.add_argument("--num_matches", type=int, default=5, help="Number of closest matches to find.")
    
    args = parser.parse_args()

    # Initialize components
    audio_processor = AudioProcessor(args.window_length, args.step_size, args.n_filters)
    storage = SpectrogramStorage(args.db)
    plotter = SpectrogramPlotter()
    clusterer = DataClusterer()

    if not os.path.exists(args.wav_path):
        raise FileNotFoundError(f"File not found: {args.wav_path}")

    # Step 1: Load and process the input WAV file
    print(f"Processing input WAV file: {args.wav_path}")
    input_spectrograms = audio_processor.process_file(args.wav_path)
    
    if not input_spectrograms:
        raise ValueError("No spectrograms were generated from the provided WAV file.")
    
    # Assume mono or take the left channel if stereo (for now)
    target_spectrogram = input_spectrograms['left'] if 'left' in input_spectrograms else input_spectrograms['mono']
    
    # Step 2: Fetch all stored spectrograms from the database
    stored_spectrograms = storage.fetch_all()
    
    # Step 3: Find the closest matches
    closest_indices = clusterer.find_closest_matches(target_spectrogram, stored_spectrograms, args.num_matches)
    
    # Step 4: Plot the closest matches
    print(f"Found {len(closest_indices)} closest matches. Plotting...")
    for idx in closest_indices:
        closest_spectrogram = stored_spectrograms[idx]
        plt = plotter.plot_mel_spectrogram(closest_spectrogram, f"closest_match_{idx}.png")
        plt.show()
        print(f"Plotted closest match index: {idx}")
        plt.close()

if __name__ == "__main__":
    main()
