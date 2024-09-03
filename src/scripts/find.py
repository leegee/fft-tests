import argparse
import os
import sys
import sounddevice as sd
from scipy.io import wavfile

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config import config

from AudioProcessor import AudioProcessor
from SpectrogramStorage import SpectrogramStorage
from SpectrogramPlotter import SpectrogramPlotter
from DataClusterer import DataClusterer

def play_wav(filename):
    """Play a WAV file."""
    samplerate, data = wavfile.read(filename)
    sd.play(data, samplerate)
    sd.wait()  # Wait until the file is done playing

def main():
    parser = argparse.ArgumentParser(description="Find closest matches to a WAV file in the database.")
    parser.add_argument("wav_path", help="Path to a WAV file to find closest matches for.")
    parser.add_argument("--window_length", type=int, default=config.FFT_WINDOW_SIZE, help="FFT window length.")
    parser.add_argument("--step_size", type=int, default=config.FFT_STEP_SIZE, help="Step size for FFT.")
    parser.add_argument("--n_filters", type=int, default=config.FFT_N_FILTERS, help="Number of Mel filters.")
    parser.add_argument("--db", default=config.DB_FILE, help="SQLite database file to store data.")
    parser.add_argument("--num_matches", type=int, default=config.NUM_MATCHES, help="Number of closest matches to find.")
    
    args = parser.parse_args()

    # Initialize components
    audio_processor = AudioProcessor(args.window_length, args.step_size, args.n_filters)
    storage = SpectrogramStorage(args.db)
    plotter = SpectrogramPlotter()
    clusterer = DataClusterer()

    if not os.path.exists(args.wav_path):
        raise FileNotFoundError(f"File not found: {args.wav_path}")

    # Play the WAV file before processing
    print(f"Playing input WAV file: {args.wav_path}")
    play_wav(args.wav_path)

    # Step 1: Load and process the input WAV file
    print(f"Processing input WAV file: {args.wav_path}")
    input_spectrograms = audio_processor.process_file(args.wav_path)
    
    if not input_spectrograms:
        raise ValueError("No spectrograms were generated from the provided WAV file.")
    
    # Assume mono or take the left channel if stereo (for now)
    target_spectrogram = input_spectrograms['left'] if 'left' in input_spectrograms else input_spectrograms['mono']
    
    # Step 2: Fetch all stored spectrograms from the database
    records = storage.fetch_all_records()  # Fetch records including metadata
    
    # Prepare lists for processing
    spectrograms = [record['spectrogram'] for record in records]
    
    # Step 3: Find the closest matches
    closest_indices = clusterer.find_closest_matches(target_spectrogram, spectrograms, args.num_matches)
    
    # Step 4: Plot the closest matches
    print(f"Found {len(closest_indices)} closest matches. Plotting...")
    for idx in closest_indices:
        # Retrieve the record based on the index
        record = records[idx]
        
        print(f'Filename: {record['filename']}')
        play_wav(record['filename'])
        
        # Plot the spectrogram
        plt = plotter.plot_mel_spectrogram(record['spectrogram'], f"closest_match_{idx}.png")
        plt.show()
        print(f"Plotted closest match for filename: {record['filename']}")
        plt.close()

if __name__ == "__main__":
    main()
