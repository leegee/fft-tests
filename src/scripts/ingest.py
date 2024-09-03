import sys
import os

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from AudioProcessor import AudioProcessor
from SpectrogramPlotter import SpectrogramPlotter
from SpectrogramStorage import SpectrogramStorage

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and cluster WAV files.")
    parser.add_argument("path", help="Path to a WAV file or a directory containing WAV files.")
    parser.add_argument("--window_length", type=int, default=256, help="FFT window length.")
    parser.add_argument("--step_size", type=int, default=512, help="Step size for FFT.")
    parser.add_argument("--n_filters", type=int, default=24, help="Number of Mel filters.")
    parser.add_argument("--db", default='ffts.sqlite3', help="SQLite database file to store data.")
    
    args = parser.parse_args()

    # Ought to only initialize the components needed as determined by cmd line args.
    audio_processor = AudioProcessor(args.window_length, args.step_size, args.n_filters)
    storage = SpectrogramStorage(args.db)
    plotter = SpectrogramPlotter()

    # Process files
    if os.path.isfile(args.path):
        print(f"Processing supplied file {args.filepath}")
        spectrograms = audio_processor.process_file(args.path)
        for channel, mel_spectrogram in spectrograms.items():
            plt = plotter.plot_mel_spectrogram(mel_spectrogram, args.path.replace('.wav', f'_{channel}_mel.png'))
            plt.close()
            storage.save_data_to_sql(mel_spectrogram, args.path.replace('.wav', f'_{channel}_mel.npy'))
    elif os.path.isdir(args.path):
        for root, dirs, files in os.walk(args.path):
            for file in files:
                if file.lower().endswith('.wav'):
                    filepath = os.path.join(root, file)
                    print(f"Found file {filepath}")
                    spectrograms = audio_processor.process_file(filepath)
                    for channel, mel_spectrogram in spectrograms.items():
                        plt = plotter.plot_mel_spectrogram(mel_spectrogram, filepath.replace('.wav', f'_{channel}_mel.png'))
                        plt.close()
                        storage.save_data_to_sql(mel_spectrogram, filepath.replace('.wav', f'_{channel}_mel.npy'))
    else:
        raise ValueError("The provided path is neither a file nor a directory.")

    storage.close()
    print(f"Done")

if __name__ == "__main__":
    main()
