import os
import sys
import numpy as np

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config import config
from SpectrogramStorage import SpectrogramStorage
from DataClusterer import DataClusterer

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and cluster WAV files.")
    parser.add_argument("--db", default=config.DB_FILE, help="SQLite database file to store data.")
    
    args = parser.parse_args()

    storage = SpectrogramStorage(args.db)
    clusterer = DataClusterer()

    # Clustering
    print(f"Clustering")
    mel_spectrograms = storage.fetch_all_spectrograms()
    
    if mel_spectrograms:
        print(f"Fetched {len(mel_spectrograms)} spectrograms.")
        valid_spectrograms = []
        
        for idx, mel in enumerate(mel_spectrograms):
            if isinstance(mel, np.ndarray) and mel.ndim >= 2:
                print(f"Spectrogram {idx} shape: {mel.shape}")
                valid_spectrograms.append(mel)
            else:
                print(f"Spectrogram {idx} is invalid: {type(mel)}")

        if valid_spectrograms:
            try:
                max_len = max(mel.shape[0] for mel in valid_spectrograms)
                num_features = valid_spectrograms[0].shape[1]
                mel_spectrograms_padded = clusterer.pad_sequences(valid_spectrograms, max_len, num_features)
                scaled_data, clusters = clusterer.cluster_data(mel_spectrograms_padded)
                plt = clusterer.plot_clusters(scaled_data, clusters)
                plt.show()
                plt.close()
            except Exception as e:
                print(f"Error during clustering: {e}")
        else:
            print("No valid spectrograms available for clustering.")

    storage.close()
    print(f"Done")

if __name__ == "__main__":
    main()
