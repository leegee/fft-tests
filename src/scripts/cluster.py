import os
import sys

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from SpectrogramStorage import SpectrogramStorage
from DataClusterer import DataClusterer

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and cluster WAV files.")
    parser.add_argument("--db", default='ffts.sqlite3', help="SQLite database file to store data.")
    
    args = parser.parse_args()

    storage = SpectrogramStorage(args.db)
    clusterer = DataClusterer()

    # Clustering
    print(f"Clustering")
    mel_spectrograms = storage.fetch_all_spectrograms()
    if len(mel_spectrograms) > 0:
        max_len = max(mel.shape[0] for mel in mel_spectrograms)
        num_features = mel_spectrograms[0].shape[1]
        mel_spectrograms_padded = clusterer.pad_sequences(mel_spectrograms, max_len, num_features)
        scaled_data, clusters = clusterer.cluster_data(mel_spectrograms_padded)
        plt = clusterer.plot_clusters(scaled_data, clusters)
        plt.show()
        plt.close()

    storage.close()
    print(f"Done")

if __name__ == "__main__":
    main()
