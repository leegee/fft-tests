import os
import io
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def fetch_data_from_sql(db_path):
    # Connect to the file database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all rows from the table where mel spectrograms are stored
    cursor.execute(f"SELECT spectrogram FROM {os.getenv('LEE_TABLE_SEPECTROGRAMS', 'mel_spectrograms')}")
    rows = cursor.fetchall()

    # Convert fetched data to a list of arrays
    mel_spectrograms = []
    for row in rows:
        # Assuming mel_spectrogram is stored as a comma-separated string
        mel_spectrogram = np.fromstring(row[0], sep=',')
        mel_spectrograms.append(mel_spectrogram)

    # Close the database connection
    conn.close()

    # Check for the maximum length
    max_len = max(len(seq) for seq in mel_spectrograms)

    # Pad sequences to the same length
    def pad_sequences(sequences, maxlen):
        padded_sequences = np.zeros((len(sequences), maxlen))
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq[:maxlen]
        return padded_sequences

    mel_spectrograms_padded = pad_sequences(mel_spectrograms, max_len)

    return mel_spectrograms_padded

def cluster_data(data):
    """Perform DBSCAN clustering on the data."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    
    return clusters

def plot_clusters(data, clusters):
    """Plot clustering results."""
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering Results')
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform DBSCAN clustering on Mel spectrogram data stored in SQLite.")
    parser.add_argument("--db", default=(os.getenv('LEE_DB_FILE', './ffts.sqlite3')), help="Path to the SQLite database file.")
    
    args = parser.parse_args()

    print(f'Using DB {args.db}')

    if not(os.path.isfile(args.db)):
        raise ValueError(f"Cannot find the DB file at {args.db}")
    
    mel_spectrograms = fetch_data_from_sql(args.db)
    
    if mel_spectrograms.size == 0:
        print(f"No data found in the database {args.db}")
        return
    
    clusters = cluster_data(mel_spectrograms)
    
    # Optional: If the data has more than 2 dimensions, you may want to perform dimensionality reduction for plotting.
    # Here, for simplicity, we assume data is 2D or we are plotting only the first two features.
    plot_clusters(mel_spectrograms, clusters)
    
    print("Clustering complete. Number of clusters found:", len(set(clusters)) - (1 if -1 in clusters else 0))

if __name__ == "__main__":
    main()
