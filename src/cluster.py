import os
import io
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def fetch_data_from_sql(db_path):
    # Connect to the file database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all rows from the table where mel spectrograms are stored
    cursor.execute(f"SELECT spectrogram FROM {os.getenv('LEE_TABLE_SEPECTROGRAMS') or 'mel_spectrograms'}")
    rows = cursor.fetchall()

    # Convert fetched data to a list of arrays
    mel_spectrograms = []
    for row in rows:
        # Convert binary blob back to NumPy array
        with io.BytesIO(row[0]) as buffer:
            mel_spectrogram = np.load(buffer, allow_pickle=True)
            mel_spectrograms.append(mel_spectrogram)

    # Close the database connection
    conn.close()

    # Ensure all arrays have the same number of dimensions by padding them
    max_len = max(mel.shape[0] for mel in mel_spectrograms)
    num_features = mel_spectrograms[0].shape[1]

    def pad_sequences(sequences, maxlen, num_features):
        padded_sequences = np.zeros((len(sequences), maxlen, num_features))
        for i, seq in enumerate(sequences):
            padded_sequences[i, :seq.shape[0], :seq.shape[1]] = seq
        return padded_sequences

    mel_spectrograms_padded = pad_sequences(mel_spectrograms, max_len, num_features)

    return mel_spectrograms_padded

def cluster_data(data):
    """Perform DBSCAN clustering on the data."""
    # Reshape the 3D data into 2D: (num_samples, num_features)
    num_samples, max_len, num_features = data.shape
    reshaped_data = data.reshape(num_samples, max_len * num_features)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(reshaped_data)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    
    return scaled_data, clusters

def plot_clusters(data, clusters):
    """Plot the clusters using PCA for dimensionality reduction to 2D."""
    # Use PCA to reduce dimensionality to 2D for plotting
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    # Ensure the number of clusters matches the number of data points
    if len(clusters) != data_2d.shape[0]:
        raise ValueError(f"Mismatch between number of clusters ({len(clusters)}) and number of data points ({data_2d.shape[0]})")
    
    # Plot the clusters
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
    plt.colorbar(scatter)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clusters of Mel Spectrograms')
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform DBSCAN clustering on Mel spectrogram data stored in SQLite.")
    parser.add_argument("--db", default=(os.getenv('LEE_DB_FILE') or './ffts.sqlite3'), help="Path to the SQLite database file.")
    
    args = parser.parse_args()

    print(f'Using DB {args.db}')

    if not(os.path.isfile(args.db)):
        raise ValueError(f"Cannot find the DB file at {args.db}")
    
    mel_spectrograms = fetch_data_from_sql(args.db)
    
    if mel_spectrograms.size == 0:
        print(f"No data found in the database {args.db}")
        return
    
    # Cluster the data
    scaled_data, clusters = cluster_data(mel_spectrograms)

    # Plot the clusters
    plot_clusters(scaled_data, clusters)
    
    print("Clustering complete. Number of clusters found:", len(set(clusters)) - (1 if -1 in clusters else 0))

if __name__ == "__main__":
    main()
