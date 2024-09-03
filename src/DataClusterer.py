import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

class DataClusterer:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    
    def cluster_data(self, data):
        """Perform DBSCAN clustering on the data."""
        # Reshape the 3D data into 2D: (num_samples, num_features)
        num_samples, max_len, num_features = data.shape
        reshaped_data = data.reshape(num_samples, max_len * num_features)

        # Standardize the data
        scaled_data = self.scaler.fit_transform(reshaped_data)
        
        # Perform DBSCAN clustering
        clusters = self.dbscan.fit_predict(scaled_data)
        
        return scaled_data, clusters

    def plot_clusters(self, data, clusters):
        """Plot the clusters using PCA for dimensionality reduction to 2D."""
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        if len(clusters) != data_2d.shape[0]:
            raise ValueError(f"Mismatch between number of clusters ({len(clusters)}) and number of data points ({data_2d.shape[0]})")
        
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
        plt.colorbar(scatter)
        plt.title('WAV Clusters')
        return plt
    
    def pad_sequences(self, sequences, max_len, num_features):
        """Pad sequences to a uniform length."""
        padded_sequences = np.zeros((len(sequences), max_len, num_features))
        for i, seq in enumerate(sequences):
            seq_len = min(seq.shape[0], max_len)
            padded_sequences[i, :seq_len, :] = seq[:seq_len, :]
        return padded_sequences
    
    def pad_spectrograms(self, spectrograms):
        """
        Pad all spectrograms to the maximum shape found in the list.
        
        Args:
            spectrograms (list of numpy.ndarray): List of spectrograms to pad.
        
        Returns:
            numpy.ndarray: Padded spectrograms in a uniform 3D array.
        """
        # Determine the maximum shape for padding
        max_rows = max(s.shape[0] for s in spectrograms)
        max_cols = max(s.shape[1] for s in spectrograms)

        # Pad each spectrogram to the max shape
        padded_spectrograms = np.array([
            np.pad(s, ((0, max_rows - s.shape[0]), (0, max_cols - s.shape[1])), mode='constant')
            for s in spectrograms
        ])
        return padded_spectrograms

    def find_closest_matches(self, target_spectrogram, spectrograms, num_matches=5):
        """
        Find the closest matches to the target spectrogram from a list of spectrograms.
        
        Args:
            target_spectrogram (numpy.ndarray): The target spectrogram to compare against.
            spectrograms (list of numpy.ndarray): List of stored spectrograms.
            num_matches (int): Number of closest matches to find.

        Returns:
            list of int: Indices of the closest matches in the spectrograms list.
        """
        # Pad all spectrograms to have the same shape
        padded_spectrograms = self.pad_spectrograms(spectrograms)
        
        # Reshape spectrograms to 2D for distance calculation
        reshaped_spectrograms = padded_spectrograms.reshape(padded_spectrograms.shape[0], -1)
        reshaped_target = np.pad(target_spectrogram, ((0, padded_spectrograms.shape[1] - target_spectrogram.shape[0]), (0, padded_spectrograms.shape[2] - target_spectrogram.shape[1])), mode='constant').reshape(1, -1)

        # Compute distances between the target and all stored spectrograms
        distances = euclidean_distances(reshaped_target, reshaped_spectrograms).flatten()

        # Get the indices of the closest matches
        closest_indices = np.argsort(distances)[:num_matches]

        return closest_indices    