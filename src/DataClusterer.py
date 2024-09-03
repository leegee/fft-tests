import os
import io
import sqlite3
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

class DataClusterer:
    def __init__(self):
        pass

    @staticmethod
    def pad_sequences(sequences, maxlen, num_features):
        """Ensure all arrays have the same number of dimensions by padding them."""
        padded_sequences = np.zeros((len(sequences), maxlen, num_features))
        for i, seq in enumerate(sequences):
            padded_sequences[i, :seq.shape[0], :seq.shape[1]] = seq
        return padded_sequences

    def cluster_data(self, data):
        """Perform DBSCAN clustering on the data."""
        num_samples, max_len, num_features = data.shape
        reshaped_data = data.reshape(num_samples, max_len * num_features)

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(reshaped_data)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(scaled_data)
        
        return scaled_data, clusters

    def plot_clusters(self, data, clusters):
        """Plot the clusters using PCA for dimensionality reduction to 2D."""
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
        plt.colorbar(scatter)
        plt.title('WAV Clusters')
        plt.show()

