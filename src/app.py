import sys
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt
from SpectrogramStorage import SpectrogramStorage  # Import your existing SpectrogramStorage class

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config import config
from AudioProcessor import AudioProcessor
from DataClusterer import DataClusterer

class AudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Application")
        self.setGeometry(100, 100, 800, 600)
        self.storage = SpectrogramStorage()
        self.audio_processor = AudioProcessor(config.FFT_WINDOW_SIZE, config.FFT_STEP_SIZE, config.FFT_N_FILTERS)
        self.clusterer = DataClusterer();
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create Buttons
        wipe_button = QPushButton("Wipe Database")
        wipe_button.clicked.connect(self.wipe_database)
        layout.addWidget(wipe_button)

        ingest_file_button = QPushButton("Ingest File")
        ingest_file_button.clicked.connect(self.ingest_file)
        layout.addWidget(ingest_file_button)

        ingest_directory_button = QPushButton("Ingest Directory")
        ingest_directory_button.clicked.connect(self.ingest_directory)
        layout.addWidget(ingest_directory_button)

        find_match_button = QPushButton("Find Closest Match")
        find_match_button.clicked.connect(self.find_closest_match)
        layout.addWidget(find_match_button)

        # Create Table Widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["ID", "Filename"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Create Scroll Area for Table Widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.table_widget)
        layout.addWidget(scroll_area)

        # Initialize Table Content
        self.update_table()

    def wipe_database(self):
        """Wipe the database with user confirmation."""
        reply = QMessageBox.question(
            self,
            'Confirm Wipe',
            'Are you sure you want to wipe the database?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.storage.conn.cursor().execute("DROP TABLE IF EXISTS mel_spectrograms")
            self.storage.create_table()
            QMessageBox.information(self, "Info", "Database wiped successfully!")
            self.update_table()  # Refresh the table after wiping the database

    def ingest_file(self):
        """Ingest a file into the database."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            mel_spectrogram = self.audio_processor.wav_file_to_mel_spectrogram(file_path)
            self.storage.save_data_to_sql(mel_spectrogram, file_path)
            self.update_table()  # Refresh the table after ingesting a file

    def ingest_directory(self):
        """Ingest all files in a directory into the database."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                mel_spectrogram = self.wav_file_to_mel_spectrogram(file_path)
                self.storage.save_data_to_sql(mel_spectrogram, file_path)
            self.update_table()  # Refresh the table after ingesting a directory

    def find_closest_match(self):
        """Find the closest match for a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            mel_spectrogram = self.audio_processor.wav_file_to_mel_spectrogram(file_path)
            closest_match = self.clusterer.find_closest_matches_in_db(mel_spectrogram)
            # closest_indices = clusterer.find_closest_matches(target_spectrogram, spectrograms, args.num_matches)
            QMessageBox.information(self, "Closest Match", f"The closest match is: {closest_match}")

    def update_table(self):
        """Update the table with data from the database."""
        records = self.storage.fetch_ids_and_paths()
        self.table_widget.setRowCount(len(records))
        for row, (record_id, filename) in enumerate(records):
            self.table_widget.setItem(row, 0, QTableWidgetItem(str(record_id)))
            self.table_widget.setItem(row, 1, QTableWidgetItem(filename))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioApp()
    window.show()
    sys.exit(app.exec())
