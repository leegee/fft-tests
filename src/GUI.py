import sys
import os
import sounddevice as sd  
import soundfile as sf
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QScrollArea, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QAction

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config import config
from AudioProcessor import AudioProcessor
from DataClusterer import DataClusterer
from SpectrogramStorage import SpectrogramStorage
from SpectrogramPlotter import SpectrogramPlotter
from ClickableQLabel import ClickableQLabel

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Application")
        self.setGeometry(100, 100, 800, 600)
        self.storage = SpectrogramStorage()
        self.audio_processor = AudioProcessor(config.FFT_WINDOW_SIZE, config.FFT_STEP_SIZE, config.FFT_N_FILTERS)
        self.clusterer = DataClusterer()
        self.plotter = SpectrogramPlotter()
        self.init_ui()

    def init_ui(self):
        # Create the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create Table Widget
        self.table_widget = QTableWidget()
        self.table_widget.setStyleSheet("""
            QTableWidget::item:selected {
                background-color: #222299;
                color: white;
            }
        """)
        self.table_widget.setColumnCount(3)  
        self.table_widget.setHorizontalHeaderLabels(["ID", "Filename", "Spectrogram"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Create Scroll Area for Table Widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.table_widget)
        layout.addWidget(scroll_area)

        # Initialize Table Content
        self.update_table()

        # Create the Menu Bar
        menu_bar = self.menuBar()

        # Create Database Menu
        database_menu = menu_bar.addMenu("&Database")

        # Add Wipe Database action
        wipe_action = QAction("&Wipe Database", self)
        wipe_action.triggered.connect(self.wipe_database)
        database_menu.addAction(wipe_action)

        # Create Ingest Menu
        ingest_menu = menu_bar.addMenu("&Add")

        # Add Ingest File action
        ingest_file_action = QAction("Add &File", self)
        ingest_file_action.triggered.connect(self.ingest_file)
        ingest_menu.addAction(ingest_file_action)

        # Add Ingest Directory action
        ingest_directory_action = QAction("Add &Directory", self)
        ingest_directory_action.triggered.connect(self.ingest_directory)
        ingest_menu.addAction(ingest_directory_action)

        # Create Match Menu
        match_menu = menu_bar.addMenu("&Find")

        # Add Find Closest Match action
        find_match_action = QAction("&Find Closest Match", self)
        find_match_action.triggered.connect(self.find_closest_match)
        match_menu.addAction(find_match_action)

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
            self.storage.drop_table
            self.update_table()  

    def ingest_file(self):
        """Ingest a file into the database."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", filter="WAV Files (*.wav)")
        if file_path:
            mel_spectrogram = self.audio_processor.wav_file_to_mel_spectrogram(file_path)
            self.storage.save_data_to_sql(mel_spectrogram, file_path)
            self.update_table() 

    def ingest_directory(self):
        """Add all files from a directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select a directory")
        if dir_path:
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.wav'):  # Only process WAV files
                    file_path = os.path.join(dir_path, file_name)
                    mel_spectrogram = self.audio_processor.wav_file_to_mel_spectrogram(file_path)
                    self.storage.save_data_to_sql(mel_spectrogram, file_path)
            self.update_table()

    def find_closest_match(self):
        """Find the closest match for a file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Select the subject of your search", filter="WAV Files (*.wav)")

        if filepath:
            print(f'Filepath {filepath}')
            mel_spectrogram = self.audio_processor.wav_file_to_mel_spectrogram(filepath)
            plot_path = filepath.replace('.wav', f'.png')
            plt = self.plotter.plot_mel_spectrogram(mel_spectrogram, plot_path)
            print(f'Show {plot_path}')
            plt.close()
            closest_match_ids = self.clusterer.find_closest_matches_in_db(mel_spectrogram)
            self.update_table()
            self.select_table_row(closest_match_ids[0])

    def update_table(self):
        """Update the table with data from the database."""
        records = self.storage.fetch_ids_and_paths()
        self.table_widget.setRowCount(len(records))
        for row, (record_id, filename) in enumerate(records):
            self.table_widget.setItem(row, 0, QTableWidgetItem(str(record_id)))
            self.table_widget.setItem(row, 1, QTableWidgetItem(filename))

            # Add Spectrogram image with clickable label
            plot_path = filename.replace('.wav', '.png')
            if os.path.exists(plot_path):
                pixmap = QPixmap(plot_path)
                image_label = ClickableQLabel()
                image_label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))  
                # Connect the clicked signal to play the audio file
                image_label.clicked.connect(lambda file=filename: self.play_audio_file(file))
                self.table_widget.setCellWidget(row, 2, image_label)
            else:
                self.table_widget.setCellWidget(row, 2, QLabel("No Image"))


    def play_audio_file(self, file_path):
        """Play an audio file given its file path."""
        try:
            print(f'Play {file_path}')
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()  # Wait until the audio finishes playing
        except Exception as e:
            print(f"Error playing file {file_path}: {e}")
            QMessageBox.critical(self, "Error", f"Failed to play audio file: {e}")

    def select_table_row(self, match_id):
        """Select the row in the table corresponding to the match_id."""
        match_id = int(match_id)
        if not isinstance(match_id, int):  # Ensure match_id is an integer
            print("Error: match_id is not an integer:", match_id)
            return

        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)  # ID is in the first column
            if item and int(item.text()) == match_id:
                print(f'Table select row for {item.text()} == {match_id}')
                self.table_widget.selectRow(row)
                self.table_widget.cellWidget(row, 2).click()  # Automatically click the Play button
                break

