import io
import os
import sys
import sqlite3
import numpy as np

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config import config

class SpectrogramStorage:
    def __init__(self, db_file=config.DB_FILE):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.create_table()

    def create_table(self):
        """Create table to store Mel spectrograms."""
        cursor = self.conn.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {config.TABLE_SEPECTROGRAMS} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                spectrogram BLOB
            )
        ''')
        self.conn.commit()

    def save_data_to_sql(self, mel_spectrogram, filename):
        """Save Mel spectrogram data to an SQLite database."""
        
        # Serialize the numpy array to a binary format
        with io.BytesIO() as buffer:
            np.save(buffer, mel_spectrogram)
            blob = buffer.getvalue()
        
        # Insert into the database
        cursor = self.conn.cursor()
        cursor.execute(f'''
            INSERT INTO {config.TABLE_SEPECTROGRAMS} (filename, spectrogram)
            VALUES (?, ?)
        ''', (filename, blob))
        
        self.conn.commit()
    
    def fetch_all_spectrograms(self):
        """Fetch all Mel spectrogram data from the SQLite database."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT spectrogram FROM {config.TABLE_SEPECTROGRAMS}")
        rows = cursor.fetchall()
        
        mel_spectrograms = []
        for row in rows:
            with io.BytesIO(row[0]) as buffer:
                mel_spectrogram = np.load(buffer, allow_pickle=True)
                mel_spectrograms.append(mel_spectrogram)
        
        return mel_spectrograms

    def fetch_ids_and_paths(self):
        """Fetch IDs and paths from the SQLite3 database."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id, filename FROM {config.TABLE_SEPECTROGRAMS}")
        records = cursor.fetchall()
        return records

    def fetch_all_records(self):
        """Fetch all Mel spectrogram data and associated metadata from the SQLite database."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id, spectrogram, filename FROM {config.TABLE_SEPECTROGRAMS}")
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            record_id, spectrogram_data, filename = row
            # Use io.BytesIO to handle the binary data
            with io.BytesIO(spectrogram_data) as buffer:
                mel_spectrogram = np.load(buffer, allow_pickle=True)
                records.append({
                    'id': record_id,
                    'spectrogram': mel_spectrogram,
                    'filename': filename
                })
        
        return records

    def close(self):
        print(f"Closing DB connection")
        self.conn.close()
        print(f"Closed DB connection")

