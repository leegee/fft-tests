import os
import io
import sqlite3
import numpy as np

class SpectrogramStorage:
    def __init__(self, db_path='ffts.sqlite3'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self):
        """Create table to store Mel spectrograms."""
        cursor = self.conn.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {os.getenv('LEE_TABLE_SEPECTROGRAMS', 'mel_spectrograms')} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                spectrogram BLOB
            )
        ''')
        self.conn.commit()

    def save_data_to_sql(self, mel_spectrogram, filename):
        """Save Mel spectrogram data to an SQLite database."""
        cursor = self.conn.cursor()
        with io.BytesIO() as buffer:
            np.save(buffer, mel_spectrogram)
            blob = buffer.getvalue()
        
        cursor.execute(f'''
            INSERT INTO {os.getenv('LEE_TABLE_SEPECTROGRAMS', 'mel_spectrograms')} (filename, spectrogram)
            VALUES (?, ?)
        ''', (filename, blob))
        
        self.conn.commit()

    def fetch_all(self):
        """Fetch all Mel spectrogram data from the SQLite database."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT spectrogram FROM {os.getenv('LEE_TABLE_SEPECTROGRAMS', 'mel_spectrograms')}")
        rows = cursor.fetchall()
        
        mel_spectrograms = []
        for row in rows:
            with io.BytesIO(row[0]) as buffer:
                mel_spectrogram = np.load(buffer, allow_pickle=True)
                mel_spectrograms.append(mel_spectrogram)
        
        return mel_spectrograms

    def close(self):
        print(f"Closing DB connection")
        self.conn.close()
        print(f"Closed DB connection")

