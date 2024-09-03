import os
import sys
import sqlite3
import dearpygui.dearpygui as dpg
import sounddevice as sd
import numpy as np
from scipy.io import wavfile

# Dynamically add 'src' to the module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Config import config
from SpectrogramStorage import SpectrogramStorage

def play_sound(filename):
    """Play the WAV file."""
    samplerate, data = wavfile.read(filename)
    sd.play(data, samplerate)
    sd.wait()  # Wait until the file is done playing

def play_button_callback(sender, app_data, user_data):
    """Callback for play button to play the corresponding sound."""
    filename = user_data
    play_sound(filename)

def create_table(records):
    """Create a table in Dear PyGui and add rows from the records."""
    with dpg.table(header_row=True):
        dpg.add_table_column(label="ID")
        dpg.add_table_column(label="Filename")
        dpg.add_table_column(label="Play Sound")

        for record in records:
            row_id, filename = record
            with dpg.table_row():
                dpg.add_text(str(row_id))
                dpg.add_text(filename)
                # Add a button in the table row to play the sound
                button_id = f"play_{row_id}"
                dpg.add_button(label="Play", tag=button_id, callback=play_button_callback, user_data=filename)

def main():
    # Initialize Dear PyGui
    dpg.create_context()

    with dpg.handler_registry():
        dpg.add_key_down_handler(key=dpg.mvKey_Escape, callback=lambda: dpg.stop_dearpygui())

    storage = SpectrogramStorage(config.DB_FILE)

    with dpg.window(label="Audio Player"):
        records = storage.fetch_ids_and_paths()
        create_table(records)

    dpg.create_viewport(title='Audio Player', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
