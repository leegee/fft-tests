    python -m venv myenv 
    source venv/Scripts/activate
    source .env
    python src/my-fft.py


Processes WAV by FFT through a Mel Filterbank ready for feature detection.

Saves spectragrams as binary and PNG, and into SQL.

SQL will be extended to link to ontology and possibily to add free tags that could be used to update the ontology.

## Ontoology wip:

    Sample
        Hits
            drums
                snare
                kick
            strings
                stab
        Plucked
            pizz
            stf
        Sustained
            bowed

## Environment

    LEE_DB_FILE             ffts.sqlite3
    LEE_TABLE_SEPECTROGRAMS    mel_spectrograms
