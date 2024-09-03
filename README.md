    python -m venv myenv 
    source venv/Scripts/activate
    source .env
    
    python src/scripts/ingest.py samples/
    python src/scripts/cluster.py
    python src/scripts/find.py samples/Lo-fi/snare/snare1.wav

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

Source `.env`.
