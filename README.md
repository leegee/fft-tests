    python -m venv myenv 
    source venv/Scripts/activate
    vi .env
    source .env
    python -m build
    
    python src/scripts/ingest.py samples/
    python src/scripts/cluster.py # no-op atm
    python src/scripts/find.py samples/Lo-fi/snare/snare1.wav

A few functional Python modules to catalogue and search WAV files, by FFT/Mel Filterbank/DBSCAN.

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
