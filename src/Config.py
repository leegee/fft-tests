import os
from dotenv import load_dotenv

# Define a standard prefix for environment variables
ENV_PREFIX = 'LEE_'

# Default configuration values
DEFAULT_CONFIG = {
    'FFT_WINDOW_SIZE': 256,
    'FFT_STEP_SIZE': 512,
    'FFT_N_FILTERS': 24,
    'DB_FILE': 'ffts.sqlite3',
    'NUM_MATCHES': 5,
    'TABLE_SEPECTROGRAMS': 'mel_sepectrograms',
    'DBSCAN_MIN_SAMPLES': 5,
    'DBSCAN_EPS': 0.5,
    'PLOT_SIZE': (100,100)
}

class Config:
    def __init__(self):
        # Load environment variables from a .env file, if it exists
        load_dotenv()  

        # Load default values and override with environment variables if available
        for key in DEFAULT_CONFIG:
            env_var = f"{ENV_PREFIX}{key}"
            setattr(self, key, self._get_env_var(env_var, DEFAULT_CONFIG[key]))

    @staticmethod
    def _get_env_var(env_var, default):
        """Get the environment variable value or return the default."""
        value = os.getenv(env_var)
        if value is not None:
            # Try to cast to integer if it's a number
            try:
                return int(value)
            except ValueError:
                return value
        return default

    def __repr__(self):
        return f"Config({', '.join(f'{k}={v}' for k, v in DEFAULT_CONFIG.items())})"

# Instantiate the config object
config = Config()
