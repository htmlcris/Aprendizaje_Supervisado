import os
import zipfile
import urllib.request

import pandas as pd


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BIKE_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "bike_sharing"
)

CSV_PATH = os.path.join(
    BIKE_PATH,
    "day.csv"
)

BIKE_URL = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike%2Bsharing%2Bdataset.zip"
)


# ============================================================
# DESCARGAR DATASET
# ============================================================

def fetch_bike_data():

    # Crear la carpeta si no existe.
    os.makedirs(
        BIKE_PATH,
        exist_ok=True
    )

    zip_path = os.path.join(
        BIKE_PATH,
        "bike_sharing.zip"
    )

    print("Descargando dataset...")

    urllib.request.urlretrieve(
        BIKE_URL,
        zip_path
    )

    print("Extrayendo dataset...")

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_file:

        zip_file.extractall(
            BIKE_PATH
        )

    print("Dataset preparado correctamente.")


# ============================================================
# CARGAR DATASET
# ============================================================

def load_bike_data():

    # Si day.csv todavía no existe,
    # se descarga automáticamente.
    if not os.path.exists(CSV_PATH):
        fetch_bike_data()

    return pd.read_csv(
        CSV_PATH
    )