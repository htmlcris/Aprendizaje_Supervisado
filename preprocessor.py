from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer

from data_loader import load_bike_data


# ============================================================
# 1. TRANSFORMADOR PERSONALIZADO
# ============================================================

class BikeAttributesAdder(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):

        # No necesita aprender parámetros.
        return self


    def transform(self, X):

        # Realizamos una copia para evitar
        # modificar los datos originales.
        X = X.copy()


        # ----------------------------------------------------
        # TEMPERATURA AL CUADRADO
        # ----------------------------------------------------
        #
        # Permite representar una posible relación
        # no completamente lineal entre temperatura
        # y cantidad de alquileres.

        X["temp_squared"] = (
            X["temp"] ** 2
        )


        # ----------------------------------------------------
        # TEMPERATURA Y HUMEDAD
        # ----------------------------------------------------
        #
        # Combina temperatura y humedad.
        #
        # Una humedad elevada reduce el valor
        # de esta característica.

        X["temp_humidity"] = (
            X["temp"]
            *
            (1 - X["hum"])
        )


        return X


# ============================================================
# 2. CARGAR Y LIMPIAR DATASET
# ============================================================

def load_clean_data():

    bikes = load_bike_data()

    # Eliminar registros completamente repetidos.
    bikes = bikes.drop_duplicates()

    return bikes


# ============================================================
# 3. SELECCIONAR VARIABLES
# ============================================================

def create_features_target(bikes):

    # --------------------------------------------------------
    # VARIABLES UTILIZADAS
    # --------------------------------------------------------
    #
    # Numéricas:
    # temp
    # hum
    # windspeed
    #
    # Binaria:
    # yr
    #
    # Categóricas:
    # season
    # mnth
    # weathersit

    X = bikes[
        [
            "temp",
            "hum",
            "windspeed",
            "yr",
            "season",
            "mnth",
            "weathersit"
        ]
    ].copy()


    # Variable que queremos predecir.
    y = bikes["cnt"].copy()

    return X, y


# ============================================================
# 4. TRAIN / TEST
# ============================================================

def split_data(X, y):

    return train_test_split(
        X,
        y,

        # 20% queda reservado exclusivamente
        # para evaluación final.
        test_size=0.20,

        # Permite reproducir la misma división.
        random_state=42
    )


# ============================================================
# 5. VARIABLES NUMERICAS
# ============================================================

numeric_features = [
    "temp",
    "hum",
    "windspeed"
]


# ============================================================
# 6. VARIABLE BINARIA
# ============================================================

# yr ya está correctamente representado como:
#
# 0 = 2011
# 1 = 2012
#
# No necesita OneHotEncoder.

binary_features = [
    "yr"
]


# ============================================================
# 7. VARIABLES CATEGORICAS
# ============================================================

categorical_features = [
    "season",
    "mnth",
    "weathersit"
]


# ============================================================
# 8. PIPELINE NUMERICO
# ============================================================

numeric_pipeline = Pipeline([

    # --------------------------------------------------------
    # CREAR ATRIBUTOS
    # --------------------------------------------------------

    (
        "attributes_adder",
        BikeAttributesAdder()
    ),


    # --------------------------------------------------------
    # VALORES FALTANTES
    # --------------------------------------------------------

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),


    # --------------------------------------------------------
    # STANDARD SCALER
    # --------------------------------------------------------
    #
    # Estandariza las variables numéricas
    # aproximadamente con:
    #
    # media = 0
    # desviación estándar = 1

    (
        "scaler",
        StandardScaler()
    )
])


# ============================================================
# 9. PIPELINE CATEGORICO
# ============================================================

categorical_pipeline = Pipeline([

    # --------------------------------------------------------
    # IMPUTACION
    # --------------------------------------------------------

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),


    # --------------------------------------------------------
    # ONE HOT ENCODER
    # --------------------------------------------------------
    #
    # season, mnth y weathersit contienen números,
    # pero esos números representan categorías.
    #
    # OneHotEncoder evita que el modelo lineal
    # interprete esos números como cantidades.

    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        )
    )
])


# ============================================================
# 10. COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer([

    # Variables numéricas
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),


    # Año 0 / 1.
    # Se mantiene sin modificación.
    (
        "binary",
        "passthrough",
        binary_features
    ),


    # Variables categóricas
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 11. OBTENER DATOS
# ============================================================

def get_data():

    bikes = load_clean_data()

    X, y = create_features_target(
        bikes
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )