import numpy as np

from sklearn.base import clone

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessor import (
    preprocessor,
    get_data
)


# ============================================================
# 1. OBTENER TRAIN Y TEST
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test
) = get_data()


# ============================================================
# 2. INSTANCIAR LOS MODELOS
# ============================================================

# ------------------------------------------------------------
# REGRESION LINEAL
# ------------------------------------------------------------

linear_regression = LinearRegression()


# ------------------------------------------------------------
# DECISION TREE
# ------------------------------------------------------------
#
# Limitamos ligeramente el árbol para reducir
# el riesgo de sobreajuste.

decision_tree = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 3. CREAR PIPELINE PARA CADA MODELO
# ============================================================

linear_pipeline = Pipeline([
    (
        "preprocessor",
        clone(preprocessor)
    ),
    (
        "model",
        linear_regression
    )
])


decision_tree_pipeline = Pipeline([
    (
        "preprocessor",
        clone(preprocessor)
    ),
    (
        "model",
        decision_tree
    )
])


random_forest_pipeline = Pipeline([
    (
        "preprocessor",
        clone(preprocessor)
    ),
    (
        "model",
        random_forest
    )
])


# ============================================================
# 4. FUNCION PARA ENTRENAR Y EVALUAR
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    # --------------------------------------------------------
    # VALIDACION CRUZADA
    # --------------------------------------------------------
    #
    # Se utilizan únicamente los datos de TRAIN.
    #
    # Esto nos permite obtener una medida más estable
    # de R2 antes de tocar el conjunto final de Test.

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )


    # --------------------------------------------------------
    # ENTRENAMIENTO FINAL
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICCIONES
    # --------------------------------------------------------

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # R2 TRAIN
    # --------------------------------------------------------

    train_r2 = r2_score(
        y_train,
        train_predictions
    )


    # --------------------------------------------------------
    # R2 TEST
    # --------------------------------------------------------

    test_r2 = r2_score(
        y_test,
        test_predictions
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    mse = mean_squared_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(mse)


    # --------------------------------------------------------
    # R2 PROMEDIO DE VALIDACION CRUZADA
    # --------------------------------------------------------

    cv_r2 = cv_scores.mean()


    # --------------------------------------------------------
    # DESVIACION DEL R2
    # --------------------------------------------------------

    cv_std = cv_scores.std()


    # --------------------------------------------------------
    # MOSTRAR RESULTADOS
    # --------------------------------------------------------

    print("\n========================================")
    print(name)
    print("========================================")

    print(
        "R2 Train:",
        round(train_r2, 4)
    )

    print(
        "R2 Test:",
        round(test_r2, 4)
    )

    print(
        "R2 CV promedio:",
        round(cv_r2, 4)
    )

    print(
        "Desviacion R2 CV:",
        round(cv_std, 4)
    )

    print(
        "MAE:",
        round(mae, 2)
    )

    print(
        "RMSE:",
        round(rmse, 2)
    )


    return {
        "R2 Train": train_r2,
        "R2 Test": test_r2,
        "R2 CV": cv_r2,
        "R2 CV STD": cv_std,
        "MAE": mae,
        "RMSE": rmse
    }


# ============================================================
# 5. EVALUAR LINEAR REGRESSION
# ============================================================

linear_results = evaluate_model(
    "LINEAR REGRESSION",
    linear_pipeline,
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# 6. EVALUAR DECISION TREE
# ============================================================

tree_results = evaluate_model(
    "DECISION TREE REGRESSION",
    decision_tree_pipeline,
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# 7. EVALUAR RANDOM FOREST
# ============================================================

forest_results = evaluate_model(
    "RANDOM FOREST REGRESSION",
    random_forest_pipeline,
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# 8. COMPARAR LOS MODELOS
# ============================================================

results = {
    "Linear Regression": linear_results,
    "Decision Tree": tree_results,
    "Random Forest": forest_results
}


# ============================================================
# 9. MEJOR MODELO SEGUN R2
# ============================================================
#
# Para seleccionar el modelo utilizamos principalmente
# el R2 de validación cruzada.
#
# Esto evita elegir un modelo únicamente porque tuvo
# suerte con una división particular de Test.

best_model = max(
    results,
    key=lambda model: results[model]["R2 CV"]
)


print("\n========================================")
print("MEJOR MODELO SEGUN R2")
print("========================================")

print(best_model)

print(
    "R2 CV:",
    round(
        results[best_model]["R2 CV"],
        4
    )
)

print(
    "R2 Test:",
    round(
        results[best_model]["R2 Test"],
        4
    )
)