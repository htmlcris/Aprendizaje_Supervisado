import numpy as np

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate
)

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
#
# El dataset ya fue dividido previamente:
#
# 80 % -> Train
# 20 % -> Test
#
# IMPORTANTE:
# El conjunto Test permanecerá intacto hasta seleccionar
# definitivamente el mejor modelo.

(
    X_train,
    X_test,
    y_train,
    y_test
) = get_data()


# ============================================================
# 2. CONFIGURAR VALIDACION CRUZADA
# ============================================================
#
# Se divide TRAIN en cinco partes (folds).
#
# En cada iteracion:
#
# 4 folds -> entrenamiento
# 1 fold  -> validacion
#
# shuffle=True mezcla los datos antes de crear los folds.
#
# random_state=42 permite reproducir exactamente
# la misma validacion.

cv_strategy = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 3. METRICAS DE VALIDACION
# ============================================================
#
# RMSE sera nuestra metrica PRINCIPAL.
#
# MAE y R2 se conservan como metricas complementarias.

scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}


# ============================================================
# 4. REGRESION LINEAL
# ============================================================
#
# LinearRegression se utiliza como modelo base.
#
# No necesita una busqueda de hiperparametros de complejidad
# como Decision Tree o Random Forest.

linear_pipeline = Pipeline([
    (
        "preprocessor",
        clone(preprocessor)
    ),
    (
        "model",
        LinearRegression()
    )
])


# ============================================================
# 5. VALIDAR REGRESION LINEAL
# ============================================================

linear_cv = cross_validate(
    linear_pipeline,
    X_train,
    y_train,
    cv=cv_strategy,
    scoring=scoring,
    return_train_score=True
)


# Scikit-Learn devuelve RMSE y MAE negativos
# porque internamente intenta maximizar las metricas.
# Multiplicamos por -1 para recuperar el error real.

linear_rmse_cv = (
    -linear_cv["test_rmse"].mean()
)

linear_rmse_std = (
    linear_cv["test_rmse"].std()
)

linear_mae_cv = (
    -linear_cv["test_mae"].mean()
)

linear_r2_cv = (
    linear_cv["test_r2"].mean()
)


# ============================================================
# 6. DECISION TREE
# ============================================================
#
# El arbol posee hiperparametros que controlan
# directamente su complejidad.
#
# En lugar de elegir manualmente:
#
# max_depth = 6
# min_samples_leaf = 5
#
# dejaremos que GridSearchCV pruebe diferentes opciones.

decision_tree_pipeline = Pipeline([
    (
        "preprocessor",
        clone(preprocessor)
    ),
    (
        "model",
        DecisionTreeRegressor(
            random_state=42
        )
    )
])


# ============================================================
# 7. HIPERPARAMETROS DEL DECISION TREE
# ============================================================
#
# max_depth:
# profundidad maxima permitida.
#
# min_samples_split:
# cantidad minima necesaria para dividir un nodo.
#
# min_samples_leaf:
# cantidad minima de registros que debe contener una hoja.

tree_param_grid = {

    "model__max_depth": [
        3,
        5,
        7,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        5
    ]
}


# ============================================================
# 8. GRID SEARCH DEL DECISION TREE
# ============================================================
#
# Se prueban todas las combinaciones.
#
# 4 profundidades
# x 3 min_samples_split
# x 3 min_samples_leaf
#
# = 36 configuraciones.
#
# Cada configuracion se evalua con 5 folds.
#
# 36 x 5 = 180 entrenamientos aproximadamente.

tree_grid = GridSearchCV(
    estimator=decision_tree_pipeline,
    param_grid=tree_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv_strategy,
    refit=True,
    return_train_score=True,
    n_jobs=-1
)

tree_grid.fit(
    X_train,
    y_train
)


# Mejor modelo encontrado automáticamente.

best_tree = tree_grid.best_estimator_


# Mejor RMSE de validacion.

tree_rmse_cv = (
    -tree_grid.best_score_
)


# Obtener la posicion de la mejor configuracion.

tree_best_index = tree_grid.best_index_


# Desviacion estandar del mejor RMSE.

tree_rmse_std = (
    tree_grid.cv_results_[
        "std_test_score"
    ][tree_best_index]
)


# ============================================================
# 9. OTRAS METRICAS DEL MEJOR DECISION TREE
# ============================================================

tree_cv = cross_validate(
    best_tree,
    X_train,
    y_train,
    cv=cv_strategy,
    scoring=scoring
)

tree_mae_cv = (
    -tree_cv["test_mae"].mean()
)

tree_r2_cv = (
    tree_cv["test_r2"].mean()
)


# ============================================================
# 10. RANDOM FOREST
# ============================================================
#
# Random Forest construye multiples arboles
# y combina sus predicciones.

random_forest_pipeline = Pipeline([
    (
        "preprocessor",
        clone(preprocessor)
    ),
    (
        "model",
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    )
])


# ============================================================
# 11. HIPERPARAMETROS DEL RANDOM FOREST
# ============================================================
#
# n_estimators:
# cantidad de arboles.
#
# max_depth:
# profundidad maxima de cada arbol.
#
# min_samples_leaf:
# numero minimo de registros por hoja.
#
# max_features:
# numero/proporcion de variables consideradas
# cuando cada arbol busca una division.

forest_param_grid = {

    "model__n_estimators": [
        100,
        300
    ],

    "model__max_depth": [
        None,
        8
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "model__max_features": [
        "sqrt",
        1.0
    ]
}


# ============================================================
# 12. GRID SEARCH RANDOM FOREST
# ============================================================
#
# 2 n_estimators
# x 2 max_depth
# x 3 min_samples_leaf
# x 2 max_features
#
# = 24 configuraciones.
#
# Cada configuracion utiliza 5 folds:
#
# 24 x 5 = 120 entrenamientos.

forest_grid = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=forest_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv_strategy,
    refit=True,
    return_train_score=True,
    n_jobs=-1
)

forest_grid.fit(
    X_train,
    y_train
)


best_forest = (
    forest_grid.best_estimator_
)


forest_rmse_cv = (
    -forest_grid.best_score_
)


forest_best_index = (
    forest_grid.best_index_
)


forest_rmse_std = (
    forest_grid.cv_results_[
        "std_test_score"
    ][forest_best_index]
)


# ============================================================
# 13. OTRAS METRICAS DEL MEJOR RANDOM FOREST
# ============================================================

forest_cv = cross_validate(
    best_forest,
    X_train,
    y_train,
    cv=cv_strategy,
    scoring=scoring
)

forest_mae_cv = (
    -forest_cv["test_mae"].mean()
)

forest_r2_cv = (
    forest_cv["test_r2"].mean()
)


# ============================================================
# 14. GUARDAR RESULTADOS DE VALIDACION
# ============================================================
#
# IMPORTANTE:
#
# Todavia NO hemos utilizado X_test para seleccionar
# ninguno de los tres modelos.

results = {

    "Linear Regression": {

        "model": linear_pipeline,

        "RMSE CV": linear_rmse_cv,

        "RMSE STD": linear_rmse_std,

        "MAE CV": linear_mae_cv,

        "R2 CV": linear_r2_cv
    },


    "Decision Tree": {

        "model": best_tree,

        "RMSE CV": tree_rmse_cv,

        "RMSE STD": tree_rmse_std,

        "MAE CV": tree_mae_cv,

        "R2 CV": tree_r2_cv
    },


    "Random Forest": {

        "model": best_forest,

        "RMSE CV": forest_rmse_cv,

        "RMSE STD": forest_rmse_std,

        "MAE CV": forest_mae_cv,

        "R2 CV": forest_r2_cv
    }
}


# ============================================================
# 15. MOSTRAR HIPERPARAMETROS ENCONTRADOS
# ============================================================

print(
    "\n========================================"
)

print(
    "MEJORES HIPERPARAMETROS"
)

print(
    "========================================"
)


print(
    "\nDecision Tree:"
)

print(
    tree_grid.best_params_
)


print(
    "\nRandom Forest:"
)

print(
    forest_grid.best_params_
)


# ============================================================
# 16. COMPARACION MEDIANTE VALIDACION CRUZADA
# ============================================================

print(
    "\n========================================"
)

print(
    "COMPARACION DE MODELOS - CROSS VALIDATION"
)

print(
    "========================================"
)


for name, result in results.items():

    print(
        f"\n{name}"
    )

    print(
        "RMSE CV:",
        round(
            result["RMSE CV"],
            2
        )
    )

    print(
        "Desviacion RMSE:",
        round(
            result["RMSE STD"],
            2
        )
    )

    print(
        "MAE CV:",
        round(
            result["MAE CV"],
            2
        )
    )

    print(
        "R2 CV:",
        round(
            result["R2 CV"],
            4
        )
    )


# ============================================================
# 17. SELECCIONAR MEJOR MODELO
# ============================================================
#
# RMSE es un ERROR.
#
# Por eso:
#
# MENOR RMSE = MEJOR MODELO.

best_model_name = min(
    results,
    key=lambda name: results[name]["RMSE CV"]
)


best_model = (
    results[best_model_name]["model"]
)


print(
    "\n========================================"
)

print(
    "MEJOR MODELO SEGUN RMSE CV"
)

print(
    "========================================"
)

print(
    best_model_name
)

print(
    "RMSE CV:",
    round(
        results[
            best_model_name
        ]["RMSE CV"],
        2
    )
)


# ============================================================
# 18. ENTRENAR MODELO FINAL
# ============================================================
#
# Una vez seleccionado mediante Cross Validation,
# se vuelve a entrenar usando TODO el 80 % de Train.

best_model.fit(
    X_train,
    y_train
)


# ============================================================
# 19. EVALUACION FINAL EN TEST
# ============================================================
#
# ESTA ES LA PRIMERA VEZ que utilizamos el 20 % Test
# para tomar una medida final del modelo seleccionado.

test_predictions = best_model.predict(
    X_test
)


# RMSE

final_mse = mean_squared_error(
    y_test,
    test_predictions
)

final_rmse = np.sqrt(
    final_mse
)


# MAE

final_mae = mean_absolute_error(
    y_test,
    test_predictions
)


# R2

final_r2 = r2_score(
    y_test,
    test_predictions
)


# ============================================================
# 20. RESULTADOS FINALES
# ============================================================

print(
    "\n========================================"
)

print(
    "EVALUACION FINAL EN TEST"
)

print(
    "========================================"
)

print(
    "Modelo:",
    best_model_name
)

print(
    "RMSE Test:",
    round(
        final_rmse,
        2
    )
)

print(
    "MAE Test:",
    round(
        final_mae,
        2
    )
)

print(
    "R2 Test:",
    round(
        final_r2,
        4
    )
)