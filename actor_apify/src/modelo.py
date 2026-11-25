"""
Modelo del capstone: probabilidad de que un comercio tenga web propia.

Se usa para priorizar los leads: entre los comercios SIN web propia, los que el modelo
ve parecidos a los que sí tienen web son los que más chances tienen de comprar una.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "tiene_web_propia"
SEMILLA = 42


def preparar_X(final: pd.DataFrame) -> pd.DataFrame:
    """
    Input: tabla final (10 columnas numéricas).
    Output: matriz de features para el modelo.
    Proceso: quita el target, aplica log(1 + x) a las reseñas y pasa rubro y ciudad a variables dummy.
    """
    X = final.drop(columns=[TARGET]).copy()
    X["n_resenas"] = np.log1p(X["n_resenas"])
    X = pd.get_dummies(X, columns=["rubro_cod", "ciudad_cod"], prefix=["rubro", "ciudad"], dtype=float)
    return X


def puntuar(final: pd.DataFrame) -> tuple[pd.Series, dict]:
    """
    Input: tabla final.
    Output: (probabilidad de tener web propia por comercio, métricas de validación cruzada).
    Proceso: regresión logística con clases balanceadas; validación cruzada estratificada de 5
             particiones; la probabilidad de cada comercio sale de la partición donde no se entrenó.
    """
    X, y = preparar_X(final), final[TARGET].astype(int)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    res = cross_validate(modelo, X, y, cv=cv, scoring=["roc_auc", "recall", "precision", "f1"])
    metricas = {k.replace("test_", ""): round(float(v.mean()), 3) for k, v in res.items() if k.startswith("test_")}
    metricas["modelo"] = "Regresión logística (class_weight balanced)"
    proba = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    return pd.Series(proba, index=final.index, name="prob_web"), metricas


def priorizar(lista: pd.DataFrame) -> pd.DataFrame:
    """
    Input: lista de leads con la columna prob_web.
    Output: la misma lista con 'prioridad' (alta / media / baja, por tercios) y ordenada para contactar.
    Proceso: divide los leads en tres tercios según prob_web y ordena por lead caliente y probabilidad.
    """
    out = lista.copy()
    out["prob_web"] = out["prob_web"].round(3)
    out["prioridad"] = pd.qcut(out["prob_web"].rank(method="first"), 3, labels=["baja", "media", "alta"]).astype(str)
    return out.sort_values(["lead_caliente", "prob_web"], ascending=False).reset_index(drop=True)
