"""Genera los archivos del tablero (EDA, inferencia, selección de features y modelo) desde data/raw."""
import json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "actor_apify/src")
import etl

SEMILLA = 42
OUT = Path("data/processed/dashboard"); OUT.mkdir(parents=True, exist_ok=True)
CODIGO = Path(__file__).read_text(encoding="utf-8")
META = [{"label": "export_dashboard.py", "value": CODIGO, "type": "code", "lang": "python"},
        {"label": "Input", "value": "data/raw/maps_concordia.json, data/raw/maps_parana.json (Apify kaix/google-maps-places-scraper, 24/11/2026)"}]

crudo = pd.concat([etl.extraer(json.load(open(f"data/raw/maps_{n}.json", encoding="utf-8")), c)
                   for n, c in [("concordia", "Concordia"), ("parana", "Paraná")]], ignore_index=True)
limpio, _ = etl.transformar(crudo)
final = etl.tabla_final(limpio)
RUBRO = {"peluqueria": "Peluquería", "ferreteria": "Ferretería", "almacen": "Almacén", "taller mecanico": "Taller mecánico",
         "panaderia": "Panadería", "veterinaria": "Veterinaria", "gimnasio": "Gimnasio", "carniceria": "Carnicería",
         "restaurante": "Restaurante", "tienda de ropa": "Tienda de ropa", "verduleria": "Verdulería", "farmacia": "Farmacia"}
PRES = {"sin_web": "Sin web", "solo_redes": "Solo redes", "plataforma_terceros": "Plataforma de terceros", "web_propia": "Web propia"}

# 1) Comercios (sin nombre, teléfono ni dirección): base del EDA
eda = pd.DataFrame({
    "rubro": limpio["rubro"].map(RUBRO), "ciudad": limpio["ciudad"].map({"concordia": "Concordia", "parana": "Paraná"}),
    "presencia": limpio["presencia"].map(PRES), "tiene_web_propia": final["tiene_web_propia"].astype(int),
    "lead_caliente": limpio["lead_caliente"], "rating": final["rating"].round(2), "n_resenas": final["n_resenas"].astype(int),
    "n_fotos": final["n_fotos"].astype(int), "horas_semana": final["horas_semana"].round(1),
    "dias_abiertos": final["dias_abiertos"].astype(int), "tiene_telefono": final["tiene_telefono"].astype(int),
    "abre_domingo": final["abre_domingo"].astype(int)})
eda.to_csv(OUT / "comercios_eda.csv", index=False)

# 2) Inferencia
filas = []
for f in ["rubro", "ciudad"]:
    t = pd.crosstab(limpio[f], limpio["tiene_web_propia"])
    chi2, p, gl, _ = stats.chi2_contingency(t)
    v = np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))
    filas.append({"pregunta": f"¿La falta de web depende de la {('ciudad' if f == 'ciudad' else 'categoría (rubro)')}?",
                  "prueba": "Chi-cuadrado", "estadistico": round(chi2, 1), "p_valor": float(f"{p:.3g}"),
                  "efecto": f"V de Cramér = {v:.2f}", "conclusion": "Sí" if p < 0.05 else "No"})
g = limpio.assign(lr=np.log1p(limpio["n_resenas"]))
kw = stats.kruskal(*[x["lr"] for _, x in g.groupby("presencia")])
n, k = len(g), g["presencia"].nunique()
filas.append({"pregunta": "¿Los comercios sin web tienen menos reseñas?", "prueba": "Kruskal-Wallis (falla normalidad del ANOVA)",
              "estadistico": round(kw.statistic, 1), "p_valor": float(f"{kw.pvalue:.3g}"),
              "efecto": f"ε² = {(kw.statistic - k + 1) / (n - k):.3f}", "conclusion": "Sí" if kw.pvalue < 0.05 else "No"})
json.dump({"rows": filas, "meta": META}, open(OUT / "inferencia.json", "w", encoding="utf-8"), ensure_ascii=False)

# 3) Selección de features y modelo
X = final.drop(columns=["tiene_web_propia"]).assign(n_resenas=lambda d: np.log1p(d["n_resenas"]))
X = pd.get_dummies(X.astype({"rubro_cod": int, "ciudad_cod": int}), columns=["rubro_cod", "ciudad_cod"], drop_first=True, dtype=float)
y = final["tiene_web_propia"].astype(int)
base = ["rating", "n_resenas", "n_fotos", "horas_semana", "dias_abiertos", "tiene_telefono", "abre_domingo"]
mi = pd.Series(mutual_info_classif(X, y, random_state=SEMILLA), index=X.columns)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEMILLA).fit(Xtr, ytr)
pi = pd.Series(permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEMILLA).importances_mean, index=X.columns)
agrupa = lambda c: "rubro_cod" if c.startswith("rubro") else ("ciudad_cod" if c.startswith("ciudad") else c)
mi_g, pi_g = mi.groupby(agrupa).sum(), pi.groupby(agrupa).sum()
pb = {c: stats.pointbiserialr(y, X[c])[0] for c in base}
lineales = pd.Series(pb).abs().sort_values(ascending=False).head(5).index.tolist()
NOMBRE = {"rating": "Rating", "n_resenas": "Reseñas (log)", "n_fotos": "Fotos", "horas_semana": "Horas por semana",
          "dias_abiertos": "Días abiertos", "tiene_telefono": "Tiene teléfono", "abre_domingo": "Abre domingo",
          "rubro_cod": "Rubro", "ciudad_cod": "Ciudad"}
feats = [{"feature": NOMBRE[c], "tipo": "lineal" if c in lineales else "no lineal",
          "correlacion_puntobiserial": round(pb[c], 3) if c in pb else None,
          "informacion_mutua": round(float(mi_g[c]), 4), "importancia_rf": round(float(pi_g[c]), 4)}
         for c in pi_g.sort_values(ascending=False).index]
json.dump({"rows": feats, "meta": META}, open(OUT / "features.json", "w", encoding="utf-8"), ensure_ascii=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
modelos = {"Línea base (estratificada)": DummyClassifier(strategy="stratified", random_state=SEMILLA),
           "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
           "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEMILLA)}
mods = []
for nombre, m in modelos.items():
    r = cross_validate(m, X, y, cv=cv, scoring=["roc_auc", "f1", "recall", "precision"])
    mods.append({"modelo": nombre, **{k.replace("test_", ""): round(float(v.mean()), 3) for k, v in r.items() if k.startswith("test_")}})
json.dump({"rows": mods, "meta": META}, open(OUT / "modelo.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(eda), filas, feats[:3], mods, sep="\n")
