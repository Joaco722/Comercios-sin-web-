"""Genera colabs/analytics/capstone_comercios_sin_web.ipynb (ejecutar desde la raíz del repo)."""
import nbformat as nbf
from pathlib import Path

ETL = Path("actor_apify/src/etl.py").read_text(encoding="utf-8")
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Buscador de comercios sin web · Capstone SIC (prueba de concepto)

**Pregunta de negocio:** ¿qué comercios de Concordia y Paraná (Entre Ríos) no tienen web propia y a cuáles conviene ofrecerles una primero?

**Cliente:** una agencia o freelancer que vende sitios web a comercios locales.
**Fuente:** Google Maps, extraído con Apify (`kaix/google-maps-places-scraper`), 12 rubros × 2 ciudades.

| Sección | Qué responde |
|---|---|
| 0 | Configuración y carga de datos |
| 1 | ETL: de la respuesta de Google Maps a una tabla limpia (embudo + validaciones) |
| 2 | EDA: distribuciones, frecuencias y correlaciones |
| 3 | Inferencia: ¿la falta de web depende del rubro / ciudad? ¿los que no tienen web tienen menos movimiento? |
| 4 | Selección de features (5 lineales + 5 no lineales) |
| 5 | Modelo: probabilidad de tener web propia y puntaje de prioridad para los leads |
| 6 | Métrica de negocio y KPI |
| 7 | Exportar resultados |

Cada función del ETL documenta **Input / Output / Proceso** para traducirla a un módulo de Make, un nodo de n8n o al Actor de Apify del repo (`actor_apify/src/etl.py`).
""")

md("## 0. Configuración")
code("""
import json, os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEMILLA = 42
ALFA = 0.05

# --- Rutas: en Colab se monta Drive; en local se usan las carpetas del repo
RUTA_DRIVE = "/content/drive/MyDrive/Capstone/comercios_sin_web/datos"      # <- CAMBIAR si la carpeta es otra
SALIDA_DRIVE = "/content/drive/MyDrive/Capstone/comercios_sin_web/salida"
RUTA_LOCAL, SALIDA_LOCAL = "../../data/raw", "../../data/processed"
ARCHIVOS = {"Concordia": "maps_concordia.json", "Paraná": "maps_parana.json"}
try:
    from google.colab import drive
    drive.mount("/content/drive")
    EN_COLAB = True
except ImportError:
    EN_COLAB = False
BASE = RUTA_DRIVE if EN_COLAB else RUTA_LOCAL
DIR_SALIDA = SALIDA_DRIVE if EN_COLAB else SALIDA_LOCAL
faltan = [f for f in ARCHIVOS.values() if not os.path.exists(os.path.join(BASE, f))]
assert not faltan, f"No se encuentran {faltan} en {BASE}"
print("Entorno:", "Colab + Drive" if EN_COLAB else "local", "| datos:", BASE)
""")

md("""
## 1. ETL
Las funciones son las mismas que usa el Actor de Apify (`actor_apify/src/etl.py`), copiadas acá para que el notebook sea autocontenido.

**Reglas de negocio documentadas**
- *Web propia* = el campo `website` de Google Maps apunta a un dominio propio. Si apunta a Instagram, Facebook, WhatsApp o Linktree es **solo redes**; si apunta a PedidosYa, Mercado Libre, Tiendanube o sitios gratuitos de Google es **plataforma de terceros**. Ninguno de los dos cuenta como web propia: los tres casos son *leads*.
- *Lead caliente* = sin web propia, con teléfono, rating ≥ 4 y al menos 10 reseñas (negocio activo y con reputación que cuidar).
- Se descartan lugares fuera de Argentina (la búsqueda desde Concordia alcanza Salto, Uruguay), cerrados y duplicados.
""")
code(ETL.replace('from __future__ import annotations\n', ''))
code("""
crudo = pd.concat([extraer(json.load(open(os.path.join(BASE, f), encoding="utf-8")), ciudad)
                   for ciudad, f in ARCHIVOS.items()], ignore_index=True)
print(f"Filas crudas: {len(crudo):,} | columnas: {crudo.shape[1]}")
crudo.head()
""")
code("""
limpio, descartes = transformar(crudo)
embudo = pd.DataFrame({"paso": ["Respuesta de Google Maps"] + [f"− {m}" for m in descartes["motivo_descarte"].value_counts().index] + ["Comercios limpios"],
                       "filas": [len(crudo)] + list(-descartes["motivo_descarte"].value_counts().values) + [len(limpio)]})
display(embudo)
print("Presencia web:"); display(limpio["presencia"].value_counts().to_frame("comercios"))
""")
code("""
final = tabla_final(limpio)
validar(final)          # contrato del dato: 700–2000 filas, ≤10 columnas, rangos y target binario
print("✅ Validaciones OK →", final.shape)
final.describe().round(2).T
""")

md("## 2. EDA\n### 2.1 Frecuencias: ¿cuántos comercios no tienen web propia?")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
orden = limpio["presencia"].value_counts().index
sns.countplot(data=limpio, y="presencia", order=orden, ax=axes[0], palette="Set2")
axes[0].set_title("Presencia web (Google Maps)")
pct = (1 - limpio.groupby("rubro")["tiene_web_propia"].mean()).sort_values() * 100
pct.plot.barh(ax=axes[1], color="#E8590C")
axes[1].set_title("% de comercios SIN web propia por rubro"); axes[1].set_xlabel("%")
for i, v in enumerate(pct): axes[1].text(v + 0.5, i, f"{v:.0f} %", va="center", fontsize=8)
plt.tight_layout(); plt.show()
print(f"Sin web propia: {100 - limpio['tiene_web_propia'].mean()*100:.1f} % de {len(limpio):,} comercios")
display(pd.crosstab(limpio["ciudad"], limpio["presencia"], normalize="index").round(3) * 100)
""")
md("### 2.2 Distribuciones")
code("""
num = ["rating", "n_resenas", "n_fotos", "horas_semana", "dias_abiertos"]
fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
for ax, c in zip(axes, num):
    x = np.log1p(final[c]) if c == "n_resenas" else final[c]
    sns.histplot(x=x, hue=final["tiene_web_propia"].map({0: "sin web", 1: "con web"}), ax=ax, kde=True, element="step", stat="density", common_norm=False)
    ax.set_title(("log(1+reseñas)" if c == "n_resenas" else c))
plt.tight_layout(); plt.show()
print("Asimetría:", final[num].skew().round(2).to_dict())
print("n_fotos tiene tope 10 en la respuesta del Actor (la búsqueda 'nearby' devuelve hasta 10 fotos).")
""")
md("### 2.3 Correlaciones")
code("""
corr = final.assign(log_resenas=np.log1p(final["n_resenas"])).drop(columns=["n_resenas"]).corr(method="spearman")
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
plt.title("Correlación de Spearman (rubro y ciudad son códigos: leer con cuidado)"); plt.show()
""")

md("""
## 3. Inferencia estadística
Siguiendo la guía *ANOVA patitos*: primero la pregunta, después la técnica.

| Pregunta | Variables | Técnica |
|---|---|---|
| ¿La falta de web depende del rubro? | categórica × categórica | Chi-cuadrado + V de Cramér (tamaño del efecto) |
| ¿Y de la ciudad? | categórica × categórica | Chi-cuadrado |
| ¿Los comercios sin web tienen menos movimiento (reseñas)? | numérica × 3+ grupos | ANOVA → supuestos → Plan B (Kruskal-Wallis) + comparaciones con Holm |
""")
code("""
def chi2_cramer(tabla):
    chi2, p, gl, _ = stats.chi2_contingency(tabla)
    v = np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1)))
    return chi2, p, gl, v
for factor in ["rubro", "ciudad"]:
    t = pd.crosstab(limpio[factor], limpio["tiene_web_propia"])
    chi2, p, gl, v = chi2_cramer(t)
    print(f"Presencia web ~ {factor}: χ² = {chi2:.1f} (gl {gl}) | p = {p:.2e} | V de Cramér = {v:.2f}")
print("Regla práctica V: 0,1 chico · 0,3 mediano · 0,5 grande")
""")
code("""
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from itertools import combinations
g = limpio.assign(log_resenas=np.log1p(limpio["n_resenas"]))
modelo = ols("log_resenas ~ C(presencia)", data=g).fit()
display(anova_lm(modelo, typ=2))
sh_p = stats.shapiro(modelo.resid.sample(500, random_state=SEMILLA)).pvalue
lev_p = stats.levene(*[x["log_resenas"] for _, x in g.groupby("presencia")], center="median").pvalue
print(f"Supuestos → Shapiro p = {sh_p:.2e} | Levene p = {lev_p:.2e}")
kw = stats.kruskal(*[x["log_resenas"] for _, x in g.groupby("presencia")])
n, k = len(g), g["presencia"].nunique()
eps2 = (kw.statistic - k + 1) / (n - k)
print(f"Plan B, Kruskal-Wallis: H = {kw.statistic:.1f} | p = {kw.pvalue:.2e} | ε² = {eps2:.3f}")
pares = []
for a, b in combinations(sorted(g["presencia"].unique()), 2):
    pa = stats.mannwhitneyu(g.loc[g.presencia == a, "n_resenas"], g.loc[g.presencia == b, "n_resenas"]).pvalue
    pares.append((a, b, g.loc[g.presencia == a, "n_resenas"].median(), g.loc[g.presencia == b, "n_resenas"].median(), pa))
pares = pd.DataFrame(pares, columns=["grupo_1", "grupo_2", "mediana_1", "mediana_2", "p"])
pares["p_holm"] = multipletests(pares["p"], method="holm")[1]
display(pares.round(4))
plt.figure(figsize=(8, 4)); sns.boxplot(data=g, x="presencia", y="log_resenas", palette="Set2")
plt.title("Reseñas (log) según presencia web"); plt.show()
""")

md("""
## 4. Selección de features (máx. 10: hasta 5 lineales + 5 no lineales)
- **Lineales:** correlación punto-biserial con el target y coeficientes de una regresión logística estandarizada.
- **No lineales:** información mutua y la importancia por permutación de un Random Forest.
""")
code("""
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, train_test_split

X = final.drop(columns=["tiene_web_propia"]).assign(n_resenas=lambda d: np.log1p(d["n_resenas"]))
X = pd.get_dummies(X.astype({"rubro_cod": int, "ciudad_cod": int}), columns=["rubro_cod", "ciudad_cod"], drop_first=True, dtype=float)
y = final["tiene_web_propia"].astype(int)
base = ["rating", "n_resenas", "n_fotos", "horas_semana", "dias_abiertos", "tiene_telefono", "abre_domingo"]
pb = {c: stats.pointbiserialr(y, X[c])[0] for c in base}
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X[base], y)
coef = dict(zip(base, logit[-1].coef_[0]))
mi = dict(zip(X.columns, mutual_info_classif(X, y, random_state=SEMILLA)))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEMILLA).fit(Xtr, ytr)
pi = permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEMILLA)
imp = dict(zip(X.columns, pi.importances_mean))
sel = pd.DataFrame({"r_puntobiserial": pd.Series(pb), "coef_logit_std": pd.Series(coef),
                    "info_mutua": pd.Series(mi), "importancia_RF": pd.Series(imp)}).round(3)
display(sel.sort_values("importancia_RF", ascending=False).head(15))
""")
code("""
lineales = pd.Series(pb).abs().sort_values(ascending=False).head(5).index.tolist()
no_lineales = (pd.Series(imp).groupby(lambda c: c.split("_")[0] + "_cod" if c.startswith(("rubro", "ciudad")) else c).sum()
               .sort_values(ascending=False).drop(lineales, errors="ignore").head(5).index.tolist())
print("5 features lineales   :", lineales)
print("Features no lineales  :", no_lineales)
print(f"Total: {len(lineales) + len(no_lineales)} features (las 9 disponibles; rubro y ciudad entran como variables dummy)")
""")

md("""
## 5. Modelo: ¿qué comercios *deberían* tener web y no la tienen?
Target: `tiene_web_propia` (15 % de positivos → clases desbalanceadas, se usa `class_weight="balanced"`).
Se compara contra una **línea base** (regla trivial) con validación cruzada estratificada de 5 particiones.

**Uso de negocio:** el modelo aprende el perfil de los comercios que *sí* tienen web. Entre los que **no** la tienen, los que tienen probabilidad alta se parecen a los que ya invirtieron en una → son los leads con más chances de comprar.
""")
code("""
from sklearn.dummy import DummyClassifier
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
modelos = {
    "Línea base (estratificada)": DummyClassifier(strategy="stratified", random_state=SEMILLA),
    "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEMILLA),
}
res = {n: cross_validate(m, X, y, cv=cv, scoring=["roc_auc", "f1", "recall", "precision"]) for n, m in modelos.items()}
tabla = pd.DataFrame({n: {k.replace("test_", ""): v.mean() for k, v in r.items() if k.startswith("test_")} for n, r in res.items()}).T.round(3)
display(tabla)
mejor = tabla["roc_auc"].idxmax(); print("Mejor modelo por ROC-AUC:", mejor)
""")
code("""
proba = cross_val_predict(modelos[mejor], X, y, cv=cv, method="predict_proba")[:, 1]
limpio["prob_web"] = proba
leads_ = limpio[limpio["es_lead"] == 1].copy()
leads_["prioridad"] = pd.qcut(leads_["prob_web"], 3, labels=["baja", "media", "alta"])
display(leads_.groupby("prioridad")[["rating", "n_resenas", "lead_caliente"]].agg(["mean"]).round(2))
plt.figure(figsize=(8, 4))
sns.histplot(data=limpio, x="prob_web", hue=limpio["tiene_web_propia"].map({0: "sin web", 1: "con web"}), bins=30, element="step", stat="density", common_norm=False)
plt.title("Probabilidad estimada de tener web propia"); plt.show()
""")

md("""
## 6. Métrica de negocio y KPI
- **Métrica de negocio:** % de comercios sin web propia por rubro y ciudad (= tamaño del mercado al que se le puede vender una web).
- **KPI:** leads calientes por corrida y **costo por lead** (costo de Apify / leads calientes).
- **Proyección** (supuestos marcados, a validar con la cliente): tasa de respuesta al mail y precio de una web básica.
""")
code("""
COSTO_CORRIDA_USD = 0.12        # 2 corridas de kaix/google-maps-places-scraper (1.415 lugares a US$ 0,08/1.000 + cómputo). Fuente: consola Apify, 24/11/2026
TASA_CONVERSION = 0.03          # SUPUESTO: 3 % de los leads contactados compra una web
PRECIO_WEB_ARS = 250_000        # SUPUESTO: precio de una web básica para comercio
n_leads, n_cal = int(limpio["es_lead"].sum()), int(limpio["lead_caliente"].sum())
kpi = pd.Series({
    "comercios analizados": len(limpio),
    "% sin web propia": round(100 * (1 - limpio["tiene_web_propia"].mean()), 1),
    "leads (sin web propia)": n_leads,
    "leads calientes": n_cal,
    "costo por lead caliente (USD)": round(COSTO_CORRIDA_USD / n_cal, 5),
    "ventas esperadas (leads calientes × conversión)": round(n_cal * TASA_CONVERSION, 1),
    "facturación esperada (ARS)": round(n_cal * TASA_CONVERSION * PRECIO_WEB_ARS),
})
display(kpi.map(lambda v: f"{v:,.5g}" if isinstance(v, float) else f"{v:,}").to_frame("valor"))
""")

md("## 7. Exportar")
code("""
os.makedirs(DIR_SALIDA, exist_ok=True)
final.to_csv(os.path.join(DIR_SALIDA, "tabla_final.csv"), index=False)
limpio.drop(columns=["horarios"]).to_csv(os.path.join(DIR_SALIDA, "comercios_limpios.csv"), index=False)
leads(limpio).merge(limpio[["nombre", "direccion", "prob_web"]], on=["nombre", "direccion"], how="left") \\
    .sort_values(["lead_caliente", "prob_web"], ascending=False).to_csv(os.path.join(DIR_SALIDA, "leads.csv"), index=False)
descartes.drop(columns=["horarios"], errors="ignore").to_csv(os.path.join(DIR_SALIDA, "auditoria_descartes.csv"), index=False)
print("Archivos en", DIR_SALIDA, "→", sorted(os.listdir(DIR_SALIDA)))
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path("colabs/analytics").mkdir(parents=True, exist_ok=True)
nbf.write(nb, "colabs/analytics/capstone_comercios_sin_web.ipynb")
print("ok", len(C), "celdas")
