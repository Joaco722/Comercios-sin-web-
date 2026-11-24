# Buscador de comercios sin web (Entre Ríos)

PoC del Capstone (Samsung Innovation Campus). Busca comercios de 12 rubros en Google Maps
(Actor `kaix/google-maps-places-scraper`), aplica el ETL del proyecto y devuelve:

- **Dataset:** comercios sin página web propia, ordenados (leads).
- **Key-value store:** `TABLA_FINAL` (CSV, 10 columnas numéricas), `LIMPIO`, `DESCARTES` (CSV) y `EDA` (JSON).

Costo aproximado: US$ 0,06 por ciudad. Para una prueba chica: `maxPorCiudad` = 20 y `validarFilas` = false.
