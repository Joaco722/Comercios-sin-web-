# Buscador de comercios sin web · Capstone SIC

Prueba de concepto: buscar comercios de Entre Ríos en Google Maps, detectar cuáles **no tienen web propia**, priorizarlos y mandarles una propuesta por mail de forma automática.

**Cliente:** agencia o freelancer que vende sitios web a comercios locales.
**Zona y rubros:** Concordia y Paraná. 12 rubros: peluquería, ferretería, almacén, taller mecánico, panadería, veterinaria, gimnasio, carnicería, restaurante, tienda de ropa, verdulería y farmacia.

## 1. Arquitectura y flujo

```
Trigger (manual / programado en Make)
   └─► Apify: Google Maps (kaix/google-maps-places-scraper)
          └─► ETL (actor_apify/src/etl.py): limpieza, presencia web, horas, flags
                 └─► EDA + inferencia + modelo (colabs/analytics)
                        └─► Destino: Drive (datos y salida) + Google Sheets (leads)
                               └─► Make: mail de propuesta a los leads (demo)
```

| Escenario de Make | Qué hace |
|---|---|
| 1 · Buscar comercios en Google Maps (Apify) → Drive | Corre el Actor en modo sincrónico (unos 30 s por ciudad) y guarda el JSON en `Drive/Capstone/comercios_sin_web/datos` |
| 3 · Mail de propuesta a comercios sin web (demo) | Lee los leads marcados "Enviar demo = Sí" de la planilla y manda el mail **a una casilla propia de prueba** |

## 2. Estructura del repositorio

```
capstone-leads/
├── actor_apify/            # Actor de Apify que encapsula el ETL (para el flujo con Make)
│   ├── src/main.py         # llama al scraper de Google Maps y aplica el ETL
│   ├── src/etl.py          # funciones del ETL (las mismas que usa el notebook)
│   ├── .actor/             # actor.json + input_schema.json
│   ├── Dockerfile
│   └── requirements.txt
├── colabs/
│   ├── analytics/capstone_comercios_sin_web.ipynb   # ETL + EDA + inferencia + modelo (ejecutado)
│   └── build_notebook.py   # genera el notebook a partir de etl.py
├── data/
│   ├── raw/                # respuesta de Google Maps (JSON por ciudad)
│   └── processed/          # tabla_final.csv, comercios_limpios.csv, leads.csv, auditoría
├── make/                   # blueprints de los escenarios
└── docs/                   # diccionario de datos y costos
```

## 3. Actor de Apify

- **Scraper usado:** [`kaix/google-maps-places-scraper`](https://apify.com/kaix/google-maps-places-scraper). Cuesta US$ 0,08 cada 1.000 lugares. Se eligió frente al oficial `compass/crawler-google-places` (US$ 4 cada 1.000 + US$ 1 por filtro), que no entraba en la capa gratuita.
- **Modo de búsqueda:** `nearby`, hasta unos 60 resultados por rubro y ciudad.
  - No usar el campo `country`: hace que se busque en todo el país (nos pasó en la primera prueba).
  - Con el modo `area` y un tope de resultados, el Actor llena el tope con el primer rubro de la lista.
- **Actor propio (`actor_apify/`):** recibe ciudades y rubros, llama al scraper, aplica el ETL, entrena el modelo y devuelve:
  - en el dataset, la **inferencia**: los leads con `prob_web` (probabilidad de tener web propia) y `prioridad` (alta / media / baja), ordenados para contactar;
  - en el key-value store: `TABLA_FINAL`, `LIMPIO`, `DESCARTES`, `EDA` y `METRICAS_MODELO`.
- **Cómo publicarlo:** en Apify Console, crear un Actor desde este repositorio de Git con la carpeta `actor_apify` (URL `https://github.com/Joaco722/Comercios-sin-web-.git#main:actor_apify`), o desde la terminal: `cd actor_apify && apify login && apify push`.

## 4. Funciones del ETL

| Función | Input | Output | Proceso |
|---|---|---|---|
| `extraer()` | Lista de lugares del Actor + ciudad buscada | DataFrame crudo (19 columnas) | Aplana los campos que usa el pipeline |
| `clasificar_presencia()` | URL del campo `website` | `web_propia` / `solo_redes` / `plataforma_terceros` / `sin_web` | Clasifica por dominio: Instagram, Facebook, WhatsApp y Linktree son redes; PedidosYa, Tiendanube y sitios de Google son terceros |
| `horas_semanales()` | Horarios por día | (horas por semana, días abiertos) | Parsea las franjas: "24 horas" = 24, "Cerrado" = 0 |
| `abre_domingo()` | Horarios por día | 1 / 0 / nan | Mira si el domingo dice "Cerrado" |
| `transformar()` | DataFrame crudo | (limpio, descartes) | Descarta lugares fuera de Argentina, cerrados, de otros rubros y duplicados; calcula presencia, horas, códigos, `es_lead` y `lead_caliente` |
| `tabla_final()` | Limpio | 10 columnas numéricas sin nulos | Imputa con la mediana del rubro |
| `validar()` | Tabla final | — (assert) | Exige 700–2.000 filas, como máximo 10 columnas, rangos válidos y target binario |
| `leads()` | Limpio | Lista de comercios para contactar | Filtra los que no tienen web propia y ordena por lead caliente, reseñas y rating |
| `modelo.preparar_X()` | Tabla final | Matriz de features | Quita el target, log(1 + reseñas) y dummies de rubro y ciudad |
| `modelo.puntuar()` | Tabla final | (prob_web por comercio, métricas) | Regresión logística balanceada con validación cruzada estratificada de 5 particiones |
| `modelo.priorizar()` | Leads con prob_web | Leads con prioridad, ordenados | Tercios de probabilidad: alta / media / baja |

**Embudo:** 1.415 lugares → 83 descartados por estar en Salto (Uruguay) → **1.332 comercios**.

## 5. Datos y modelo

- **Tabla final** (`data/processed/tabla_final.csv`): 1.332 filas × 10 columnas.
  - Target: `tiene_web_propia`.
  - Features: `rating`, `n_resenas`, `n_fotos`, `horas_semana`, `dias_abiertos`, `tiene_telefono`, `abre_domingo`, `rubro_cod` y `ciudad_cod`.
- **Hallazgos de inferencia:**
  - El **84,8 %** de los comercios no tiene web propia.
  - La falta de web **depende del rubro** (χ², p < 0,001; V de Cramér 0,37, efecto mediano a grande). Va de 2 % de verdulerías con web a 51 % de farmacias.
  - **No depende de la ciudad** (p = 0,49).
  - Los comercios sin ninguna presencia web tienen menos reseñas: mediana de 16 contra 56 de los que tienen web propia. Por Kruskal-Wallis, p < 0,001 y ε² = 0,06. Se usó Kruskal-Wallis porque fallan los supuestos del ANOVA (Shapiro).
- **Modelo:** probabilidad de tener web propia, con validación cruzada de 5 particiones.

| Modelo | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Línea base (estratificada) | 0,51 | 0,18 | 0,17 |
| **Regresión logística (balanceada)** | **0,80** | **0,72** | **0,43** |
| Random Forest | 0,79 | 0,60 | 0,43 |

- **Uso del modelo:** entre los comercios sin web, los que tienen probabilidad alta se parecen a los que ya invirtieron en una. En el tercio de prioridad alta, el 77 % son leads calientes; en el de prioridad baja, el 22 %.

## 6. Métricas

- **Métrica de negocio:** % de comercios sin web propia por rubro y ciudad, que es el tamaño del mercado.
  - **KPI 1:** leads calientes por corrida. Un lead caliente no tiene web propia, tiene teléfono, rating ≥ 4 y al menos 10 reseñas. Hoy hay **574**.
  - **KPI 2:** costo por lead caliente: unos US$ 0,0002.
- **Métrica del modelo:** ROC-AUC 0,80 frente a 0,51 de la línea base, con recall de 0,72 para la clase "tiene web".

## 7. Costo computacional por corrida

| Concepto | Valor |
|---|---|
| Tiempo por ciudad (modo `nearby`, 12 rubros) | unos 30 s |
| Lugares por ciudad | unos 700 |
| Costo en Apify | unos US$ 0,06 por ciudad (US$ 0,08 cada 1.000 lugares + cómputo) |
| Operaciones de Make | 4 por corrida del escenario 1 y 2 por mail enviado |
| Capa gratuita | Apify US$ 5 por mes; Make 1.000 operaciones por mes |

## 8. Cómo correr el proyecto

1. **Datos:** en Make, ejecutar el escenario 1. Otra opción es correr el Actor propio con `apify call`. También se pueden usar los JSON que ya están en `data/raw`.
2. **Notebook:** abrir `colabs/analytics/capstone_comercios_sin_web.ipynb` en Colab.
   - Los datos tienen que estar en `MyDrive/Capstone/comercios_sin_web/datos`.
   - En local usa `data/raw`.
3. **Mails:** escenario 3 de Make. Necesita conectar Google Sheets y Gmail. Manda solo a la casilla de prueba.
4. **Tablero:** Data Studio, conectado a la planilla "Capstone · Comercios sin web (tablero y leads)".

## Limitaciones conocidas

- **Fotos:** `n_fotos` tiene un tope de 10 en la respuesta del Actor.
- **Perfil reclamado:** `isClaimed` llega siempre en `false` en el modo `nearby`, así que no se usa.
- **Sitios mal clasificados:** un sitio en un dominio propio que en realidad es una plantilla gratuita se cuenta como web propia.
- **Email de los comercios:** Google Maps no lo publica. Para enviar mails reales hace falta otra fuente; el escenario 3 es una demostración con casilla propia.
- **Supuestos sin validar:** la conversión del 3 % y el precio de una web son supuestos que hay que validar con la cliente.
