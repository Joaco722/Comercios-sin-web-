# Diccionario de datos · tabla_final.csv

| Columna | Rol | Tipo | Origen (Google Maps) | Transformación |
|---|---|---|---|---|
| tiene_web_propia | target | 0/1 | `website` | 1 si el dominio es propio; 0 si no hay web, es una red social (Instagram, Facebook, WhatsApp, Linktree) o una plataforma de terceros (PedidosYa, Tiendanube, sitios de Google) |
| rating | feature | 1–5 | `rating` | Faltantes (5 %) imputados con la mediana del rubro |
| n_resenas | feature | entero | `reviewCount` | Faltante = 0. En el modelo se usa log(1 + x) |
| n_fotos | feature | 0–10 | `imagesCount` | El Actor devuelve como máximo 10 |
| horas_semana | feature | 0–168 | `hours` | Suma de las franjas de cada día; faltantes con la mediana del rubro |
| dias_abiertos | feature | 0–7 | `hours` | Días que no figuran como "Cerrado" |
| tiene_telefono | feature | 0/1 | `phone` | 1 si hay teléfono |
| abre_domingo | feature | 0/1 | `hours` | 1 si el domingo no figura como "Cerrado" |
| rubro_cod | feature | 1–12 | `matchedQuery` | 1 peluquería, 2 ferretería, 3 almacén, 4 taller mecánico, 5 panadería, 6 veterinaria, 7 gimnasio, 8 carnicería, 9 restaurante, 10 tienda de ropa, 11 verdulería, 12 farmacia |
| ciudad_cod | feature | 1–2 | ciudad de la búsqueda | 1 Concordia, 2 Paraná |

Regla de lead caliente (no es columna de la tabla final, está en `comercios_limpios.csv`): sin web propia, con teléfono, rating ≥ 4 y al menos 10 reseñas.
