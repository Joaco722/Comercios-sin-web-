"""
ETL del buscador de comercios sin web (Google Maps -> tabla de leads).

Cada función documenta Input, Output y Proceso para poder traducirla
1 a 1 a un módulo de Make / nodo de n8n, o correrla dentro del Actor de Apify.
Solo usa la librería estándar + pandas/numpy.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reglas de negocio (constantes documentadas)
# ---------------------------------------------------------------------------
DOMINIOS_REDES = (
    "instagram.com", "facebook.com", "fb.com", "fb.me", "wa.me", "whatsapp.com",
    "linktr.ee", "tiktok.com", "twitter.com", "x.com", "youtube.com", "linkedin.com",
    "beacons.ai", "taplink.cc", "bio.link",
)
# Plataformas de tienda/reserva de terceros: el comercio tiene presencia, pero no web propia.
DOMINIOS_TERCEROS = (
    "pedidosya.com", "rappi.com", "mercadolibre.com", "mercadoshops.com",
    "tiendanube.com", "mitiendanube.com", "empretienda.com", "google.com", "business.site", "negocio.site",
    "wixsite.com", "blogspot.com", "agendapro.com", "reservio.com",
)
CIUDADES = {"concordia": 1, "parana": 2, "gualeguaychu": 3, "concepcion del uruguay": 4}
RUBROS = {
    "peluqueria": 1, "ferreteria": 2, "almacen": 3, "taller mecanico": 4, "panaderia": 5,
    "veterinaria": 6, "gimnasio": 7, "carniceria": 8, "restaurante": 9, "tienda de ropa": 10,
    "verduleria": 11, "farmacia": 12,
}
# Lead "caliente": sin web propia, abierto, con teléfono, buena reputación y movimiento.
LEAD_MIN_RATING = 4.0
LEAD_MIN_RESENAS = 10


def sin_acentos(texto: str) -> str:
    """Input: str. Output: str en minúsculas sin tildes. Proceso: normaliza NFKD."""
    t = unicodedata.normalize("NFKD", str(texto or ""))
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


# ---------------------------------------------------------------------------
# E - Extract
# ---------------------------------------------------------------------------
def extraer(registros: list[dict], ciudad_busqueda: str) -> pd.DataFrame:
    """
    Input: lista de dicts tal como los devuelve el Actor de Google Maps + ciudad buscada.
    Output: DataFrame crudo con las columnas que usa el pipeline (una fila por lugar).
    Proceso: aplana los campos anidados que interesan y agrega la ciudad de la búsqueda.
    """
    filas = []
    for r in registros:
        filas.append({
            "place_id": r.get("placeId"),
            "nombre": r.get("title"),
            "rubro_busqueda": r.get("matchedQuery"),
            "categoria_maps": r.get("primaryTypeDisplayName"),
            "ciudad_busqueda": ciudad_busqueda,
            "ciudad_maps": r.get("city"),
            "pais": r.get("countryCode"),
            "direccion": r.get("formattedAddress"),
            "latitud": r.get("latitude"),
            "longitud": r.get("longitude"),
            "rating": r.get("rating"),
            "n_resenas": r.get("reviewCount"),
            "n_fotos": r.get("imagesCount"),
            "web": r.get("website"),
            "telefono": r.get("phone"),
            "perfil_reclamado": r.get("isClaimed"),
            "estado": r.get("businessStatus"),
            "horarios": r.get("hours"),
            "url_maps": r.get("googleMapsUri"),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# T - Transform
# ---------------------------------------------------------------------------
def clasificar_presencia(url: str | None) -> str:
    """
    Input: URL del campo 'website' de Google Maps (o None).
    Output: 'web_propia' | 'solo_redes' | 'plataforma_terceros' | 'sin_web'.
    Proceso: mira el dominio; Instagram/Facebook/WhatsApp/Linktree cuentan como redes,
             tiendas y sitios gratuitos de terceros no cuentan como web propia.
    """
    if url is None or (isinstance(url, float) and np.isnan(url)) or not str(url).strip():
        return "sin_web"
    u = str(url).strip()
    if not u.startswith("http"):
        u = "http://" + u
    dominio = (urlparse(u).netloc or "").lower().removeprefix("www.")
    partes = dominio.split(".")

    def coincide(d: str) -> bool:
        # 'pedidosya.com' coincide con 'pedidosya.com.ar' y con 'www.pedidosya.com'
        base = d.split(".")[0]
        return dominio == d or dominio.endswith("." + d) or base in partes[:-1]

    if any(coincide(d) for d in DOMINIOS_REDES):
        return "solo_redes"
    if any(coincide(d) for d in DOMINIOS_TERCEROS):
        return "plataforma_terceros"
    return "web_propia"


_RX_FRANJA = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)?\s*[-–]\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)?",
    re.I,
)


def _a_hora(h: str, m: str | None, sufijo: str | None, sufijo_fin: str | None) -> float:
    hora = int(h) % 12 if (sufijo or sufijo_fin) else int(h)
    suf = (sufijo or sufijo_fin or "").lower().replace(" ", "").replace(".", "")
    if suf == "pm":
        hora += 12
    return hora + (int(m) / 60 if m else 0)


def horas_semanales(horarios) -> tuple[float, float]:
    """
    Input: lista [{'day': 'lunes', 'hours': '8 a.m.-12 p.m., 4-8 p.m.'}, ...] o None.
    Output: (horas abiertas por semana, días abiertos por semana); (nan, nan) si no hay dato.
    Proceso: parsea cada franja horaria; 'Abierto las 24 horas' = 24 h; 'Cerrado' = 0.
    """
    if not isinstance(horarios, list) or not horarios:
        return (np.nan, np.nan)
    total, dias = 0.0, 0
    for d in horarios:
        txt = sin_acentos(d.get("hours", ""))
        if "cerrado" in txt or "closed" in txt:
            continue
        if "24 horas" in txt or "24 hours" in txt:
            total += 24
            dias += 1
            continue
        horas_dia = 0.0
        for m in _RX_FRANJA.finditer(txt):
            h1, m1, s1, h2, m2, s2 = m.groups()
            # si solo el fin tiene sufijo (ej. '4-8 p.m.'), el inicio hereda el sufijo
            ini = _a_hora(h1, m1, s1 or s2, None)
            fin = _a_hora(h2, m2, s2, None)
            if fin <= ini:
                fin += 12 if fin + 12 > ini else 24
            horas_dia += min(fin - ini, 24)
        if horas_dia > 0:
            total += min(horas_dia, 24)
            dias += 1
    return (round(total, 1), float(dias))


def abre_domingo(horarios) -> float:
    """
    Input: lista de horarios por día (o None).
    Output: 1.0 si atiende el domingo, 0.0 si no, nan si no hay horarios cargados.
    Proceso: busca el día 'domingo' y mira si dice 'Cerrado'.
    """
    if not isinstance(horarios, list) or not horarios:
        return np.nan
    for d in horarios:
        if sin_acentos(d.get("day", "")) in ("domingo", "sunday"):
            return 0.0 if "cerrado" in sin_acentos(d.get("hours", "")) else 1.0
    return 0.0


def transformar(crudo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input: DataFrame crudo de extraer() (puede venir de varias corridas/ciudades).
    Output: (limpio, descartes). limpio = una fila por comercio con columnas derivadas;
            descartes = filas eliminadas con el motivo (auditoría del embudo).
    Proceso: normaliza textos, descarta fuera de Argentina / cerrados / duplicados /
             rubro no buscado, clasifica la presencia web y calcula horas de atención.
    """
    df = crudo.copy()
    descartes = []

    def descartar(mascara, motivo):
        nonlocal df
        out = df[mascara].assign(motivo_descarte=motivo)
        descartes.append(out)
        df = df[~mascara]

    descartar(df["place_id"].isna(), "sin identificador")
    descartar(df["pais"].fillna("AR") != "AR", "fuera de Argentina (p. ej. Salto, UY)")
    descartar(df["estado"].fillna("OPERATIONAL") != "OPERATIONAL", "cerrado temporal o definitivo")
    df = df.assign(rubro=df["rubro_busqueda"].map(sin_acentos),
                   ciudad=df["ciudad_busqueda"].map(sin_acentos))
    descartar(~df["rubro"].isin(RUBROS), "rubro fuera de la búsqueda")
    descartar(df.duplicated("place_id", keep="first"), "duplicado (apareció en otra búsqueda)")

    df["presencia"] = df["web"].map(clasificar_presencia)
    df["tiene_web_propia"] = (df["presencia"] == "web_propia").astype(int)
    df["tiene_telefono"] = df["telefono"].notna().astype(int)
    # isClaimed llega siempre en False en la búsqueda 'nearby' del Actor: no se usa como feature.
    df["abre_domingo"] = df["horarios"].map(abre_domingo)
    hs = df["horarios"].map(horas_semanales)
    df["horas_semana"] = hs.map(lambda t: t[0])
    df["dias_abiertos"] = hs.map(lambda t: t[1])
    df["n_resenas"] = pd.to_numeric(df["n_resenas"], errors="coerce").fillna(0).astype(int)
    df["n_fotos"] = pd.to_numeric(df["n_fotos"], errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rubro_cod"] = df["rubro"].map(RUBROS)
    df["ciudad_cod"] = df["ciudad"].map(CIUDADES)
    df["es_lead"] = (df["tiene_web_propia"] == 0).astype(int)
    df["lead_caliente"] = ((df["es_lead"] == 1) & (df["tiene_telefono"] == 1)
                           & (df["rating"].fillna(0) >= LEAD_MIN_RATING)
                           & (df["n_resenas"] >= LEAD_MIN_RESENAS)).astype(int)
    aud = pd.concat(descartes, ignore_index=True) if descartes else pd.DataFrame()
    return df.reset_index(drop=True), aud


COLUMNAS_FINALES = [
    "tiene_web_propia",   # target (clasificación)
    "rating", "n_resenas", "n_fotos", "horas_semana", "dias_abiertos",
    "tiene_telefono", "abre_domingo", "rubro_cod", "ciudad_cod",
]


def tabla_final(limpio: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame limpio de transformar().
    Output: tabla 100 % numérica, sin nulos, con COLUMNAS_FINALES (10 columnas).
    Proceso: imputa rating, horas, días y domingo faltantes con la mediana del rubro
             (o la global si el rubro no tiene dato).
    """
    t = limpio.copy()
    for col in ["rating", "horas_semana", "dias_abiertos", "abre_domingo"]:
        med_rubro = t.groupby("rubro")[col].transform("median")
        t[col] = t[col].fillna(med_rubro).fillna(t[col].median())
    t = t[COLUMNAS_FINALES].astype(float)
    assert t.notna().all().all(), "Quedaron nulos en la tabla final"
    return t


# ---------------------------------------------------------------------------
# L - Load (validaciones = contrato del dato)
# ---------------------------------------------------------------------------
def validar(final: pd.DataFrame, min_filas: int = 700, max_filas: int = 2000) -> None:
    """
    Input: tabla final. Output: None (lanza AssertionError si algo no cumple).
    Proceso: verifica cantidad de filas, columnas numéricas, rangos y target binario.
    """
    assert min_filas <= len(final) <= max_filas, f"Filas fuera de rango: {len(final)}"
    assert final.shape[1] <= 10, "Más de 10 columnas"
    assert final["tiene_web_propia"].isin([0, 1]).all()
    assert final["rating"].between(1, 5).all()
    assert (final["horas_semana"] >= 0).all() and (final["horas_semana"] <= 168).all()
    assert (final["dias_abiertos"] >= 0).all() and (final["dias_abiertos"] <= 7).all()


def leads(limpio: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame limpio. Output: lista de comercios sin web propia, ordenada para contactar.
    Proceso: filtra es_lead == 1 y ordena por lead caliente, reseñas y rating.
    """
    cols = ["nombre", "rubro", "ciudad", "presencia", "web", "telefono", "rating",
            "n_resenas", "lead_caliente", "direccion", "url_maps"]
    return (limpio[limpio["es_lead"] == 1][cols]
            .sort_values(["lead_caliente", "n_resenas", "rating"], ascending=False)
            .reset_index(drop=True))
