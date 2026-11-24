"""
Actor de Apify: buscador de comercios sin web propia.

Encapsula el ETL (y un resumen EDA) dentro de Apify, como pide la cátedra para
el flujo con Make: Make solo dispara este Actor y recibe el resultado.

Flujo:
  1. Llama al Actor público kaix/google-maps-places-scraper por cada ciudad.
  2. Aplica etl.extraer -> etl.transformar -> etl.tabla_final -> etl.validar.
  3. Guarda: dataset por defecto = leads (comercios sin web propia, ordenados);
             key-value store: TABLA_FINAL (CSV), LIMPIO (CSV), DESCARTES (CSV), EDA (JSON).
"""
from __future__ import annotations

import asyncio

import pandas as pd
from apify import Actor

import etl

SCRAPER = "kaix/google-maps-places-scraper"


async def main() -> None:
    async with Actor:
        cfg = await Actor.get_input() or {}
        ciudades = cfg.get("ciudades", ["Concordia", "Paraná", "Gualeguaychú", "Concepción del Uruguay"])
        rubros = cfg.get("rubros", list(etl.RUBROS))
        max_por_ciudad = int(cfg.get("maxPorCiudad", 0))
        radio_km = float(cfg.get("radioKm", 8))
        validar = bool(cfg.get("validarFilas", True))

        crudos = []
        for ciudad in ciudades:
            Actor.log.info(f"Buscando {len(rubros)} rubros en {ciudad}…")
            run = await Actor.call(SCRAPER, run_input={
                "searchTerms": rubros,
                "location": f"{ciudad}, Entre Ríos, Argentina",
                # "nearby" + sin "country": con "area"/"country" el actor recorría todo el país
                "maxResults": max_por_ciudad,  # 0 = sin tope (≈60 por rubro en modo nearby)
                "language": "es",
                "searchType": "nearby",
                "radius": radio_km,
                "websiteFilter": "all",
            })
            items = (await Actor.apify_client.dataset(run.default_dataset_id).list_items()).items
            crudos.append(etl.extraer(items, ciudad))

        crudo = pd.concat(crudos, ignore_index=True)
        limpio, descartes = etl.transformar(crudo)
        final = etl.tabla_final(limpio)
        if validar:
            etl.validar(final)
        lista = etl.leads(limpio)

        eda = {
            "filas_crudas": int(len(crudo)),
            "filas_limpias": int(len(limpio)),
            "descartes_por_motivo": descartes["motivo_descarte"].value_counts().to_dict() if len(descartes) else {},
            "pct_sin_web_propia": round(float(1 - limpio["tiene_web_propia"].mean()) * 100, 1),
            "presencia": limpio["presencia"].value_counts().to_dict(),
            "pct_sin_web_por_rubro": (100 - limpio.groupby("rubro")["tiene_web_propia"].mean() * 100).round(1).to_dict(),
            "leads": int(limpio["es_lead"].sum()),
            "leads_calientes": int(limpio["lead_caliente"].sum()),
        }

        store = await Actor.open_key_value_store()
        await store.set_value("TABLA_FINAL", final.to_csv(index=False), content_type="text/csv")
        await store.set_value("LIMPIO", limpio.drop(columns=["horarios"]).to_csv(index=False), content_type="text/csv")
        await store.set_value("DESCARTES", descartes.drop(columns=["horarios"], errors="ignore").to_csv(index=False),
                              content_type="text/csv")
        await store.set_value("EDA", eda)
        await Actor.push_data(lista.where(lista.notna(), None).to_dict(orient="records"))
        Actor.log.info(f"Listo: {eda['leads']} leads ({eda['leads_calientes']} calientes) de {eda['filas_limpias']} comercios.")


if __name__ == "__main__":
    asyncio.run(main())
