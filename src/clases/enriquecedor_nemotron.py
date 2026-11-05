"""Enriquecimiento matemático, geodésico y contextual urbano con NVIDIA Nemotron."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


class EnriquecedorNemotron:
    """Enriquece el DataFrame de viviendas con métricas derivadas, distancias Haversine y NVIDIA Nemotron."""

    FACTOR_SQFT_A_M2: float = 10.7639
    RADIO_TIERRA_KM: float = 6371.0088

    # Polos urbanos, comerciales y de transporte de referencia en King County, WA
    POLOS_REFERENCIA: dict[str, tuple[float, float]] = {
        "Downtown Seattle": (47.6062, -122.3321),
        "Bellevue Comercial": (47.6101, -122.2015),
        "Aeropuerto SeaTac": (47.4502, -122.3088),
    }

    # Anclas geográficas oficiales de barrios de Seattle y municipios de King County (RAG geodésico)
    ANCLAS_KING_COUNTY: tuple[tuple[str, str, float, float], ...] = (
        ("Downtown / Belltown - Seattle", "Seattle", 47.6101, -122.3370),
        ("Capitol Hill / Central District - Seattle", "Seattle", 47.6210, -122.3120),
        ("Queen Anne / Magnolia - Seattle", "Seattle", 47.6390, -122.3750),
        ("Ballard / Crown Hill - Seattle", "Seattle", 47.6760, -122.3840),
        ("Green Lake / Wallingford - Seattle", "Seattle", 47.6750, -122.3350),
        ("University District / Laurelhurst - Seattle", "Seattle", 47.6620, -122.2900),
        ("Northgate / Lake City - Seattle", "Seattle", 47.7120, -122.3050),
        ("West Seattle", "Seattle", 47.5550, -122.3860),
        ("Beacon Hill / Columbia City - Seattle", "Seattle", 47.5580, -122.2920),
        ("Rainier Beach / South Seattle", "Seattle", 47.5150, -122.2650),
        ("Shoreline / Lake Forest Park", "North King County", 47.7560, -122.3300),
        ("Kenmore / Bothell", "North King County", 47.7610, -122.2150),
        ("Woodinville", "North King County", 47.7540, -122.1550),
        ("Downtown Bellevue", "Eastside (Bellevue/Kirkland/Redmond)", 47.6101, -122.2015),
        ("Eastgate / Factoria - Bellevue", "Eastside (Bellevue/Kirkland/Redmond)", 47.5760, -122.1520),
        ("Mercer Island", "Eastside (Bellevue/Kirkland/Redmond)", 47.5707, -122.2221),
        ("Medina / Clyde Hill", "Eastside (Bellevue/Kirkland/Redmond)", 47.6280, -122.2280),
        ("Kirkland", "Eastside (Bellevue/Kirkland/Redmond)", 47.6815, -122.2087),
        ("Redmond", "Eastside (Bellevue/Kirkland/Redmond)", 47.6740, -122.1215),
        ("Sammamish", "Eastside (Bellevue/Kirkland/Redmond)", 47.6163, -122.0356),
        ("Issaquah", "Eastside (Bellevue/Kirkland/Redmond)", 47.5301, -122.0326),
        ("Newcastle", "Eastside (Bellevue/Kirkland/Redmond)", 47.5390, -122.1650),
        ("Renton", "South King County", 47.4829, -122.2171),
        ("Tukwila / SeaTac", "South King County", 47.4480, -122.2950),
        ("Burien / Normandy Park", "South King County", 47.4650, -122.3480),
        ("Des Moines", "South King County", 47.4018, -122.3243),
        ("Federal Way", "South King County", 47.3223, -122.3126),
        ("Kent", "South King County", 47.3809, -122.2348),
        ("Auburn", "South King County", 47.3073, -122.2285),
        ("Covington", "South King County", 47.3590, -122.1120),
        ("Maple Valley / Black Diamond", "South King County", 47.3550, -122.0350),
        ("Vashon Island", "Vashon / Islas", 47.4470, -122.4600),
        ("Duvall / Carnation", "Foothills / Este Semirrural", 47.7000, -121.9500),
        ("Snoqualmie / Fall City", "Foothills / Este Semirrural", 47.5400, -121.8400),
        ("North Bend", "Foothills / Este Semirrural", 47.4957, -121.7868),
        ("Enumclaw", "Foothills / Este Semirrural", 47.2043, -121.9915),
    )

    MODELOS_RESPALDO: tuple[str, ...] = (
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3.5-bolt-e4b",
    )

    VALORES_NIVEL: set[str] = {"Alto", "Medio", "Bajo"}
    VALORES_PARQUES: set[str] = {"Alta", "Media", "Baja"}
    VALORES_ENTORNO: set[str] = {
        "Urbano denso y comercial",
        "Suburbano residencial familiar",
        "Frente costero / lacustre recreativo",
        "Semirrural / Boscoso",
    }
    VALORES_AGUA: set[str] = {
        "Estrecho marino (Puget Sound)",
        "Lago interior (Washington/Sammamish/Union)",
        "Río / Interior sin costa",
    }

    def __init__(
        self,
        modelo: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        paso_cuadrante: float = 0.05,
        ruta_env: str | Path | None = None,
    ) -> None:
        self.modelo = modelo
        self.base_url = base_url
        self.paso_cuadrante = paso_cuadrante
        self._cargar_entorno(ruta_env)

    @staticmethod
    def _cargar_entorno(ruta_env: str | Path | None = None) -> None:
        """Carga variables desde .env en la raíz del proyecto si existe, sin exponer secretos."""
        if ruta_env is not None:
            load_dotenv(dotenv_path=Path(ruta_env), override=False)
            return

        actual = Path.cwd().resolve()
        for carpeta in [actual, *actual.parents]:
            candidato = carpeta / ".env"
            if candidato.exists():
                load_dotenv(dotenv_path=candidato, override=False)
                break

    def calcular_variables_derivadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Incorpora conversiones monetarias, áreas en m², antigüedad, remodelación y tramos analíticos."""
        enr = df.copy()

        # 1. Conversión exacta de ln(precio + 1) a dólares estadounidenses: e^x - 1
        enr["Precio (USD)"] = np.expm1(enr["Precio (log)"]).round(0)

        # 2. Conversión de superficies de pies cuadrados (sqft) a metros cuadrados (m²)
        enr["Área habitable (m2)"] = (
            enr["Área habitable (sqft)"] / self.FACTOR_SQFT_A_M2
        ).round(2)
        enr["Área del lote (m2)"] = (
            enr["Área del lote (sqft)"] / self.FACTOR_SQFT_A_M2
        ).round(2)
        enr["Área habitable vecinos (m2)"] = (
            enr["Área habitable vecinos (sqft)"] / self.FACTOR_SQFT_A_M2
        ).round(2)

        # 3. Indicador unitario estandarizado para peritos avaluadores
        enr["Precio por m2 (USD)"] = (
            enr["Precio (USD)"] / enr["Área habitable (m2)"]
        ).round(2)

        # 4. Variables temporales, antigüedad y condición de remodelación
        enr["Periodo de venta"] = enr["Fecha de venta"].dt.strftime("%Y-%m")
        enr["Antigüedad (años)"] = (
            enr["Año de venta"] - enr["Año de construcción"]
        ).clip(lower=0)
        enr["Década de construcción"] = (
            ((enr["Año de construcción"] // 10) * 10).astype(str) + "s"
        )
        enr["Condición de remodelación"] = np.where(
            enr["Año de renovación"] > 0, "Remodelada", "No remodelada"
        )

        # 5. Categorización de tamaño habitable y nivel de calificación constructiva
        enr["Categoría de tamaño"] = pd.cut(
            enr["Área habitable (m2)"],
            bins=[0, 130, 200, 280, np.inf],
            labels=[
                "1. Pequeña (<130 m²)",
                "2. Mediana (130–199 m²)",
                "3. Grande (200–279 m²)",
                "4. Muy grande (≥280 m²)",
            ],
            right=False,
        ).astype(str)

        condiciones_calif = [
            enr["Calificación"] <= 6,
            enr["Calificación"] == 7,
            enr["Calificación"].between(8, 9),
            enr["Calificación"] >= 10,
        ]
        etiquetas_calif = [
            "1. Baja (4–6)",
            "2. Estándar (7)",
            "3. Media-Alta (8–9)",
            "4. Alta / Lujo (10–13)",
        ]
        enr["Nivel de calificación"] = np.select(
            condiciones_calif, etiquetas_calif, default="Sin clasificar"
        )

        return enr

    def calcular_distancias_polos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula distancias geodésicas Haversine (km) a centros comerciales y de transporte."""
        enr = df.copy()
        lat = enr["Latitud"].to_numpy()
        lon = enr["Longitud"].to_numpy()

        for nombre_polo, (lat_ref, lon_ref) in self.POLOS_REFERENCIA.items():
            enr[f"Distancia {nombre_polo} (km)"] = self._haversine_vectorizado(
                lat, lon, lat_ref, lon_ref
            ).round(2)

        enr["Distancia polo comercial más cercano (km)"] = np.minimum(
            enr["Distancia Downtown Seattle (km)"],
            enr["Distancia Bellevue Comercial (km)"],
        ).round(2)

        enr["Anillo urbano"] = pd.cut(
            enr["Distancia polo comercial más cercano (km)"],
            bins=[-0.01, 10.0, 20.0, np.inf],
            labels=[
                "1. Céntrico (<10 km)",
                "2. Suburbano cercano (10–20 km)",
                "3. Periférico (>20 km)",
            ],
        ).astype(str)

        return enr

    def enriquecer_con_nemotron(
        self,
        df: pd.DataFrame,
        ruta_catalogo: str | Path,
        tamano_lote: int = 16,
        forzar_regeneracion: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Enriquece el DataFrame mediante un catálogo espacial de cuadrantes generado con NVIDIA Nemotron.

        Devuelve:
            (df_enriquecido, df_catalogo_barrios, df_validacion_union)
        """
        ruta_cat = Path(ruta_catalogo)
        ruta_cat.parent.mkdir(parents=True, exist_ok=True)

        trabajo = df.copy()
        lat_q = (trabajo["Latitud"] / self.paso_cuadrante).round(0) * self.paso_cuadrante
        lon_q = (trabajo["Longitud"] / self.paso_cuadrante).round(0) * self.paso_cuadrante
        trabajo["_id_cuadrante"] = [
            f"{la:.2f}_{lo:.2f}" for la, lo in zip(lat_q, lon_q)
        ]

        cuadrantes_df = (
            pd.DataFrame(
                {
                    "id_cuadrante": trabajo["_id_cuadrante"],
                    "lat_cuadrante": lat_q.round(2),
                    "long_cuadrante": lon_q.round(2),
                }
            )
            .drop_duplicates(subset=["id_cuadrante"])
            .sort_values(["lat_cuadrante", "long_cuadrante"])
            .reset_index(drop=True)
        )

        catalogo_existente = pd.DataFrame()
        if ruta_cat.exists() and not forzar_regeneracion:
            catalogo_existente = pd.read_csv(ruta_cat)

        ids_existentes = (
            set(catalogo_existente["id_cuadrante"].astype(str))
            if not catalogo_existente.empty and "id_cuadrante" in catalogo_existente.columns
            else set()
        )
        faltantes_df = cuadrantes_df[
            ~cuadrantes_df["id_cuadrante"].isin(ids_existentes)
        ].reset_index(drop=True)

        if not faltantes_df.empty:
            nuevos_df = self._consultar_lotes_nemotron(
                faltantes_df, tamano_lote=tamano_lote
            )
            catalogo_final = (
                pd.concat([catalogo_existente, nuevos_df], ignore_index=True)
                .drop_duplicates(subset=["id_cuadrante"], keep="last")
                .sort_values(["lat_cuadrante", "long_cuadrante"])
                .reset_index(drop=True)
            )
            catalogo_final.to_csv(ruta_cat, index=False, encoding="utf-8")
        else:
            catalogo_final = (
                catalogo_existente.drop_duplicates(subset=["id_cuadrante"], keep="last")
                .sort_values(["lat_cuadrante", "long_cuadrante"])
                .reset_index(drop=True)
            )

        # Validaciones previas y posteriores a la unión (requisito explícito de EA3)
        filas_antes = len(trabajo)
        suma_precio_antes = float(trabajo["Precio (USD)"].sum())
        claves_unicas_catalogo = catalogo_final["id_cuadrante"].is_unique

        columnas_union = [
            "id_cuadrante",
            "Barrio / Sector",
            "Municipio / Subregión",
            "Acceso a transporte",
            "Parques y zonas verdes",
            "Nivel de comercio y servicios",
            "Zonas comunes y entorno",
            "Cuerpo de agua cercano",
        ]
        unido = trabajo.merge(
            catalogo_final[columnas_union],
            left_on="_id_cuadrante",
            right_on="id_cuadrante",
            how="left",
            validate="many_to_one",
        ).drop(columns=["_id_cuadrante", "id_cuadrante"])

        # Clasificación específica de Vista al mar vs. Vista a lago interior a nivel de vivienda
        es_frente_agua = unido["Vista al agua"] == "Sí"
        es_costa_marina = (
            unido["Cuerpo de agua cercano"].str.contains("Puget Sound", na=False)
            | (unido["Longitud"] <= -122.335)
        )
        unido["Vista al mar o lago"] = np.select(
            [
                es_frente_agua & es_costa_marina,
                es_frente_agua & (~es_costa_marina),
            ],
            [
                "Vista al mar (Puget Sound)",
                "Vista a lago / río interior",
            ],
            default="Sin vista directa al agua",
        )

        filas_despues = len(unido)
        suma_precio_despues = float(unido["Precio (USD)"].sum())
        nulos_post_union = int(
            unido[
                [
                    "Barrio / Sector",
                    "Municipio / Subregión",
                    "Acceso a transporte",
                    "Parques y zonas verdes",
                    "Nivel de comercio y servicios",
                    "Zonas comunes y entorno",
                    "Vista al mar o lago",
                ]
            ]
            .isna()
            .sum()
            .sum()
        )

        validacion_df = pd.DataFrame(
            [
                {
                    "Control de unión (LEFT JOIN)": "Clave primaria única en catálogo Nemotron",
                    "Resultado antes": f"{len(cuadrantes_df)} cuadrantes requeridos",
                    "Resultado después": f"{len(catalogo_final)} cuadrantes únicos (is_unique={claves_unicas_catalogo})",
                    "Estado": "OK" if claves_unicas_catalogo else "ERROR",
                },
                {
                    "Control de unión (LEFT JOIN)": "Conservación exacta de registros (sin multiplicación)",
                    "Resultado antes": f"{filas_antes:,} filas",
                    "Resultado después": f"{filas_despues:,} filas",
                    "Estado": "OK" if filas_antes == filas_despues else "ERROR",
                },
                {
                    "Control de unión (LEFT JOIN)": "Conservación exacta del importe total Precio (USD)",
                    "Resultado antes": f"USD {suma_precio_antes:,.0f}",
                    "Resultado después": f"USD {suma_precio_despues:,.0f}",
                    "Estado": (
                        "OK"
                        if abs(suma_precio_antes - suma_precio_despues) < 1e-3
                        else "ERROR"
                    ),
                },
                {
                    "Control de unión (LEFT JOIN)": "Cobertura sin valores nulos en variables enriquecidas",
                    "Resultado antes": "0 nulos esperados",
                    "Resultado después": f"{nulos_post_union} nulos",
                    "Estado": "OK" if nulos_post_union == 0 else "ERROR",
                },
            ]
        )

        return unido, catalogo_final, validacion_df

    def obtener_diccionario_variables_nuevas(self) -> pd.DataFrame:
        """Devuelve la documentación técnica de todas las variables incorporadas en AE3."""
        registros = [
            ("Fecha de venta", "datetime64[ns]", "Fecha ISO", "Construida desde Año, Mes y Día de venta", "Análisis temporal exacto"),
            ("Precio (USD)", "float64", "USD", "exp(Precio (log)) - 1 redondeado", "Variable objetivo en dólares reales"),
            ("Área habitable (m2)", "float64", "m²", "Área habitable (sqft) / 10.7639", "Comparación en Sistema Internacional"),
            ("Área del lote (m2)", "float64", "m²", "Área del lote (sqft) / 10.7639", "Tamaño del terreno en m²"),
            ("Área habitable vecinos (m2)", "float64", "m²", "Área habitable vecinos (sqft) / 10.7639", "Contexto constructivo del vecindario"),
            ("Precio por m2 (USD)", "float64", "USD/m²", "Precio (USD) / Área habitable (m2)", "Métrica estándar de valoración unitaria"),
            ("Periodo de venta", "object", "Año-Mes", "Fecha de venta en formato YYYY-MM", "Evolución mensual de precios y volumen"),
            ("Antigüedad (años)", "int64", "Años", "max(0, Año de venta - Año de construcción)", "Evaluación de depreciación / edad"),
            ("Década de construcción", "object", "Década", "(Año de construcción // 10) * 10 + 's'", "Comparación histórica por cohorte"),
            ("Condición de remodelación", "object", "Categoría", "'Remodelada' si Año de renovación > 0; 'No remodelada' si 0", "Prima de precio por renovación"),
            ("Categoría de tamaño", "object", "Tramo ordinal", "Corte de Área habitable (m2) en <130, 130–199, 200–279, ≥280", "Segmentación por tamaño de vivienda"),
            ("Nivel de calificación", "object", "Tramo ordinal", "Agrupación de Calificación en Baja (4–6), Estándar (7), Media-Alta (8–9), Alta/Lujo (10–13)", "Estabilidad estadística en extremos"),
            ("Distancia Downtown Seattle (km)", "float64", "km", "Fórmula geodésica Haversine hacia (47.6062, -122.3321)", "Cercanía al centro financiero principal"),
            ("Distancia Bellevue Comercial (km)", "float64", "km", "Fórmula geodésica Haversine hacia (47.6101, -122.2015)", "Cercanía al polo comercial/tech del Eastside"),
            ("Distancia Aeropuerto SeaTac (km)", "float64", "km", "Fórmula geodésica Haversine hacia (47.4502, -122.3088)", "Conectividad aeroportuaria regional"),
            ("Distancia polo comercial más cercano (km)", "float64", "km", "mín(Distancia Seattle, Distancia Bellevue)", "Accesibilidad comercial efectiva"),
            ("Anillo urbano", "object", "Tramo ordinal", "Céntrico (<10 km), Suburbano cercano (10–20 km), Periférico (>20 km)", "Gradiente centro-periferia"),
            ("Barrio / Sector", "object", "Texto", "Inferido con NVIDIA Nemotron por cuadrante (0.05° × 0.05°)", "Identificación legible del barrio/ciudad"),
            ("Municipio / Subregión", "object", "Categoría", "Inferido con NVIDIA Nemotron (Seattle, Eastside, South/North King, Islas, Foothills)", "Comparación subregional"),
            ("Acceso a transporte", "object", "Ordinal", "Inferido con NVIDIA Nemotron: Alto, Medio, Bajo", "Dotación vial, tren ligero, buses y ferris"),
            ("Parques y zonas verdes", "object", "Ordinal", "Inferido con NVIDIA Nemotron: Alta, Media, Baja", "Acceso a parques, reservas y frentes verdes"),
            ("Nivel de comercio y servicios", "object", "Ordinal", "Inferido con NVIDIA Nemotron: Alto, Medio, Bajo", "Densidad comercial del entorno"),
            ("Zonas comunes y entorno", "object", "Categoría", "Inferido con NVIDIA Nemotron según vocación urbana/recreativa", "Caracterización de equipamiento vecinal"),
            ("Cuerpo de agua cercano", "object", "Categoría", "Inferido con NVIDIA Nemotron: Puget Sound, Lago interior o Interior", "Contexto hidrográfico del cuadrante"),
            ("Vista al mar o lago", "object", "Categoría", "Combina 'Vista al agua' real con costa marina vs. lago interior", "Diferencia prima costera marina vs. lacustre"),
        ]
        return pd.DataFrame(
            registros,
            columns=["Variable incorporada", "Tipo", "Unidad", "Fórmula o regla", "Utilidad analítica"],
        )

    def _ancla_mas_cercana(self, lat: float, lon: float) -> tuple[str, str, float]:
        """Localiza el centroide urbano real más cercano en King County para anclar el prompt del LLM."""
        anclas_lat = np.array([a[2] for a in self.ANCLAS_KING_COUNTY])
        anclas_lon = np.array([a[3] for a in self.ANCLAS_KING_COUNTY])
        dists = self._haversine_vectorizado(anclas_lat, anclas_lon, lat, lon)
        # Vashon Island está separada por el estrecho marino de Puget Sound (solo aplica al oeste del canal)
        for idx, (nombre, _, _, _) in enumerate(self.ANCLAS_KING_COUNTY):
            if nombre == "Vashon Island" and not (47.32 <= lat <= 47.52 and lon <= -122.38):
                dists[idx] = 999.0
        idx_min = int(np.argmin(dists))
        nombre, subregion, _, _ = self.ANCLAS_KING_COUNTY[idx_min]
        return nombre, subregion, round(float(dists[idx_min]), 1)

    def _consultar_lotes_nemotron(
        self, cuadrantes_df: pd.DataFrame, tamano_lote: int = 12
    ) -> pd.DataFrame:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No se encontró NVIDIA_API_KEY en el entorno ni en .env, y faltan cuadrantes en el catálogo."
            )

        cliente = OpenAI(base_url=self.base_url, api_key=api_key)
        modelos_orden = [self.modelo] + [
            m for m in self.MODELOS_RESPALDO if m != self.modelo
        ]

        resultados: list[dict[str, Any]] = []
        total = len(cuadrantes_df)
        tam_efectivo = min(tamano_lote, 12)

        for inicio in range(0, total, tam_efectivo):
            lote = cuadrantes_df.iloc[inicio : inicio + tam_efectivo]
            lats = lote["lat_cuadrante"].to_numpy(dtype=float)
            lons = lote["long_cuadrante"].to_numpy(dtype=float)
            d_sea = self._haversine_vectorizado(lats, lons, 47.6062, -122.3321)
            d_bel = self._haversine_vectorizado(lats, lons, 47.6101, -122.2015)

            items_entrada = []
            anclas_lote: dict[str, tuple[str, str]] = {}
            for (_, fila), ds, db in zip(lote.iterrows(), d_sea, d_bel):
                cid = str(fila["id_cuadrante"])
                la = round(float(fila["lat_cuadrante"]), 2)
                lo = round(float(fila["long_cuadrante"]), 2)
                ancla_nom, ancla_sub, dist_ancla = self._ancla_mas_cercana(la, lo)
                anclas_lote[cid] = (ancla_nom, ancla_sub)
                items_entrada.append(
                    {
                        "id_cuadrante": cid,
                        "lat": la,
                        "lon": lo,
                        "referencia_cercana": f"{ancla_nom} ({dist_ancla} km)",
                        "subregion_base": ancla_sub,
                        "dist_seattle_km": round(float(ds), 1),
                        "dist_bellevue_km": round(float(db), 1),
                    }
                )

            respuesta_lote = self._llamar_llm_con_reintento(
                cliente, modelos_orden, items_entrada
            )
            mapa_resp = {
                str(item.get("id_cuadrante", "")): item for item in respuesta_lote
            }

            for _, fila in lote.iterrows():
                cid = str(fila["id_cuadrante"])
                la = float(fila["lat_cuadrante"])
                lo = float(fila["long_cuadrante"])
                bruto = mapa_resp.get(cid, {})
                ancla_nom, ancla_sub = anclas_lote[cid]
                if not bruto.get("barrio_sector") or (
                    "vashon" in str(bruto.get("barrio_sector", "")).lower()
                    and lo > -122.38
                ):
                    bruto["barrio_sector"] = ancla_nom
                    bruto["municipio_subregion"] = ancla_sub
                resultados.append(self._normalizar_registro_cuadrante(cid, la, lo, bruto))

            time.sleep(0.2)

        return pd.DataFrame(resultados)

    def _llamar_llm_con_reintento(
        self,
        cliente: OpenAI,
        modelos: list[str],
        items_entrada: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt_sistema = (
            "Eres un analista urbano e inmobiliario experto en King County, Washington (EE. UU.). "
            "Usa la coordenada (lat, lon) y la referencia_cercana de cada punto para caracterizar su equipamiento urbano real "
            "(transporte, parques, comercio, zonas comunes y cuerpo de agua). "
            "Evalúa cada id_cuadrante individualmente. Responde ÚNICAMENTE un arreglo JSON válido."
        )
        prompt_usuario = (
            "Para cada cuadrante recibido, devuelve un objeto JSON con EXACTAMENTE estas claves:\n"
            "- id_cuadrante: el mismo string recibido.\n"
            "- barrio_sector: nombre conciso del barrio o municipio en King County acorde con referencia_cercana.\n"
            "- municipio_subregion: uno de ['Seattle', 'Eastside (Bellevue/Kirkland/Redmond)', 'North King County', 'South King County', 'Foothills / Este Semirrural', 'Vashon / Islas'].\n"
            "- acceso_transporte: uno de ['Alto', 'Medio', 'Bajo'] (conectividad I-5/I-90/I-405/SR-520, Link Light Rail, buses King County Metro, ferris).\n"
            "- parques_zonas_verdes: uno de ['Alta', 'Media', 'Baja'] (cercanía a parques urbanos, lagos, reservas o senderos).\n"
            "- nivel_comercio: uno de ['Alto', 'Medio', 'Bajo'] (densidad de centros comerciales, restaurantes y servicios).\n"
            "- zonas_comunes_entorno: uno de ['Urbano denso y comercial', 'Suburbano residencial familiar', 'Frente costero / lacustre recreativo', 'Semirrural / Boscoso'].\n"
            "- cuerpo_agua_cercano: uno de ['Estrecho marino (Puget Sound)', 'Lago interior (Washington/Sammamish/Union)', 'Río / Interior sin costa'].\n\n"
            f"Cuadrantes:\n{json.dumps(items_entrada, ensure_ascii=False)}"
        )

        ultimo_error: Exception | None = None
        for modelo_actual in modelos:
            for intento in range(2):
                try:
                    resp = cliente.chat.completions.create(
                        model=modelo_actual,
                        messages=[
                            {"role": "system", "content": prompt_sistema},
                            {"role": "user", "content": prompt_usuario},
                        ],
                        temperature=0.15,
                        max_tokens=3000,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    texto = resp.choices[0].message.content or ""
                    return self._extraer_json_lista(texto)
                except Exception as exc:  # noqa: BLE001
                    ultimo_error = exc
                    time.sleep(0.8 + intento)

        raise RuntimeError(
            f"No fue posible obtener respuesta JSON válida desde NVIDIA Nemotron: {ultimo_error}"
        )

    @staticmethod
    def _extraer_json_lista(texto: str) -> list[dict[str, Any]]:
        limpio = texto.strip()
        limpio = re.sub(r"^```(?:json)?\s*", "", limpio)
        limpio = re.sub(r"\s*```$", "", limpio)
        inicio = limpio.find("[")
        fin = limpio.rfind("]")
        if inicio == -1 or fin == -1 or fin <= inicio:
            raise ValueError(f"La respuesta no contiene un arreglo JSON: {limpio[:200]}")
        datos = json.loads(limpio[inicio : fin + 1])
        if not isinstance(datos, list):
            raise ValueError("El JSON parseado no es una lista.")
        return datos

    def _normalizar_registro_cuadrante(
        self, cid: str, lat: float, lon: float, bruto: dict[str, Any]
    ) -> dict[str, Any]:
        barrio = str(bruto.get("barrio_sector", "")).strip() or f"Sector ({lat:.2f}, {lon:.2f})"

        subregion = str(bruto.get("municipio_subregion", "")).strip()
        subregiones_validas = {
            "Seattle",
            "Eastside (Bellevue/Kirkland/Redmond)",
            "North King County",
            "South King County",
            "Foothills / Este Semirrural",
            "Vashon / Islas",
        }
        if subregion not in subregiones_validas:
            if lon < -122.42:
                subregion = "Vashon / Islas"
            elif lon < -122.24 and 47.50 <= lat <= 47.74:
                subregion = "Seattle"
            elif -122.24 <= lon <= -121.98 and lat >= 47.52:
                subregion = "Eastside (Bellevue/Kirkland/Redmond)"
            elif lon > -121.98:
                subregion = "Foothills / Este Semirrural"
            elif lat > 47.70:
                subregion = "North King County"
            else:
                subregion = "South King County"

        transporte = str(bruto.get("acceso_transporte", "Medio")).strip().capitalize()
        if transporte not in self.VALORES_NIVEL:
            transporte = "Medio"

        parques = str(bruto.get("parques_zonas_verdes", "Media")).strip().capitalize()
        if parques not in self.VALORES_PARQUES:
            parques = "Media"

        comercio = str(bruto.get("nivel_comercio", "Medio")).strip().capitalize()
        if comercio not in self.VALORES_NIVEL:
            comercio = "Medio"

        entorno = str(bruto.get("zonas_comunes_entorno", "")).strip()
        if entorno not in self.VALORES_ENTORNO:
            entorno = "Suburbano residencial familiar"

        agua = str(bruto.get("cuerpo_agua_cercano", "")).strip()
        if agua not in self.VALORES_AGUA:
            agua = (
                "Estrecho marino (Puget Sound)"
                if lon <= -122.35
                else "Río / Interior sin costa"
            )

        return {
            "id_cuadrante": cid,
            "lat_cuadrante": lat,
            "long_cuadrante": lon,
            "Barrio / Sector": barrio,
            "Municipio / Subregión": subregion,
            "Acceso a transporte": transporte,
            "Parques y zonas verdes": parques,
            "Nivel de comercio y servicios": comercio,
            "Zonas comunes y entorno": entorno,
            "Cuerpo de agua cercano": agua,
            "Modelo LLM": self.modelo,
        }

    @classmethod
    def _haversine_vectorizado(
        cls,
        lat1: np.ndarray,
        lon1: np.ndarray,
        lat2: float,
        lon2: float,
    ) -> np.ndarray:
        rad_lat1 = np.radians(lat1)
        rad_lon1 = np.radians(lon1)
        rad_lat2 = np.radians(lat2)
        rad_lon2 = np.radians(lon2)

        dlat = rad_lat2 - rad_lat1
        dlon = rad_lon2 - rad_lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(rad_lat1) * np.cos(rad_lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arcsin(np.sqrt(a))
        return cls.RADIO_TIERRA_KM * c
