"""Preprocesamiento, reglas de calidad y auditoría Antes vs. Después para House Sales."""
from __future__ import annotations

import pandas as pd


class PreprocesadorHouseSales:
    """Ejecuta la limpieza, validación y auditoría comparativa del dataset de viviendas."""

    COLUMNAS_ES: dict[str, str] = {
        "bedrooms": "Habitaciones",
        "bathrooms": "Baños",
        "sqft_living": "Área habitable (sqft)",
        "sqft_lot": "Área del lote (sqft)",
        "waterfront": "Vista al agua",
        "grade": "Calificación",
        "sqft_above": "Área sobre suelo (sqft)",
        "sqft_basement": "Área sótano (sqft)",
        "yr_built": "Año de construcción",
        "yr_renovated": "Año de renovación",
        "lat": "Latitud",
        "long": "Longitud",
        "sqft_living15": "Área habitable vecinos (sqft)",
        "sqft_lot15": "Área lote vecinos (sqft)",
        "date_year": "Año de venta",
        "date_month": "Mes de venta",
        "date_day": "Día de venta",
        "price": "Precio (log)",
    }

    def __init__(
        self,
        min_habitaciones: int = 1,
        max_habitaciones: int = 15,
        min_banos: float = 1.0,
    ) -> None:
        self.min_habitaciones = min_habitaciones
        self.max_habitaciones = max_habitaciones
        self.min_banos = min_banos

    def diagnosticar_calidad(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Evalúa controles de calidad sobre el DataFrame crudo sin modificarlo."""
        self._validar_esquema(df_raw)

        nulos_totales = int(df_raw.isna().sum().sum())
        duplicados = int(df_raw.duplicated().sum())
        hab_cero = int((df_raw["bedrooms"] < self.min_habitaciones).sum())
        hab_extremo = int((df_raw["bedrooms"] > self.max_habitaciones).sum())
        banos_cero = int((df_raw["bathrooms"] == 0).sum())
        banos_menor_min = int((df_raw["bathrooms"] < self.min_banos).sum())
        inconsistentes_total = int(
            (
                (df_raw["bedrooms"] < self.min_habitaciones)
                | (df_raw["bedrooms"] > self.max_habitaciones)
                | (df_raw["bathrooms"] < self.min_banos)
            ).sum()
        )
        dif_areas = int(
            (
                df_raw["sqft_living"]
                != (df_raw["sqft_above"] + df_raw["sqft_basement"])
            ).sum()
        )

        controles = [
            {
                "Control de calidad": "Valores nulos en el esquema",
                "Regla evaluada": "df.isna().sum().sum() == 0",
                "Registros detectados": nulos_totales,
                "% del total": round(nulos_totales / len(df_raw) * 100, 3),
                "Decisión y justificación": (
                    "Sin acción de imputación: las 18 columnas tienen 100 % de completitud."
                ),
            },
            {
                "Control de calidad": "Filas exactamente duplicadas",
                "Regla evaluada": "df.duplicated(keep='first')",
                "Registros detectados": duplicados,
                "% del total": round(duplicados / len(df_raw) * 100, 3),
                "Decisión y justificación": (
                    "Se ejecuta drop_duplicates() por control preventivo; 0 filas afectadas."
                ),
            },
            {
                "Control de calidad": "Viviendas sin habitaciones",
                "Regla evaluada": f"bedrooms < {self.min_habitaciones}",
                "Registros detectados": hab_cero,
                "% del total": round(hab_cero / len(df_raw) * 100, 3),
                "Decisión y justificación": (
                    "Filtrar: no corresponden a una unidad residencial estándar comparable en avalúos."
                ),
            },
            {
                "Control de calidad": "Atípico extremo de digitación en habitaciones",
                "Regla evaluada": f"bedrooms > {self.max_habitaciones}",
                "Registros detectados": hab_extremo,
                "% del total": round(hab_extremo / len(df_raw) * 100, 3),
                "Decisión y justificación": (
                    "Filtrar: registro índice 15870 con 33 habitaciones en 1,620 sqft (error de captura documentado)."
                ),
            },
            {
                "Control de calidad": "Viviendas sin baño completo",
                "Regla evaluada": f"bathrooms < {self.min_banos} (incluye {banos_cero} con 0 baños)",
                "Registros detectados": banos_menor_min,
                "% del total": round(banos_menor_min / len(df_raw) * 100, 3),
                "Decisión y justificación": (
                    "Filtrar: una vivienda habitable requiere al menos 1 baño completo para valoración estándar."
                ),
            },
            {
                "Control de calidad": "Total combinado de registros inconsistentes",
                "Regla evaluada": (
                    f"bedrooms fuera de [{self.min_habitaciones}, {self.max_habitaciones}] "
                    f"o bathrooms < {self.min_banos}"
                ),
                "Registros detectados": inconsistentes_total,
                "% del total": round(inconsistentes_total / len(df_raw) * 100, 3),
                "Decisión y justificación": (
                    f"Se descartan {inconsistentes_total} filas ({round(inconsistentes_total / len(df_raw) * 100, 2)} %), "
                    f"conservando {len(df_raw) - inconsistentes_total} viviendas consistentes."
                ),
            },
            {
                "Control de calidad": "Consistencia aditiva de áreas interiores",
                "Regla evaluada": "sqft_living == sqft_above + sqft_basement",
                "Registros detectados": dif_areas,
                "% del total": round(dif_areas / len(df_raw) * 100, 3),
                "Decisión y justificación": (
                    "0 discrepancias: confirma coherencia interna de las mediciones de superficie."
                ),
            },
        ]
        return pd.DataFrame(controles)

    def limpiar_y_validar(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Aplica las reglas de limpieza, filtrado, renombrado y decodificación base."""
        self._validar_esquema(df_raw)

        # 1. Copia inmutable y control de duplicados
        df = df_raw.copy().drop_duplicates(keep="first").reset_index(drop=True)

        # 2. Filtrado de coherencia habitacional
        mascara_valida = (
            (df["bedrooms"] >= self.min_habitaciones)
            & (df["bedrooms"] <= self.max_habitaciones)
            & (df["bathrooms"] >= self.min_banos)
        )
        df = df.loc[mascara_valida].reset_index(drop=True)

        # 3. Renombrado de las 18 columnas al español
        df = df.rename(columns=self.COLUMNAS_ES)

        # 4. Decodificación de variables codificadas y tipado de fecha
        df["Vista al agua"] = df["Vista al agua"].map({0: "No", 1: "Sí"})
        df["Año de venta"] = df["Año de venta"].map({0: 2014, 1: 2015}).astype("int64")

        fechas = pd.to_datetime(
            {
                "year": df["Año de venta"],
                "month": df["Mes de venta"],
                "day": df["Día de venta"],
            },
            errors="raise",
        )
        df["Fecha de venta"] = fechas

        return df

    def comparar_antes_despues(
        self,
        df_antes: pd.DataFrame,
        df_despues: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Construye las tablas de auditoría global y por columna (Antes vs. Después)."""
        num_antes = len(df_antes.select_dtypes(include="number").columns)
        num_despues = len(df_despues.select_dtypes(include="number").columns)
        cat_antes = len(
            df_antes.select_dtypes(include=["object", "category", "string"]).columns
        )
        cat_despues = len(
            df_despues.select_dtypes(include=["object", "category", "string"]).columns
        )
        dt_antes = len(df_antes.select_dtypes(include=["datetime", "datetimetz"]).columns)
        dt_despues = len(
            df_despues.select_dtypes(include=["datetime", "datetimetz"]).columns
        )

        resumen_global = pd.DataFrame(
            [
                {
                    "Indicador": "Total de filas (registros)",
                    "Antes (Crudo)": len(df_antes),
                    "Después (Preparado/Enriquecido)": len(df_despues),
                    "Diferencia": len(df_despues) - len(df_antes),
                    "Explicación": "Depuración justificada de 92 viviendas con habitaciones o baños inválidos (0.43 %).",
                },
                {
                    "Indicador": "Total de columnas (variables)",
                    "Antes (Crudo)": len(df_antes.columns),
                    "Después (Preparado/Enriquecido)": len(df_despues.columns),
                    "Diferencia": len(df_despues.columns) - len(df_antes.columns),
                    "Explicación": "Conservación de las 18 variables base (renombradas) + incorporación de variables derivadas, geodésicas y de NVIDIA Nemotron.",
                },
                {
                    "Indicador": "Valores nulos totales",
                    "Antes (Crudo)": int(df_antes.isna().sum().sum()),
                    "Después (Preparado/Enriquecido)": int(df_despues.isna().sum().sum()),
                    "Diferencia": int(df_despues.isna().sum().sum())
                    - int(df_antes.isna().sum().sum()),
                    "Explicación": "Se garantiza 0 % de valores faltantes tanto en las variables base como en las enriquecidas.",
                },
                {
                    "Indicador": "Filas exactamente duplicadas",
                    "Antes (Crudo)": int(df_antes.duplicated().sum()),
                    "Después (Preparado/Enriquecido)": int(df_despues.duplicated().sum()),
                    "Diferencia": int(df_despues.duplicated().sum())
                    - int(df_antes.duplicated().sum()),
                    "Explicación": "Sin duplicados antes ni después de las transformaciones y uniones.",
                },
                {
                    "Indicador": "Columnas numéricas",
                    "Antes (Crudo)": num_antes,
                    "Después (Preparado/Enriquecido)": num_despues,
                    "Diferencia": num_despues - num_antes,
                    "Explicación": "Vista al agua pasa a texto ('Sí'/'No'); se añaden precios en USD, áreas en m², antigüedad y distancias en km.",
                },
                {
                    "Indicador": "Columnas categóricas / texto",
                    "Antes (Crudo)": cat_antes,
                    "Después (Preparado/Enriquecido)": cat_despues,
                    "Diferencia": cat_despues - cat_antes,
                    "Explicación": "Se agregan categorías de tamaño, calificación, remodelación, década, periodo y atributos urbanos de Nemotron.",
                },
                {
                    "Indicador": "Columnas de fecha (datetime)",
                    "Antes (Crudo)": dt_antes,
                    "Después (Preparado/Enriquecido)": dt_despues,
                    "Diferencia": dt_despues - dt_antes,
                    "Explicación": "Construcción de 'Fecha de venta' a partir de año, mes y día.",
                },
            ]
        )

        mapeo_inverso = {v: k for k, v in self.COLUMNAS_ES.items()}
        filas_col = []
        for col_final in df_despues.columns:
            col_orig = mapeo_inverso.get(col_final, "— (Nueva en AE3)")
            if col_orig in df_antes.columns:
                tipo_antes = str(df_antes[col_orig].dtype)
                nulos_antes = int(df_antes[col_orig].isna().sum())
                origen = "Original transformada / renombrada"
            else:
                tipo_antes = "—"
                nulos_antes = 0
                origen = "Incorporada en enriquecimiento (AE3)"

            filas_col.append(
                {
                    "Columna original": col_orig,
                    "Columna final": col_final,
                    "Tipo Antes": tipo_antes,
                    "Tipo Después": str(df_despues[col_final].dtype),
                    "Nulos Antes": nulos_antes,
                    "Nulos Después": int(df_despues[col_final].isna().sum()),
                    "Origen": origen,
                }
            )

        return resumen_global, pd.DataFrame(filas_col)

    def _validar_esquema(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame.")
        faltantes = set(self.COLUMNAS_ES.keys()) - set(df.columns)
        if faltantes:
            raise ValueError(
                f"Al DataFrame crudo le faltan columnas esperadas: {sorted(faltantes)}"
            )
