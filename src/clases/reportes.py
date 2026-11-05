"""Generación de reportes descriptivos, tablas agrupadas y auditoría de DataFrames."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class Reportes:
    """Imprime, exporta tablas analíticas y guarda resúmenes legibles de un DataFrame."""

    def generar_reporte(
        self,
        df: pd.DataFrame,
        dataset_name: str | None = None,
        config: str | None = None,
        split: str | None = None,
        muestra: str | None = None,
        incluir_categoricas: bool = True,
        top_categoricas: int = 10,
        incluir_timestamp: bool = False,
    ) -> str:
        """Genera el contenido textual del reporte descriptivo del DataFrame."""
        self._validar_dataframe(df)
        numericas = df.select_dtypes(include="number")
        categoricas = df.select_dtypes(include=["object", "category", "string"])

        lineas = [
            "REPORTE DEL DATAFRAME",
            "=" * 80,
        ]

        if incluir_timestamp:
            lineas.append(
                f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if dataset_name is not None:
            lineas.append(f"Dataset: {dataset_name}")
        if config is not None:
            lineas.append(f"Configuración: {config}")
        if split is not None:
            lineas.append(f"Split: {split}")
        if muestra is not None:
            lineas.append(f"Muestra / Cobertura: {muestra}")

        total_filas = len(df)
        lineas += [
            f"Filas: {total_filas}",
            f"Columnas: {len(df.columns)}",
            f"Duplicados: {int(df.duplicated().sum())}",
            "",
            "Columnas y tipos:",
            *[f"- {columna}: {tipo}" for columna, tipo in df.dtypes.items()],
            "",
            "Valores nulos por columna (cantidad y porcentaje):",
            *[
                f"- {columna}: {cantidad} ({(cantidad / total_filas * 100) if total_filas else 0.0:.2f}%)"
                for columna, cantidad in df.isna().sum().items()
            ],
        ]

        if not numericas.empty:
            tabla_num = (
                numericas.describe().T.round(3)
                if len(numericas.columns) > 10
                else numericas.describe().round(3)
            )
            lineas += [
                "",
                "Estadísticas numéricas:",
                tabla_num.to_string(),
            ]

        if incluir_categoricas and not categoricas.empty:
            lineas += [
                "",
                "Resumen de variables categóricas (frecuencias principales):",
            ]
            for col in categoricas.columns:
                unicos = int(df[col].nunique(dropna=True))
                conteos = df[col].value_counts(dropna=False).head(top_categoricas)
                porcentajes = (conteos / total_filas * 100).round(2)
                resumen_cat = pd.DataFrame(
                    {"Frecuencia": conteos, "Porcentaje (%)": porcentajes}
                )
                lineas.append(f"\n[{col}] (Valores únicos: {unicos})")
                lineas.append(resumen_cat.to_string())

        return "\n".join(lineas) + "\n"

    def imprimir_reporte(
        self,
        df: pd.DataFrame,
        dataset_name: str | None = None,
        config: str | None = None,
        split: str | None = None,
        muestra: str | None = None,
        incluir_categoricas: bool = True,
        top_categoricas: int = 10,
        incluir_timestamp: bool = False,
    ) -> None:
        """Imprime en la consola el reporte del DataFrame."""
        print(
            self.generar_reporte(
                df,
                dataset_name=dataset_name,
                config=config,
                split=split,
                muestra=muestra,
                incluir_categoricas=incluir_categoricas,
                top_categoricas=top_categoricas,
                incluir_timestamp=incluir_timestamp,
            ),
            end="",
        )

    def guardar_reporte(
        self,
        df: pd.DataFrame,
        ruta: str | Path,
        dataset_name: str | None = None,
        config: str | None = None,
        split: str | None = None,
        muestra: str | None = None,
        incluir_categoricas: bool = True,
        top_categoricas: int = 10,
        incluir_timestamp: bool = False,
    ) -> Path:
        """Guarda el reporte en un archivo de texto UTF-8 y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        reporte = self.generar_reporte(
            df,
            dataset_name=dataset_name,
            config=config,
            split=split,
            muestra=muestra,
            incluir_categoricas=incluir_categoricas,
            top_categoricas=top_categoricas,
            incluir_timestamp=incluir_timestamp,
        )
        path.write_text(reporte, encoding="utf-8")
        return path

    def tabla_estadistica_grupo(
        self,
        df: pd.DataFrame,
        col_grupo: str | list[str],
        col_valor: str = "Precio (USD)",
        col_secundaria: str | None = "Precio por m2 (USD)",
        ordenar_por: str | None = None,
        ascendente: bool = True,
        top_n: int | None = None,
        ruta_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Calcula estadísticas descriptivas completas por grupo para responder preguntas de negocio.

        Incluye: n, % participación, media, mediana, moda, desviación estándar,
        coeficiente de variación (%), Q1, Q3, RIC, mínimo, máximo y mediana unitaria (USD/m²).
        """
        self._validar_dataframe(df)
        cols_agrupacion = [col_grupo] if isinstance(col_grupo, str) else list(col_grupo)
        total = len(df)

        filas: list[dict[str, object]] = []
        agrupado = df.groupby(cols_agrupacion, dropna=False, sort=True)

        for clave, sub in agrupado:
            valores = sub[col_valor].dropna()
            if valores.empty:
                continue

            claves_tupla = clave if isinstance(clave, tuple) else (clave,)
            registro: dict[str, object] = {
                nombre_col: valor_clave
                for nombre_col, valor_clave in zip(cols_agrupacion, claves_tupla)
            }

            n = len(valores)
            media = float(valores.mean())
            mediana = float(valores.median())
            modas = valores.mode()
            moda = float(modas.iloc[0]) if not modas.empty else np.nan
            desv = float(valores.std(ddof=1)) if n > 1 else 0.0
            cv = (desv / media * 100.0) if media != 0 else 0.0
            q1 = float(valores.quantile(0.25))
            q3 = float(valores.quantile(0.75))
            ric = q3 - q1

            registro.update(
                {
                    "Viviendas (n)": int(n),
                    "Participación (%)": round(n / total * 100.0, 2),
                    "Media (USD)": round(media, 0),
                    "Mediana (USD)": round(mediana, 0),
                    "Moda (USD)": round(moda, 0),
                    "Desv. Estándar (USD)": round(desv, 0),
                    "Coef. Variación (%)": round(cv, 2),
                    "Q1 (USD)": round(q1, 0),
                    "Q3 (USD)": round(q3, 0),
                    "RIC (USD)": round(ric, 0),
                    "Mínimo (USD)": round(float(valores.min()), 0),
                    "Máximo (USD)": round(float(valores.max()), 0),
                }
            )

            if col_secundaria and col_secundaria in sub.columns:
                sec = sub[col_secundaria].dropna()
                if not sec.empty:
                    registro["Mediana (USD/m²)"] = round(float(sec.median()), 2)
                    registro["RIC (USD/m²)"] = round(
                        float(sec.quantile(0.75) - sec.quantile(0.25)), 2
                    )

            filas.append(registro)

        resultado = pd.DataFrame(filas)
        if ordenar_por and ordenar_por in resultado.columns:
            resultado = resultado.sort_values(
                by=ordenar_por, ascending=ascendente
            ).reset_index(drop=True)

        if top_n is not None:
            resultado = resultado.head(top_n).reset_index(drop=True)

        if ruta_csv is not None:
            destino = Path(ruta_csv)
            destino.parent.mkdir(parents=True, exist_ok=True)
            resultado.to_csv(destino, index=False, encoding="utf-8")

        return resultado

    def reporte_descriptivo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera una tabla de auditoría y metadata del DataFrame con tipos, longitudes, nulos y límites."""
        self._validar_dataframe(df)
        filas_meta = []

        for col in df.columns:
            serie = df[col]
            es_numerica = pd.api.types.is_numeric_dtype(serie)

            longitudes = serie.dropna().astype(str).str.len()
            long_min = int(longitudes.min()) if not longitudes.empty else 0
            long_max = int(longitudes.max()) if not longitudes.empty else 0

            if es_numerica:
                q1 = serie.quantile(0.25)
                q3 = serie.quantile(0.75)
                iqr = q3 - q1
                lim_inf_iqr = q1 - 1.5 * iqr
                lim_sup_iqr = q3 + 1.5 * iqr
                atipicos = int(
                    ((serie < lim_inf_iqr) | (serie > lim_sup_iqr)).sum()
                )

                minimo = round(float(serie.min()), 4)
                maximo = round(float(serie.max()), 4)
                media = round(float(serie.mean()), 4)
                mediana = round(float(serie.median()), 4)
                lim_inf = round(float(lim_inf_iqr), 4)
                lim_sup = round(float(lim_sup_iqr), 4)
            else:
                minimo = None
                maximo = None
                media = None
                mediana = None
                lim_inf = None
                lim_sup = None
                atipicos = None

            filas_meta.append(
                {
                    "Columna": col,
                    "Tipo": str(serie.dtype),
                    "No Nulos": int(serie.notna().sum()),
                    "Nulos": int(serie.isna().sum()),
                    "Únicos": int(serie.nunique(dropna=True)),
                    "Long. Mín (caract.)": long_min,
                    "Long. Máx (caract.)": long_max,
                    "Límite Mín": minimo,
                    "Mediana": mediana,
                    "Media": media,
                    "Límite Máx": maximo,
                    "Límite Inf (IQR)": lim_inf,
                    "Límite Sup (IQR)": lim_sup,
                    "Atípicos (IQR)": atipicos,
                }
            )

        return pd.DataFrame(filas_meta)

    def generar_reporte_metadata(
        self,
        df: pd.DataFrame,
        dataset_name: str | None = None,
        config: str | None = None,
        split: str | None = None,
    ) -> str:
        """Genera el reporte textual de auditoría de metadatos del DataFrame."""
        self._validar_dataframe(df)
        metadata_df = self.reporte_descriptivo_data(df)
        lineas = [
            "REPORTE DE AUDITORÍA DE METADATOS DEL DATAFRAME",
            "=" * 80,
        ]

        if dataset_name is not None:
            lineas.append(f"Dataset: {dataset_name}")
        if config is not None:
            lineas.append(f"Configuración: {config}")
        if split is not None:
            lineas.append(f"Split: {split}")

        lineas += [
            f"Filas: {len(df)}",
            f"Columnas: {len(df.columns)}",
            f"Duplicados: {int(df.duplicated().sum())}",
            "",
            "Tabla de auditoría de metadatos (tipos, longitudes, límites y atípicos IQR):",
            metadata_df.to_string(index=False),
        ]

        return "\n".join(lineas) + "\n"

    def guardar_reporte_metadata(
        self,
        df: pd.DataFrame,
        ruta: str | Path,
        dataset_name: str | None = None,
        config: str | None = None,
        split: str | None = None,
    ) -> Path:
        """Guarda el reporte de auditoría de metadatos en un archivo .txt y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        reporte = self.generar_reporte_metadata(
            df,
            dataset_name=dataset_name,
            config=config,
            split=split,
        )
        path.write_text(reporte, encoding="utf-8")
        return path

    @staticmethod
    def _validar_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
