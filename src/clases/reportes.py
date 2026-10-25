"""Generación de reportes de DataFrames."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class Reportes:
    """Imprime y guarda un resumen legible de un DataFrame."""

    def generar_reporte(
        self,
        df: pd.DataFrame,
        dataset_name: str | None = None,
        config: str | None = None,
        split: str | None = None,
        muestra: str | None = None,
    ) -> str:
        """Genera el contenido textual del reporte."""
        self._validar_dataframe(df)
        numericas = df.select_dtypes(include="number")
        lineas = [
            "REPORTE DEL DATAFRAME",
            "=" * 60,
        ]

        if dataset_name is not None:
            lineas.append(f"Dataset: {dataset_name}")
        if config is not None:
            lineas.append(f"Configuración: {config}")
        if split is not None:
            lineas.append(f"Split: {split}")
        if muestra is not None:
            lineas.append(f"Muestra / Cobertura: {muestra}")

        lineas += [
            f"Filas: {len(df)}",
            f"Columnas: {len(df.columns)}",
            f"Duplicados: {int(df.duplicated().sum())}",
            "",
            "Columnas y tipos:",
            *[f"- {columna}: {tipo}" for columna, tipo in df.dtypes.items()],
            "",
            "Valores nulos por columna:",
            *[f"- {columna}: {cantidad}" for columna,
                cantidad in df.isna().sum().items()],
        ]

        if not numericas.empty:
            lineas += [
                "",
                "Estadísticas numéricas:",
                numericas.describe().round(3).to_string(),
            ]

        return "\n".join(lineas) + "\n"

    def imprimir_reporte(
        self,
        df: pd.DataFrame,
        dataset_name: str | None = None,
        config: str | None = None,
        split: str | None = None,
        muestra: str | None = None,
    ) -> None:
        """Imprime en la consola el reporte del DataFrame."""
        print(
            self.generar_reporte(
                df,
                dataset_name=dataset_name,
                config=config,
                split=split,
                muestra=muestra,
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
    ) -> Path:
        """Guarda el reporte en un archivo de texto y devuelve su ruta."""
        path = Path(ruta)
        path.parent.mkdir(parents=True, exist_ok=True)
        reporte = self.generar_reporte(
            df,
            dataset_name=dataset_name,
            config=config,
            split=split,
            muestra=muestra,
        )
        path.write_text(reporte, encoding="utf-8")
        return path

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

            filas_meta.append({
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
            })

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
