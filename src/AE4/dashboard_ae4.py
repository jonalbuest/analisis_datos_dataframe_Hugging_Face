"""Dashboard de la AE4: precios de vivienda en King County con Streamlit + Plotly.

La clase :class:`DashboardAE4` reutiliza lo construido en las entregas anteriores:

* el dataset preparado y enriquecido de AE3 (21.521 viviendas, 43 variables);
* ``Reportes.tabla_estadistica_grupo`` (``src/clases/reportes.py``), la misma función que generó las
  tablas P1–P6 de AE3, para las tablas estadísticas de cada pregunta;
* la auditoría antes vs. después (``PreprocesadorHouseSales``) y el diccionario de variables
  (``EnriquecedorNemotron``), que el notebook de AE4 exporta a ``src/AE4/datos/``.

Flujo de trabajo (igual que en el curso): el notebook ``src/AE4/notebooks/visualizacion_ae4.ipynb``
construye y revisa cada gráfico con esta clase; la app (``src/AE4/app.py``) usa la misma clase
para publicar el dashboard. Así las cifras del notebook y de la app son las mismas.

Uso en Jupyter::

    dash = DashboardAE4.desde_carpeta("src/AE4/datos")
    dash.kpis(dash.datos)
    dash.figura_p1_tamano(dash.datos).show()
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Reuso de AE2/AE3: se importa solo el módulo de reportes (pandas y numpy), sin cargar todo el paquete
# (así la app no necesita openai, datasets ni python-dotenv).
RUTA_CLASES = Path(__file__).resolve().parents[1] / "clases"
if str(RUTA_CLASES) not in sys.path:
    sys.path.insert(0, str(RUTA_CLASES))
from reportes import Reportes  # noqa: E402

ARCHIVO_DATOS = "reg_cat_house_sales_enriquecido.csv"
ARCHIVO_AUDITORIA = "auditoria_global.csv"
ARCHIVO_AUDITORIA_COLUMNAS = "auditoria_columnas.csv"
ARCHIVO_DICCIONARIO = "diccionario_variables.csv"

N_MINIMO_BARRIO = 50          # mismo criterio que la tabla P4 de AE3
UMBRAL_BIEN_CALIFICADA = 8    # estratos Media-Alta (8–9) y Alta/Lujo (10–13)

ORDEN_TAMANOS = ["1. Pequeña (<130 m²)", "2. Mediana (130–199 m²)", "3. Grande (200–279 m²)", "4. Muy grande (≥280 m²)"]
ORDEN_CALIFICACION = ["1. Baja (4–6)", "2. Estándar (7)", "3. Media-Alta (8–9)", "4. Alta / Lujo (10–13)"]
ORDEN_VISTA = ["Sin vista directa al agua", "Vista a lago / río interior", "Vista al mar (Puget Sound)"]
ORDEN_ANILLO = ["1. Céntrico (<10 km)", "2. Suburbano cercano (10–20 km)", "3. Periférico (>20 km)"]
ORDEN_BANOS = ["1", "1,5 a 2", "2,25 a 3", "3,25 o más"]
ORDEN_NIVEL = ["Bajo", "Medio", "Alto", "Media", "Alta"]
COLORES_CALIFICACION = {ORDEN_CALIFICACION[0]: "#9ecae1", ORDEN_CALIFICACION[1]: "#6baed6",
                        ORDEN_CALIFICACION[2]: "#2171b5", ORDEN_CALIFICACION[3]: "#08306b"}
COLORES_VISTA = {ORDEN_VISTA[0]: "#bdbdbd", ORDEN_VISTA[1]: "#2171b5", ORDEN_VISTA[2]: "#e6550d"}
COLORES_REMODELACION = {"Remodelada": "#e6550d", "No remodelada": "#9ecae1"}
COLOR_PRINCIPAL = "#2171b5"
COLOR_ALERTA = "#e6550d"
CENTRO_MAPA = {"lat": 47.50, "lon": -122.15}
CONFIG_GRAFICO = {"displayModeBar": False}

PREGUNTA_NEGOCIO = (
    "¿Cómo varía el precio de venta de las viviendas vendidas en el Condado de King entre mayo de 2014 y mayo "
    "de 2015 según su tamaño, su calificación de construcción, su ubicación frente al agua y su antigüedad o remodelación?"
)

# Títulos iguales a los del documento de AE3; indicador, visual y decisión para cada pregunta.
PREGUNTAS = {
    "P1": {"titulo": "Variación del precio según tamaño habitable y capacidad (habitaciones y baños)",
           "indicador": "Mediana del precio (USD) por tamaño, habitaciones y baños; correlación r con el precio.",
           "visual": "Barras de mediana por grupo; barras de correlación.",
           "decision": "Usar rangos de referencia por tamaño para la primera estimación del avalúo."},
    "P2": {"titulo": "Diferencias de precio según nivel de calificación constructiva",
           "indicador": "Mediana del precio (USD) y número de viviendas por nivel y por grado (4–13).",
           "visual": "Barras de mediana y de conteo; línea por grado; mapa de las bien calificadas.",
           "decision": "Usar la calificación como criterio principal para ubicar una vivienda en un rango."},
    "P3": {"titulo": "Prima descriptiva de precio por vista al mar (Puget Sound) vs. vista a lago interior",
           "indicador": "Número de viviendas y mediana del precio (USD y USD/m²) por tipo de vista.",
           "visual": "Barras de conteo, de mediana y de USD/m²; mapa de las viviendas con vista.",
           "decision": "Tratar la vista al agua como un segmento aparte, no como regla general del mercado."},
    "P4": {"titulo": "Variación del precio por subregión, barrios, cercanía comercial, transporte, parques y entorno",
           "indicador": "Mediana del precio por m² (USD/m²) por subregión, tamaño y anillo urbano; mediana por barrio.",
           "visual": "Barras ordenadas, mapa de calor, mapa de precio por m², entorno y ranking de barrios.",
           "decision": "Priorizar la ubicación en la matriz de referencia geográfica; barrios como orientación."},
    "P5": {"titulo": "Variación del precio según década de construcción y condición de remodelación",
           "indicador": "Participación (%) de remodeladas y mediana del precio (USD) por década y condición.",
           "visual": "Torta de participación; barras agrupadas por década.",
           "decision": "Considerar la remodelación como valor adicional al comparar viviendas de la misma época."},
    "P6": {"titulo": "Dinámica mensual del volumen de ventas y los precios (mayo 2014 – mayo 2015)",
           "indicador": "Ventas (n), mediana mensual (USD) y su variación porcentual frente al mes anterior.",
           "visual": "Líneas de ventas y de mediana; barras de variación; barras apiladas por calificación.",
           "decision": "Planear cuándo hay más movimiento de viviendas y revisar precios con ritmo trimestral."},
}

COLUMNAS_REQUERIDAS = {
    "Precio (USD)", "Precio por m2 (USD)", "Fecha de venta", "Periodo de venta", "Área habitable (m2)",
    "Área del lote (m2)", "Habitaciones", "Baños", "Calificación", "Nivel de calificación", "Categoría de tamaño",
    "Antigüedad (años)", "Década de construcción", "Condición de remodelación", "Vista al mar o lago",
    "Municipio / Subregión", "Barrio / Sector", "Anillo urbano", "Acceso a transporte",
    "Nivel de comercio y servicios", "Parques y zonas verdes", "Latitud", "Longitud",
}

COLUMNAS_APP = sorted(COLUMNAS_REQUERIDAS | {"Cuerpo de agua cercano", "Zonas comunes y entorno"})


def usd(x: float) -> str:
    return f"USD {x:,.0f}".replace(",", ".")


def numero(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def pct(x: float, decimales: int = 1) -> str:
    return f"{x * 100:.{decimales}f} %".replace(".", ",")


def fuerza_corr(r: float) -> str:
    a = abs(r)
    if a >= 0.7:
        return "fuerte"
    if a >= 0.4:
        return "moderada"
    if a >= 0.2:
        return "débil"
    return "muy débil o nula"


def _esc(texto: str) -> str:
    """Streamlit interpreta $ como LaTeX; se escapa por si aparece en un texto."""
    return texto.replace("$", "\\$")


def grupo_banos(b: float) -> str:
    if b < 1.5:
        return ORDEN_BANOS[0]
    if b < 2.25:
        return ORDEN_BANOS[1]
    if b < 3.25:
        return ORDEN_BANOS[2]
    return ORDEN_BANOS[3]


class DashboardAE4:
    def __init__(self, datos: pd.DataFrame, auditoria: pd.DataFrame | None = None,
                 auditoria_columnas: pd.DataFrame | None = None, diccionario: pd.DataFrame | None = None) -> None:
        faltan = COLUMNAS_REQUERIDAS - set(datos.columns)
        if faltan:
            raise ValueError(f"Faltan columnas en el dataset: {sorted(faltan)}")
        self.datos = datos.copy()
        self.datos["Fecha de venta"] = pd.to_datetime(self.datos["Fecha de venta"])
        self.datos["Bien calificada"] = self.datos["Calificación"] >= UMBRAL_BIEN_CALIFICADA
        self.datos["Grupo de baños"] = self.datos["Baños"].apply(grupo_banos)
        self.auditoria = auditoria
        self.auditoria_columnas = auditoria_columnas
        self.diccionario = diccionario
        self.reportes = Reportes()

    # ---------- carga y exportación ----------
    @classmethod
    def desde_csv(cls, ruta: str | Path) -> "DashboardAE4":
        return cls(pd.read_csv(ruta))

    @classmethod
    def desde_carpeta(cls, carpeta: str | Path) -> "DashboardAE4":
        """Carga el dataset y, si existen, la auditoría y el diccionario exportados por el notebook."""
        carpeta = Path(carpeta)

        def opcional(nombre: str) -> pd.DataFrame | None:
            ruta = carpeta / nombre
            return pd.read_csv(ruta) if ruta.exists() else None

        return cls(pd.read_csv(carpeta / ARCHIVO_DATOS), opcional(ARCHIVO_AUDITORIA),
                   opcional(ARCHIVO_AUDITORIA_COLUMNAS), opcional(ARCHIVO_DICCIONARIO))

    @staticmethod
    def exportar_datos(origen: str | Path, destino: str | Path, solo_columnas_app: bool = False) -> Path:
        """Copia el dataset de AE3 a la carpeta de la app (el workflow borra y regenera src/AE3/datos)."""
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        if solo_columnas_app:
            pd.read_csv(origen, usecols=lambda c: c in COLUMNAS_APP).to_csv(destino, index=False)
        else:
            shutil.copyfile(origen, destino)
        return destino

    # ---------- filtros ----------
    def filtrar(self, tamanos, calificaciones, municipios, fechas, extra: dict | None = None) -> pd.DataFrame:
        """Filtros principales y, opcionalmente, los de "Más filtros" (vista, remodelación, anillo, década, precio, habitaciones)."""
        d = self.datos
        d = d[d["Categoría de tamaño"].isin(tamanos)]
        d = d[d["Nivel de calificación"].isin(calificaciones)]
        d = d[d["Municipio / Subregión"].isin(municipios)]
        ini, fin = fechas
        d = d[(d["Fecha de venta"] >= pd.Timestamp(ini)) & (d["Fecha de venta"] <= pd.Timestamp(fin))]
        extra = extra or {}
        for columna, clave in (("Vista al mar o lago", "vista"), ("Condición de remodelación", "remodelacion"),
                               ("Anillo urbano", "anillo")):
            if extra.get(clave) is not None:
                d = d[d[columna].isin(extra[clave])]
        if extra.get("decadas"):
            decadas = sorted(self.datos["Década de construcción"].unique())
            desde, hasta = decadas.index(extra["decadas"][0]), decadas.index(extra["decadas"][1])
            d = d[d["Década de construcción"].isin(decadas[desde:hasta + 1])]
        if extra.get("precio"):
            d = d[d["Precio (USD)"].between(*extra["precio"])]
        if extra.get("habitaciones"):
            d = d[d["Habitaciones"].between(*extra["habitaciones"])]
        return d

    # ---------- métricas ----------
    @staticmethod
    def resumen(d: pd.DataFrame, columna: str | list[str], orden: list[str] | None = None) -> pd.DataFrame:
        r = d.groupby(columna).agg(
            viviendas=("Precio (USD)", "size"),
            mediana=("Precio (USD)", "median"),
            media=("Precio (USD)", "mean"),
            mediana_m2=("Precio por m2 (USD)", "median"),
        ).reset_index()
        if orden is not None and isinstance(columna, str):
            r = r.set_index(columna).reindex([o for o in orden if o in set(r[columna])]).reset_index()
        return r

    def tabla_estadistica(self, d: pd.DataFrame, columna: str | list[str], orden: list[str] | None = None) -> pd.DataFrame:
        """Tabla completa por grupo con la misma función de AE3 (Reportes.tabla_estadistica_grupo)."""
        t = self.reportes.tabla_estadistica_grupo(d, columna)
        if orden is not None and isinstance(columna, str):
            t = t.set_index(columna).reindex([o for o in orden if o in set(t[columna])]).reset_index()
        return t

    @staticmethod
    def correlacion(d: pd.DataFrame, a: str, b: str) -> float:
        return float(d[a].corr(d[b], method="pearson"))

    def kpis(self, d: pd.DataFrame) -> dict[str, tuple[str, str | None]]:
        """Seis indicadores; el segundo valor compara con el total del dataset cuando hay filtros."""
        base = self.datos
        n, nb = max(len(d), 1), max(len(base), 1)
        filtrado = len(d) != len(base)

        def delta(actual: float, total: float, es_pct: bool = False) -> str | None:
            if not filtrado or total == 0:
                return None
            if es_pct:
                return f"{(actual - total) * 100:+.1f} p.p. vs. total".replace(".", ",")
            return f"{(actual / total - 1) * 100:+.1f} % vs. total".replace(".", ",")

        m, mb = d["Precio (USD)"].median(), base["Precio (USD)"].median()
        m2, m2b = d["Precio por m2 (USD)"].median(), base["Precio por m2 (USD)"].median()
        bc, bcb = d["Bien calificada"].sum() / n, base["Bien calificada"].sum() / nb
        va = (d["Vista al mar o lago"] != ORDEN_VISTA[0]).sum() / n
        vab = (base["Vista al mar o lago"] != ORDEN_VISTA[0]).sum() / nb
        re = (d["Condición de remodelación"] == "Remodelada").sum() / n
        reb = (base["Condición de remodelación"] == "Remodelada").sum() / nb
        return {
            "Viviendas": (numero(len(d)), f"{len(d) / nb * 100:.1f} % del total".replace(".", ",") if filtrado else None),
            "Precio típico (mediana)": (usd(m), delta(m, mb)),
            "Mediana USD/m²": (usd(m2), delta(m2, m2b)),
            "Bien calificadas (8+)": (pct(bc, 0), delta(bc, bcb, True)),
            "Con vista al agua": (pct(va, 2), delta(va, vab, True)),
            "Remodeladas": (pct(re), delta(re, reb, True)),
        }

    # ---------- respuestas de negocio (calculadas) ----------
    def respuesta(self, pid: str, d: pd.DataFrame) -> str:
        if pid == "P1":
            r = self.resumen(d, "Categoría de tamaño", ORDEN_TAMANOS)
            r_ = self.correlacion(d, "Precio (USD)", "Área habitable (m2)")
            veces = r["mediana"].iloc[-1] / r["mediana"].iloc[0] if len(r) > 1 else 1.0
            veces_txt = f"{veces:.1f}".replace(".", ",")
            return (f"La mediana pasa de {usd(r['mediana'].iloc[0])} en la categoría más pequeña a "
                    f"{usd(r['mediana'].iloc[-1])} en la más grande ({veces_txt} veces). "
                    f"El área habitable es la característica más asociada al precio (r = {r_:.2f}, {fuerza_corr(r_)}).")
        if pid == "P2":
            bien, resto = d[d["Bien calificada"]], d[~d["Bien calificada"]]
            if bien.empty or resto.empty:
                return "La selección no tiene viviendas de ambos grupos (8 o más y menos de 8) para comparar."
            return (f"{numero(len(bien))} viviendas ({pct(len(bien) / len(d), 0)}) tienen calificación 8 o más. "
                    f"Su mediana es {usd(bien['Precio (USD)'].median())}, frente a {usd(resto['Precio (USD)'].median())} "
                    f"del resto. Calificación y área van juntas (r = {self.correlacion(d, 'Calificación', 'Área habitable (m2)'):.2f}).")
        if pid == "P3":
            r = self.resumen(d, "Vista al mar o lago", ORDEN_VISTA).set_index("Vista al mar o lago")
            con = int((d["Vista al mar o lago"] != ORDEN_VISTA[0]).sum())
            partes = [f"{k.lower()}: {usd(r.loc[k, 'mediana'])} (n = {int(r.loc[k, 'viviendas'])})" for k in r.index]
            return (f"Solo {con} de {numero(len(d))} viviendas ({pct(con / max(len(d), 1), 2)}) tienen vista al agua. "
                    "Mediana " + "; ".join(partes) + ".")
        if pid == "P4":
            r = self.resumen(d, "Municipio / Subregión").sort_values("mediana_m2", ascending=False)
            return (f"El m² más caro está en {r['Municipio / Subregión'].iloc[0]} ({usd(r['mediana_m2'].iloc[0])}/m²) "
                    f"y el más barato en {r['Municipio / Subregión'].iloc[-1]} ({usd(r['mediana_m2'].iloc[-1])}/m²).")
        if pid == "P5":
            r = self.resumen(d, "Condición de remodelación").set_index("Condición de remodelación")
            if not {"Remodelada", "No remodelada"} <= set(r.index):
                return "La selección incluye un solo tipo de condición; no hay comparación."
            dec = self.resumen(d, ["Década de construcción", "Condición de remodelación"])
            dec = dec[dec["viviendas"] >= 10].pivot(index="Década de construcción", columns="Condición de remodelación",
                                                    values="mediana").dropna()
            if {"Remodelada", "No remodelada"} <= set(dec.columns):
                mayor = int((dec["Remodelada"] > dec["No remodelada"]).sum())
            else:   # con pocos datos no queda ninguna década con 10 casas en cada grupo
                dec, mayor = dec.iloc[0:0], 0
            return (f"Solo el {pct(r.loc['Remodelada', 'viviendas'] / r['viviendas'].sum())} está remodelado. "
                    f"Mediana remodeladas {usd(r.loc['Remodelada', 'mediana'])} frente a "
                    f"{usd(r.loc['No remodelada', 'mediana'])} no remodeladas. En {mayor} de {len(dec)} décadas comparables "
                    "(con al menos 10 casas en cada grupo) la remodelada tiene mayor precio típico.")
        if pid == "P6":
            r = self.resumen(d, "Periodo de venta").sort_values("Periodo de venta")
            mx, mn = r.loc[r["viviendas"].idxmax()], r.loc[r["viviendas"].idxmin()]
            return (f"Más ventas en {mx['Periodo de venta']} ({numero(mx['viviendas'])}); menos en "
                    f"{mn['Periodo de venta']} ({numero(mn['viviendas'])}). La mediana mensual va de "
                    f"{usd(r['mediana'].min())} a {usd(r['mediana'].max())}. Mayo de 2015 es parcial (hasta el 27).")
        raise KeyError(pid)

    def respuestas(self, d: pd.DataFrame) -> list[dict]:
        return [{"id": pid, "pregunta": p["titulo"], "indicador": p["indicador"], "respuesta": self.respuesta(pid, d),
                 "visual": p["visual"], "decision": p["decision"]} for pid, p in PREGUNTAS.items()]

    # ---------- figuras (se usan igual en el notebook y en la app) ----------
    @staticmethod
    def _estilo(fig: go.Figure, alto: int = 400) -> go.Figure:
        fig.update_layout(height=alto, margin=dict(l=10, r=10, t=60, b=10), template="plotly_white",
                          separators=",.", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""))
        return fig

    def barras(self, df, x, y, titulo, etiqueta, horizontal=False, orden_desc=False, formato=",.0f") -> go.Figure:
        if horizontal:
            fig = px.bar(df, x=y, y=x, orientation="h", text=y, title=titulo, color_discrete_sequence=[COLOR_PRINCIPAL])
            fig.update_layout(xaxis_title=etiqueta, yaxis_title="")
            fig.update_yaxes(categoryorder="total ascending" if orden_desc else "trace")
        else:
            fig = px.bar(df, x=x, y=y, text=y, title=titulo, color_discrete_sequence=[COLOR_PRINCIPAL])
            fig.update_layout(xaxis_title="", yaxis_title=etiqueta)
        fig.update_traces(texttemplate=f"%{{text:{formato}}}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        return self._estilo(fig)

    def mapa(self, df, color, titulo, colores=None, orden=None, continuo=False, rango=None) -> go.Figure:
        """Mapa con fondo de calles (carto-positron, sin clave). Necesita internet para cargar el fondo."""
        argumentos = dict(lat="Latitud", lon="Longitud", color=color, zoom=8.6, center=CENTRO_MAPA,
                          map_style="carto-positron", title=titulo,
                          hover_data={"Precio (USD)": ":,.0f", "Precio por m2 (USD)": ":,.0f",
                                      "Barrio / Sector": True, "Latitud": False, "Longitud": False})
        if continuo:
            argumentos.update(color_continuous_scale="YlOrRd", range_color=rango)
        else:
            argumentos.update(color_discrete_map=colores, category_orders={color: orden})
        fig = px.scatter_map(df, **argumentos)
        fig.update_traces(marker=dict(size=5, opacity=0.75))
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0), separators=",.",
                          legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text=""))
        return fig

    # P1
    def figura_p1_tamano(self, d):
        r = self.resumen(d, "Categoría de tamaño", ORDEN_TAMANOS)
        return self.barras(r, "Categoría de tamaño", "mediana", "Precio típico según tamaño de la casa",
                           "Mediana del precio (USD)", horizontal=True)

    def figura_p1_habitaciones(self, d):
        r = self.resumen(d, "Habitaciones").query("Habitaciones.between(1, 6)")
        return self.barras(r, "Habitaciones", "mediana", "Precio típico según número de habitaciones (1 a 6)",
                           "Mediana del precio (USD)")

    def figura_p1_banos(self, d):
        r = self.resumen(d, "Grupo de baños", ORDEN_BANOS)
        return self.barras(r, "Grupo de baños", "mediana", "Precio típico según número de baños",
                           "Mediana del precio (USD)")

    def figura_p1_correlacion(self, d):
        nombres = {"Área habitable": "Área habitable (m2)", "Calificación": "Calificación", "Baños": "Baños",
                   "Habitaciones": "Habitaciones", "Área del lote": "Área del lote (m2)", "Antigüedad": "Antigüedad (años)"}
        cor = pd.DataFrame({"Característica": list(nombres),
                            "r": [self.correlacion(d, "Precio (USD)", c) for c in nombres.values()]})
        cor["Relación"] = cor["r"].apply(lambda v: "Sube junto con el precio" if v > 0 else "Va en sentido contrario")
        fig = px.bar(cor.sort_values("r"), x="r", y="Característica", orientation="h", text="r", color="Relación",
                     color_discrete_map={"Sube junto con el precio": COLOR_PRINCIPAL, "Va en sentido contrario": COLOR_ALERTA},
                     title="Qué tanto se parece el precio a cada característica (correlación r)",
                     labels={"r": "Correlación r (de -1 a 1)", "Característica": ""})
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[-1, 1])
        return self._estilo(fig, 380)

    # Variables de P1 para ver la correlación de cerca: (columna, etiqueta, ruido horizontal, ancho de grupo)
    VARIABLES_CORRELACION = [
        ("Área habitable (m2)", "Área habitable (m²)", 0.0, 25),
        ("Calificación", "Calificación (grado 4 a 13)", 0.25, 1),
        ("Baños", "Baños", 0.12, 0.5),
        ("Antigüedad (años)", "Antigüedad (años)", 0.0, 10),
    ]

    def _titulo_r(self, d, columna, etiqueta) -> str:
        if d[columna].nunique() < 2:
            return f"{etiqueta} · r no calculable"
        r = self.correlacion(d, "Precio (USD)", columna)
        return f"{etiqueta} · r = {r:.2f}".replace(".", ",")

    COLORES_UBICACION = {
        "Anillo urbano": {ORDEN_ANILLO[0]: "#b2182b", ORDEN_ANILLO[1]: "#f4a582", ORDEN_ANILLO[2]: "#4393c3"},
        "Municipio / Subregión": {"Seattle": "#b2182b", "Eastside (Bellevue/Kirkland/Redmond)": "#ef8a62",
                                  "North King County": "#92c5de", "South King County": "#2166ac",
                                  "Foothills / Este Semirrural": "#7fbc41", "Vashon / Islas": "#878787"},
    }

    def figura_p1_dispersion(self, d, n_muestra: int = 3000, color_por: str | None = None):
        """Una nube de puntos por variable con su línea de tendencia y el r en el título.

        color_por: None, "Anillo urbano" o "Municipio / Subregión" para ver cómo la ubicación explica los puntos altos.
        """
        variables = self.VARIABLES_CORRELACION
        fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.16,
                            subplot_titles=[self._titulo_r(d, c, e) for c, e, _, _ in variables])
        muestra = d.sample(min(n_muestra, len(d)), random_state=3)
        techo = float(d["Precio (USD)"].quantile(0.99))
        ruido = np.random.default_rng(0)
        colores = self.COLORES_UBICACION.get(color_por, {}) if color_por else {}
        for i, (col, etiqueta, jitter, _) in enumerate(variables):
            fila, columna = i // 2 + 1, i % 2 + 1
            x = muestra[col].to_numpy(dtype=float)
            if jitter:
                x = x + ruido.uniform(-jitter, jitter, len(x))
            grupos = [(None, np.ones(len(muestra), dtype=bool), COLOR_PRINCIPAL)] if not colores else [
                (g, (muestra[color_por] == g).to_numpy(), c) for g, c in colores.items()]
            for nombre, mascara, color in grupos:
                if not mascara.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=x[mascara], y=muestra["Precio (USD)"].to_numpy()[mascara], mode="markers",
                    name=nombre, legendgroup=nombre, showlegend=bool(nombre) and i == 0,
                    marker=dict(size=4, opacity=0.35 if colores else 0.25, color=color),
                    customdata=muestra["Barrio / Sector"].to_numpy()[mascara],
                    hovertemplate=f"{etiqueta}: %{{x:,.1f}}<br>Precio: USD %{{y:,.0f}}<br>%{{customdata}}<extra></extra>"),
                    row=fila, col=columna)
            x_min, x_max = float(d[col].min()), float(d[col].quantile(0.995))
            if d[col].nunique() >= 2:
                pendiente, corte = np.polyfit(d[col].astype(float), d["Precio (USD)"].astype(float), 1)
                xs = np.linspace(x_min, x_max, 50)
                fig.add_trace(go.Scatter(x=xs, y=pendiente * xs + corte, mode="lines", name="Tendencia (recta)",
                                         line=dict(color="#333333" if colores else COLOR_ALERTA, width=3),
                                         showlegend=(i == 0)), row=fila, col=columna)
            margen = max((x_max - x_min) * 0.03, jitter * 1.5)
            fig.update_xaxes(title_text=etiqueta, range=[x_min - margen, x_max + margen], row=fila, col=columna)
            fig.update_yaxes(title_text="Precio (USD)" if columna == 1 else "", range=[0, techo], row=fila, col=columna)
        fig = self._estilo(fig, 780)
        fig.update_layout(title="Precio frente a cada característica: cada punto es una casa", margin=dict(t=80, b=90),
                          legend=dict(orientation="h", yanchor="top", y=-0.1, x=0.5, xanchor="center", itemsizing="constant"))
        return fig

    def figura_p1_lineas(self, d, n_minimo: int = 20):
        """Mediana del precio por tramos de cada variable: muestra la tendencia sin el ruido de los puntos."""
        variables = self.VARIABLES_CORRELACION
        fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.16,
                            subplot_titles=[self._titulo_r(d, c, e) for c, e, _, _ in variables])
        for i, (col, etiqueta, _, ancho) in enumerate(variables):
            fila, columna = i // 2 + 1, i % 2 + 1
            tramo = (np.floor(d[col] / ancho) * ancho + (ancho / 2 if ancho > 1 else 0)).rename("Tramo")
            r = d.groupby(tramo)["Precio (USD)"].agg(["median", "size"]).reset_index()
            r = r[r["size"] >= n_minimo]
            fig.add_trace(go.Scatter(
                x=r["Tramo"], y=r["median"], mode="lines+markers", showlegend=False, customdata=r["size"],
                line=dict(color=COLOR_PRINCIPAL, width=3), marker=dict(size=7),
                hovertemplate=f"{etiqueta}: %{{x:,.1f}}<br>Mediana: USD %{{y:,.0f}}<br>Viviendas: %{{customdata:,.0f}}<extra></extra>"),
                row=fila, col=columna)
            fig.update_xaxes(title_text=etiqueta, row=fila, col=columna)
            fig.update_yaxes(title_text="Mediana del precio (USD)" if columna == 1 else "", rangemode="tozero",
                             row=fila, col=columna)
        fig = self._estilo(fig, 700)
        fig.update_layout(title=f"Precio típico (mediana) a medida que sube cada característica (grupos con al menos {n_minimo} casas)",
                          margin=dict(t=90))
        return fig

    # P1 · casas atípicas (caras para su número de baños) y su ubicación
    def atipicas(self, d: pd.DataFrame) -> pd.DataFrame:
        """Marca como atípica la vivienda cuyo precio supera Q3 + 1,5 × RIC de su grupo de baños (regla de Tukey)."""
        grupo = d.groupby("Grupo de baños")["Precio (USD)"]
        q1, q3 = grupo.transform(lambda s: s.quantile(0.25)), grupo.transform(lambda s: s.quantile(0.75))
        return d.assign(**{"Umbral atípico (USD)": q3 + 1.5 * (q3 - q1)}).assign(
            **{"Atípica": lambda x: x["Precio (USD)"] > x["Umbral atípico (USD)"]})

    def comparacion_atipicas(self, d: pd.DataFrame) -> pd.DataFrame:
        """Porcentajes y medianas de ubicación y calidad: casas atípicas frente al resto."""
        a = self.atipicas(d)
        filas = []
        for nombre, sub in (("Atípicas (caras para sus baños)", a[a["Atípica"]]), ("Resto de casas", a[~a["Atípica"]])):
            if sub.empty:
                continue
            filas.append({
                "Grupo": nombre,
                "Viviendas": len(sub),
                "% a menos de 10 km de un centro comercial": (sub["Anillo urbano"] == ORDEN_ANILLO[0]).mean() * 100,
                "% en Seattle o Eastside": sub["Municipio / Subregión"].isin(
                    ["Seattle", "Eastside (Bellevue/Kirkland/Redmond)"]).mean() * 100,
                "% con vista al agua": (sub["Vista al mar o lago"] != ORDEN_VISTA[0]).mean() * 100,
                "% con calificación 8 o más": sub["Bien calificada"].mean() * 100,
                "Mediana distancia al centro comercial (km)": sub["Distancia polo comercial más cercano (km)"].median()
                if "Distancia polo comercial más cercano (km)" in sub else np.nan,
                "Mediana área (m²)": sub["Área habitable (m2)"].median(),
                "Mediana USD/m²": sub["Precio por m2 (USD)"].median(),
            })
        return pd.DataFrame(filas)

    def figura_p1_atipicas(self, d):
        c = self.comparacion_atipicas(d)
        indicadores = ["% a menos de 10 km de un centro comercial", "% en Seattle o Eastside",
                       "% con calificación 8 o más", "% con vista al agua"]
        largo = c.melt(id_vars="Grupo", value_vars=indicadores, var_name="Indicador", value_name="Porcentaje")
        cortos = {indicadores[0]: "A menos de 10 km<br>de un centro comercial", indicadores[1]: "En Seattle<br>o Eastside",
                  indicadores[2]: "Calificación<br>8 o más", indicadores[3]: "Con vista<br>al agua"}
        largo["Indicador"] = largo["Indicador"].map(cortos)
        fig = px.bar(largo, x="Indicador", y="Porcentaje", color="Grupo", barmode="group", text="Porcentaje",
                     color_discrete_map={"Atípicas (caras para sus baños)": COLOR_ALERTA, "Resto de casas": "#9ecae1"},
                     title="¿Dónde están las casas caras para su número de baños?",
                     labels={"Porcentaje": "Porcentaje de viviendas (%)", "Indicador": ""})
        fig.update_traces(texttemplate="%{text:.0f} %", textposition="outside", cliponaxis=False)
        fig.update_yaxes(range=[0, 105])
        return self._estilo(fig, 420)

    def figura_p1_mapa_atipicas(self, d):
        a = self.atipicas(d)
        resto = a[~a["Atípica"]]
        base = pd.concat([resto.sample(min(4000, len(resto)), random_state=4), a[a["Atípica"]]])
        base = base.assign(Tipo=np.where(base["Atípica"], "Atípica (cara para sus baños)", "Resto (muestra)"))
        return self.mapa(base, "Tipo", "Ubicación de las casas atípicas frente al resto",
                         {"Resto (muestra)": "#c6dbef", "Atípica (cara para sus baños)": COLOR_ALERTA},
                         ["Resto (muestra)", "Atípica (cara para sus baños)"])

    def respuesta_atipicas(self, d) -> str:
        a = self.atipicas(d)
        at = a[a["Atípica"]]
        if at.empty:
            return "Con los filtros actuales no hay casas atípicas por precio."
        c = self.comparacion_atipicas(d).set_index("Grupo")
        texto = (f"{numero(len(at))} casas ({pct(len(at) / len(a))}) cuestan mucho más que las demás con su mismo número de "
                 f"baños. El {c.iloc[0]['% a menos de 10 km de un centro comercial']:.0f} % de ellas está a menos de 10 km de "
                 f"un centro comercial")
        if len(c) > 1:
            texto += f", frente al {c.iloc[1]['% a menos de 10 km de un centro comercial']:.0f} % del resto"
        uno = at[at["Grupo de baños"] == ORDEN_BANOS[0]]
        if len(uno) >= 5:
            barrio = uno["Barrio / Sector"].value_counts()
            umbral = uno["Umbral atípico (USD)"].iloc[0]
            texto += (f". Por ejemplo, {numero(len(uno))} casas de 1 baño cuestan más de {usd(umbral)}; la mayoría está en "
                      f"{uno['Municipio / Subregión'].value_counts().index[0]} (barrio más frecuente: {barrio.index[0]}, "
                      f"{numero(barrio.iloc[0])} casas)")
        return texto + "."

    # P2
    def figura_p2_mediana(self, d):
        r = self.resumen(d, "Nivel de calificación", ORDEN_CALIFICACION)
        return self.barras(r, "Nivel de calificación", "mediana", "Precio típico según nivel de calificación",
                           "Mediana del precio (USD)")

    def figura_p2_conteo(self, d):
        r = self.resumen(d, "Nivel de calificación", ORDEN_CALIFICACION)
        return self.barras(r, "Nivel de calificación", "viviendas", "Cuántas casas hay en cada nivel", "Viviendas (n)")

    def figura_p2_grado(self, d):
        r = self.resumen(d, "Calificación").sort_values("Calificación")
        fig = px.line(r, x="Calificación", y="mediana", markers=True, custom_data=["viviendas"],
                      title="Precio típico por grado de calificación (4 a 13)",
                      labels={"Calificación": "Grado del condado", "mediana": "Mediana del precio (USD)"})
        fig.update_traces(line_color=COLOR_PRINCIPAL,
                          hovertemplate="Grado %{x}<br>Mediana: USD %{y:,.0f}<br>Viviendas: %{customdata[0]:,.0f}<extra></extra>")
        fig.update_xaxes(dtick=1)
        return self._estilo(fig, 380)

    def figura_p2_mapa(self, d):
        bien = d[d["Bien calificada"]]
        return self.mapa(bien, "Nivel de calificación",
                         f"Dónde están las viviendas bien calificadas (8 o más): {numero(len(bien))}",
                         COLORES_CALIFICACION, ORDEN_CALIFICACION)

    def figura_p2_mapa_baja(self, d):
        baja = d[~d["Bien calificada"]]
        return self.mapa(baja, "Nivel de calificación",
                         f"Dónde están las viviendas con calificación menor a 8: {numero(len(baja))}",
                         {ORDEN_CALIFICACION[0]: "#a63603", ORDEN_CALIFICACION[1]: "#fd8d3c"},
                         [ORDEN_CALIFICACION[1], ORDEN_CALIFICACION[0]])   # Baja se dibuja encima para que se vea

    def figura_p2_subregion(self, d):
        """Cómo se reparten por subregión las casas de calificación alta y baja (porcentaje de cada grupo)."""
        t = d.assign(Grupo=np.where(d["Bien calificada"], "Calificación 8 o más", "Calificación menor a 8"))
        r = (t.groupby(["Grupo", "Municipio / Subregión"]).size() / t.groupby("Grupo").size()).mul(100)
        r = r.rename("Porcentaje").reset_index()
        orden = (r[r["Grupo"] == "Calificación 8 o más"].sort_values("Porcentaje", ascending=False)["Municipio / Subregión"].tolist()
                 or sorted(r["Municipio / Subregión"].unique()))
        r["Municipio / Subregión"] = r["Municipio / Subregión"].str.replace(" (", "<br>(", regex=False)
        orden = [o.replace(" (", "<br>(") for o in orden]
        fig = px.bar(r, x="Municipio / Subregión", y="Porcentaje", color="Grupo", barmode="group", text="Porcentaje",
                     category_orders={"Municipio / Subregión": orden,
                                      "Grupo": ["Calificación menor a 8", "Calificación 8 o más"]},
                     color_discrete_map={"Calificación menor a 8": "#fd8d3c", "Calificación 8 o más": "#08306b"},
                     title="En qué subregión está cada grupo (porcentaje de las casas del grupo)",
                     labels={"Porcentaje": "Porcentaje del grupo (%)", "Municipio / Subregión": ""})
        fig.update_traces(texttemplate="%{text:.0f} %", textposition="outside", cliponaxis=False)
        return self._estilo(fig, 420)

    # P3
    def figura_p3_conteo(self, d):
        con = int((d["Vista al mar o lago"] != ORDEN_VISTA[0]).sum())
        partes = pd.DataFrame({"Grupo": ["Con vista al agua", "Sin vista directa"], "Viviendas": [con, len(d) - con]})
        return self.barras(partes, "Grupo", "Viviendas", "Cuántas casas tienen vista al agua", "Viviendas (n)")

    def figura_p3_mediana(self, d):
        r = self.resumen(d, "Vista al mar o lago", ORDEN_VISTA)
        return self.barras(r, "Vista al mar o lago", "mediana", "Precio típico según vista", "Mediana del precio (USD)")

    def figura_p3_m2(self, d):
        r = self.resumen(d, "Vista al mar o lago", ORDEN_VISTA)
        return self.barras(r, "Vista al mar o lago", "mediana_m2", "Precio por m² según vista", "Mediana (USD/m²)")

    def figura_p3_mapa(self, d):
        con_vista = d[d["Vista al mar o lago"] != ORDEN_VISTA[0]]
        return self.mapa(con_vista, "Vista al mar o lago",
                         f"Dónde están las viviendas con vista al agua: {numero(len(con_vista))}",
                         COLORES_VISTA, ORDEN_VISTA)

    @staticmethod
    def _con_vista(d: pd.DataFrame) -> pd.Series:
        return pd.Series(np.where(d["Vista al mar o lago"] != ORDEN_VISTA[0], "Con vista al agua", "Sin vista"),
                         index=d.index, name="Vista")

    def figura_p3_caja(self, d):
        fig = px.box(d, x="Vista al mar o lago", y="Precio (USD)", color="Vista al mar o lago", points="outliers",
                     log_y=True, color_discrete_map=COLORES_VISTA, category_orders={"Vista al mar o lago": ORDEN_VISTA},
                     title="Cómo se reparten los precios según la vista (escala logarítmica)",
                     labels={"Vista al mar o lago": "", "Precio (USD)": "Precio (USD, escala logarítmica)"})
        marcas = [100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
        fig.update_yaxes(tickvals=marcas, ticktext=[usd(v) for v in marcas], minor=dict(showgrid=False))
        fig.update_layout(showlegend=False)
        return self._estilo(fig, 420)

    def figura_p3_por_tamano(self, d):
        r = self.resumen(d.assign(Vista=self._con_vista(d)), ["Categoría de tamaño", "Vista"])
        fig = px.bar(r, x="Categoría de tamaño", y="mediana", color="Vista", barmode="group", text="mediana",
                     custom_data=["viviendas"], category_orders={"Categoría de tamaño": ORDEN_TAMANOS,
                                                                 "Vista": ["Sin vista", "Con vista al agua"]},
                     color_discrete_map={"Sin vista": "#bdbdbd", "Con vista al agua": COLOR_ALERTA},
                     title="Misma categoría de tamaño, con y sin vista al agua",
                     labels={"mediana": "Mediana del precio (USD)", "Categoría de tamaño": ""})
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>Mediana: USD %{y:,.0f}<br>Viviendas: %{customdata[0]:,.0f}<extra></extra>")
        return self._estilo(fig, 420)

    def figura_p3_por_subregion(self, d, n_minimo: int = 5):
        t = d.assign(Vista=self._con_vista(d))
        r = self.resumen(t, ["Municipio / Subregión", "Vista"])
        validas = r[(r["Vista"] == "Con vista al agua") & (r["viviendas"] >= n_minimo)]["Municipio / Subregión"]
        r = r[r["Municipio / Subregión"].isin(validas)].copy()
        r["Municipio / Subregión"] = r["Municipio / Subregión"].str.replace(" (", "<br>(", regex=False)
        fig = px.bar(r, x="Municipio / Subregión", y="mediana", color="Vista", barmode="group", text="mediana",
                     custom_data=["viviendas"], category_orders={"Vista": ["Sin vista", "Con vista al agua"]},
                     color_discrete_map={"Sin vista": "#bdbdbd", "Con vista al agua": COLOR_ALERTA},
                     title=f"Misma subregión, con y sin vista al agua (subregiones con al menos {n_minimo} casas con vista)",
                     labels={"mediana": "Mediana del precio (USD)", "Municipio / Subregión": ""})
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>Mediana: USD %{y:,.0f}<br>Viviendas: %{customdata[0]:,.0f}<extra></extra>")
        return self._estilo(fig, 420)

    def perfil_p3(self, d) -> pd.DataFrame:
        """Cómo son las casas con vista frente a las demás (ayuda a entender de dónde sale la prima)."""
        filas = []
        for nombre, sub in d.groupby(self._con_vista(d)):
            filas.append({
                "Grupo": nombre, "Viviendas": len(sub),
                "Mediana precio (USD)": sub["Precio (USD)"].median(),
                "Mediana USD/m²": sub["Precio por m2 (USD)"].median(),
                "Mediana área (m²)": sub["Área habitable (m2)"].median(),
                "% calificación 8 o más": sub["Bien calificada"].mean() * 100,
                "% a menos de 10 km de un centro comercial": (sub["Anillo urbano"] == ORDEN_ANILLO[0]).mean() * 100,
                "Mediana antigüedad (años)": sub["Antigüedad (años)"].median(),
            })
        return pd.DataFrame(filas).sort_values("Grupo")

    def prima_por_tamano(self, d, n_minimo: int = 5) -> pd.DataFrame:
        r = self.resumen(d.assign(Vista=self._con_vista(d)), ["Categoría de tamaño", "Vista"])
        t = r.pivot(index="Categoría de tamaño", columns="Vista", values="mediana")
        n = r.pivot(index="Categoría de tamaño", columns="Vista", values="viviendas")
        if not {"Con vista al agua", "Sin vista"} <= set(t.columns):
            return pd.DataFrame()
        t = t[n["Con vista al agua"].fillna(0) >= n_minimo].dropna()
        return (t["Con vista al agua"] / t["Sin vista"]).rename("Veces").reset_index()

    # P4
    def figura_p4_subregion(self, d):
        r = self.resumen(d, "Municipio / Subregión").sort_values("mediana_m2", ascending=False)
        return self.barras(r, "Municipio / Subregión", "mediana_m2", "Precio por m² en cada subregión (de mayor a menor)",
                           "Mediana del precio (USD/m²)", horizontal=True, orden_desc=True)

    MEDIDAS_CALOR = {
        "Precio por m²": ("Precio por m2 (USD)", "median", "USD por m²", "Blues"),
        "Precio típico de la casa": ("Precio (USD)", "median", "USD", "Purples"),
        "Número de casas": ("Precio (USD)", "size", "Casas", "Greens"),
    }
    N_MINIMO_CASILLA = 20

    def tabla_calor(self, d) -> pd.DataFrame:
        """Por subregión y tamaño: casas, mediana del precio y mediana por m²."""
        return (d.groupby(["Municipio / Subregión", "Categoría de tamaño"])
                .agg(casas=("Precio (USD)", "size"), precio=("Precio (USD)", "median"),
                     m2=("Precio por m2 (USD)", "median")).reset_index())

    def figura_p4_calor(self, d, medida: str = "Precio por m²"):
        """Mapa de calor subregión × tamaño. medida: 'Precio por m²', 'Precio típico de la casa' o 'Número de casas'."""
        columna, funcion, unidad, escala = self.MEDIDAS_CALOR[medida]
        t = self.tabla_calor(d)
        valor = {"Precio por m²": "m2", "Precio típico de la casa": "precio", "Número de casas": "casas"}[medida]
        orden_filas = self.resumen(d, "Municipio / Subregión").sort_values("mediana_m2", ascending=False)["Municipio / Subregión"]
        columnas = [c for c in ORDEN_TAMANOS if c in set(d["Categoría de tamaño"])]

        def tabla(campo):
            return t.pivot(index="Municipio / Subregión", columns="Categoría de tamaño", values=campo) \
                    .reindex(index=orden_filas, columns=columnas)

        z, casas, precio, m2 = tabla(valor), tabla("casas"), tabla("precio"), tabla("m2")
        pocas = casas < self.N_MINIMO_CASILLA
        if valor != "casas":
            z = z.mask(pocas)          # con menos de 20 casas el valor no es confiable: casilla en gris
        texto = z.apply(lambda c: c.map(lambda v: "" if pd.isna(v) else (f"{v:,.0f}".replace(",", "."))))
        if valor != "casas":
            texto = texto.where(~pocas, "pocas casas")
        etiquetas_x = [c.split(". ", 1)[-1].replace(" (", "<br>(") for c in columnas]
        etiquetas_y = [f.replace(" (", "<br>(") for f in z.index]
        fig = go.Figure(go.Heatmap(
            z=z.values, x=etiquetas_x, y=etiquetas_y, colorscale=escala, text=texto.values, texttemplate="%{text}",
            colorbar=dict(title=unidad), hoverongaps=False,
            customdata=np.dstack([casas.values, precio.values, m2.values]),
            hovertemplate=("<b>%{y}</b> · casa %{x}<br>Casas: %{customdata[0]:,.0f}<br>"
                           "Precio típico: USD %{customdata[1]:,.0f}<br>Precio por m²: USD %{customdata[2]:,.0f}<extra></extra>")))
        fig.update_layout(plot_bgcolor="#eeeeee", xaxis_title="Tamaño de la casa", yaxis_title="",
                          title=f"{medida} según subregión (filas) y tamaño de casa (columnas)")
        fig.update_yaxes(autorange="reversed")
        return self._estilo(fig, 480)

    def explicar_calor(self, d) -> str:
        t = self.tabla_calor(d)
        t = t[t["casas"] >= self.N_MINIMO_CASILLA]
        if len(t) < 2:
            return ""
        caro, barato = t.loc[t["m2"].idxmax()], t.loc[t["m2"].idxmin()]
        corto = lambda c: c.split(". ", 1)[-1].split(" (")[0].lower()
        veces = f"{caro['m2'] / barato['m2']:.1f}".replace(".", ",")
        return (f"El metro cuadrado más caro está en una casa {corto(caro['Categoría de tamaño'])} de {caro['Municipio / Subregión']} "
                f"({usd(caro['m2'])} por m²) y el más barato en una casa {corto(barato['Categoría de tamaño'])} de "
                f"{barato['Municipio / Subregión']} ({usd(barato['m2'])} por m²): {veces} veces menos.")

    def figura_p4_mapa_m2(self, d):
        rango = tuple(d["Precio por m2 (USD)"].quantile([0.05, 0.95]).round(0))
        return self.mapa(d, "Precio por m2 (USD)", "Mapa del precio por m² (cada punto es una vivienda)",
                         continuo=True, rango=rango)

    def figura_p4_anillo(self, d):
        r = self.resumen(d, "Anillo urbano", ORDEN_ANILLO)
        return self.barras(r, "Anillo urbano", "mediana", "Precio típico según distancia al polo comercial más cercano",
                           "Mediana del precio (USD)")

    ATRIBUTOS_ENTORNO = {"Acceso a transporte": "¿Qué tan fácil es llegar en transporte?",
                         "Nivel de comercio y servicios": "¿Cuántos comercios y servicios hay cerca?",
                         "Parques y zonas verdes": "¿Cuántos parques hay cerca?"}

    def resumen_entorno(self, d) -> pd.DataFrame:
        """Por atributo y nivel: viviendas, % del total, mediana y lo que explica la diferencia (zona, tamaño, calificación)."""
        filas = []
        for atributo in self.ATRIBUTOS_ENTORNO:
            for nivel, sub in d.groupby(atributo):
                zonas = sub["Municipio / Subregión"].value_counts(normalize=True)
                filas.append({"Atributo": atributo, "Nivel": nivel, "Viviendas": len(sub),
                              "% de las casas": len(sub) / len(d) * 100, "Mediana (USD)": sub["Precio (USD)"].median(),
                              "Subregión principal": zonas.index[0], "% en esa subregión": zonas.iloc[0] * 100,
                              "Mediana área (m²)": sub["Área habitable (m2)"].median(),
                              "% calificación 8 o más": sub["Bien calificada"].mean() * 100})
        t = pd.DataFrame(filas)
        t["orden"] = t["Nivel"].map({n: i for i, n in enumerate(ORDEN_NIVEL)})
        return t.sort_values(["Atributo", "orden"]).drop(columns="orden")

    def figura_p4_entorno(self, d):
        t = self.resumen_entorno(d)
        general = float(d["Precio (USD)"].median())
        t["Etiqueta"] = t.apply(lambda f: f"<b>{f['Nivel']}</b><br>{numero(f['Viviendas'])} casas<br>({f['% de las casas']:.0f} %)",
                                axis=1)
        t["Comparación"] = np.where(t["Mediana (USD)"] >= general, "Por encima de la mediana general",
                                    "Por debajo de la mediana general")
        atributos = list(self.ATRIBUTOS_ENTORNO)
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                            subplot_titles=[f"{a}<br><sup>{p}</sup>" for a, p in self.ATRIBUTOS_ENTORNO.items()])
        colores = {"Por encima de la mediana general": COLOR_PRINCIPAL, "Por debajo de la mediana general": "#fdae6b"}
        vistos = set()
        for j, atributo in enumerate(atributos, start=1):
            sub = t[t["Atributo"] == atributo]
            for comp, grupo in sub.groupby("Comparación", sort=False):
                fig.add_trace(go.Bar(
                    x=grupo["Etiqueta"], y=grupo["Mediana (USD)"], name=comp, marker_color=colores[comp],
                    legendgroup=comp, showlegend=comp not in vistos, text=grupo["Mediana (USD)"],
                    texttemplate="USD %{text:,.0f}", textposition="outside", cliponaxis=False,
                    customdata=grupo[["Subregión principal", "% en esa subregión", "Mediana área (m²)", "% calificación 8 o más"]],
                    hovertemplate=("%{x}<br>Mediana: USD %{y:,.0f}<br>Subregión principal: %{customdata[0]} "
                                   "(%{customdata[1]:.0f} %)<br>Área típica: %{customdata[2]:.0f} m²<br>"
                                   "Calificación 8 o más: %{customdata[3]:.0f} %<extra></extra>")), row=1, col=j)
                vistos.add(comp)
            fig.update_xaxes(categoryorder="array", categoryarray=sub["Etiqueta"].tolist(), row=1, col=j)
        fig.add_hline(y=general, line_dash="dot", line_color="#555555")
        fig.add_annotation(xref="paper", yref="y", x=1, y=general, xanchor="right", yshift=-12, showarrow=False,
                           text=f"Mediana general: {usd(general)}", font=dict(size=11, color="#555555"))
        fig.update_xaxes(tickangle=0)
        fig.update_yaxes(title_text="Mediana del precio (USD)", row=1, col=1)
        fig = self._estilo(fig, 540)
        fig.update_layout(title="¿Las casas con más transporte, comercio y parques cerca cuestan más?",
                          margin=dict(t=110, b=40),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.5, xanchor="center"))
        return fig

    def explicar_entorno(self, d) -> list[str]:
        """Una frase por atributo: qué nivel es el más caro y qué tiene esa zona que lo explica."""
        t = self.resumen_entorno(d)
        frases = []
        for atributo in self.ATRIBUTOS_ENTORNO:
            sub = t[t["Atributo"] == atributo]
            if len(sub) < 2:
                continue
            caro, barato = sub.loc[sub["Mediana (USD)"].idxmax()], sub.loc[sub["Mediana (USD)"].idxmin()]
            frases.append(
                f"**{atributo}:** el nivel más caro es *{caro['Nivel']}* ({usd(caro['Mediana (USD)'])}) y el más barato "
                f"*{barato['Nivel']}* ({usd(barato['Mediana (USD)'])}). Las casas de nivel *{caro['Nivel']}* están sobre todo "
                f"en {caro['Subregión principal']} ({caro['% en esa subregión']:.0f} %), miden {caro['Mediana área (m²)']:.0f} m² "
                f"y el {caro['% calificación 8 o más']:.0f} % tiene calificación 8 o más.")
        return frases

    def tabla_p4_barrios(self, d, n_top: int) -> pd.DataFrame:
        b = self.resumen(d, ["Barrio / Sector", "Municipio / Subregión"]).query("viviendas >= @N_MINIMO_BARRIO")
        return b.sort_values("mediana", ascending=False).head(n_top)

    # P5
    def figura_p5_torta(self, d):
        r = self.resumen(d, "Condición de remodelación")
        fig = px.pie(r, names="Condición de remodelación", values="viviendas", hole=0.35, color="Condición de remodelación",
                     color_discrete_map=COLORES_REMODELACION, title="Parte de las casas según remodelación")
        fig.update_traces(textinfo="percent+label")
        return self._estilo(fig)

    def figura_p5_decadas(self, d):
        r = self.resumen(d, ["Década de construcción", "Condición de remodelación"])
        fig = px.bar(r, x="Década de construcción", y="mediana", color="Condición de remodelación", barmode="group",
                     color_discrete_map=COLORES_REMODELACION, custom_data=["viviendas"],
                     category_orders={"Década de construcción": sorted(r["Década de construcción"].unique())},
                     title="Precio típico por década de construcción, según remodelación",
                     labels={"mediana": "Mediana del precio (USD)", "Década de construcción": "Década de construcción"})
        fig.update_traces(hovertemplate="%{x}<br>Mediana: USD %{y:,.0f}<br>Viviendas: %{customdata[0]:,.0f}<extra></extra>")
        return self._estilo(fig, 420)

    # P6
    def figura_p6_ventas(self, d):
        r = self.resumen(d, "Periodo de venta").sort_values("Periodo de venta")
        fig = px.line(r, x="Periodo de venta", y="viviendas", markers=True, title="Ventas por mes",
                      labels={"viviendas": "Ventas (n)", "Periodo de venta": "Mes"})
        fig.update_traces(line_color=COLOR_PRINCIPAL)
        if "2015-05" in set(r["Periodo de venta"]):
            fig.add_annotation(x="2015-05", y=float(r.loc[r["Periodo de venta"] == "2015-05", "viviendas"].iloc[0]),
                               text="Mes parcial", showarrow=True, arrowhead=2, ay=-40)
        return self._estilo(fig)

    def figura_p6_mediana(self, d):
        r = self.resumen(d, "Periodo de venta").sort_values("Periodo de venta")
        fig = px.line(r, x="Periodo de venta", y="mediana", markers=True, title="Precio típico por mes",
                      labels={"mediana": "Mediana del precio (USD)", "Periodo de venta": "Mes"})
        fig.update_traces(line_color=COLOR_PRINCIPAL)
        fig.add_hline(y=float(d["Precio (USD)"].median()), line_dash="dot", annotation_text="Mediana del periodo")
        return self._estilo(fig)

    def figura_p6_variacion(self, d):
        r = self.resumen(d, "Periodo de venta").sort_values("Periodo de venta")
        r["Variación (%)"] = r["mediana"].pct_change() * 100
        r = r.dropna(subset=["Variación (%)"])
        r["Signo"] = r["Variación (%)"].apply(lambda v: "Sube" if v >= 0 else "Baja")
        fig = px.bar(r, x="Periodo de venta", y="Variación (%)", color="Signo", text="Variación (%)",
                     color_discrete_map={"Sube": COLOR_PRINCIPAL, "Baja": COLOR_ALERTA},
                     title="Variación de la mediana frente al mes anterior",
                     labels={"Periodo de venta": "Mes", "Variación (%)": "Variación (%)"})
        fig.update_traces(texttemplate="%{text:+.1f} %", textposition="outside", cliponaxis=False)
        return self._estilo(fig, 380)

    def figura_p6_apilado(self, d):
        r = d.groupby(["Periodo de venta", "Nivel de calificación"]).size().reset_index(name="Ventas")
        fig = px.bar(r, x="Periodo de venta", y="Ventas", color="Nivel de calificación", color_discrete_map=COLORES_CALIFICACION,
                     category_orders={"Nivel de calificación": ORDEN_CALIFICACION,
                                      "Periodo de venta": sorted(d["Periodo de venta"].unique())},
                     title="Ventas por mes, según nivel de calificación", labels={"Ventas": "Ventas (n)", "Periodo de venta": "Mes"})
        return self._estilo(fig, 420)

    # ---------- componentes de la página ----------
    @staticmethod
    def _grafico(fig, contenedor=st) -> None:
        contenedor.plotly_chart(fig, width="stretch", config=CONFIG_GRAFICO)

    def _encabezado(self, pid: str, d: pd.DataFrame) -> None:
        p = PREGUNTAS[pid]
        st.subheader(f"{pid}. {p['titulo']}")
        st.info(_esc(self.respuesta(pid, d)))
        st.caption(f"Indicador: {p['indicador']}  ·  Decisión: {p['decision']}")

    def _tabla(self, d: pd.DataFrame, columna: str | list[str], titulo: str, orden: list[str] | None = None) -> None:
        with st.expander(f"Tabla estadística · {titulo} (misma función de AE3)"):
            t = self.tabla_estadistica(d, columna, orden)
            st.dataframe(t, hide_index=True, width="stretch")
            nombre = "_".join([columna] if isinstance(columna, str) else columna)
            st.download_button("Descargar tabla (CSV)", t.to_csv(index=False).encode("utf-8"),
                               file_name=f"tabla_{nombre}.csv".replace(" ", "_").replace("/", "-"), mime="text/csv",
                               key=f"descarga_{nombre}")

    def tab_p1(self, d) -> None:
        self._encabezado("P1", d)
        c1, c2 = st.columns(2)
        self._grafico(self.figura_p1_tamano(d), c1)
        self._grafico(self.figura_p1_habitaciones(d), c2)
        c3, c4 = st.columns(2)
        self._grafico(self.figura_p1_banos(d), c3)
        self._grafico(self.figura_p1_correlacion(d), c4)
        st.markdown("**Cómo leerlo:** cada barra es un grupo de casas; entre más larga, más cuesta la casa típica de ese grupo. "
                    "La correlación va de -1 a 1: cerca de 1, el precio y esa característica suben juntos; cerca de 0, no tienen que ver.")
        st.markdown("#### La correlación vista de cerca: área habitable, calificación, baños y antigüedad")
        opciones = {"Anillo urbano (distancia al centro comercial)": "Anillo urbano",
                    "Subregión": "Municipio / Subregión", "Sin color": None}
        eleccion = st.radio("Colorear los puntos por:", list(opciones), horizontal=True, key="p1_color")
        self._grafico(self.figura_p1_dispersion(d, color_por=opciones[eleccion]))
        st.markdown("**Cómo leerlo:** cada punto es una casa. Si los puntos forman una nube que **sube** de izquierda a derecha "
                    "y se pega a la línea naranja, la r es alta (área habitable). Si la nube es **plana y regada**, la r es "
                    "cercana a 0 (antigüedad). La línea naranja es la tendencia recta calculada con todas las casas.")
        self._grafico(self.figura_p1_lineas(d))
        st.markdown("**Cómo leerlo:** cada punto de la línea es el precio típico de un grupo de casas (por ejemplo, las de "
                    "150 a 175 m²). Una línea que sube con fuerza indica una r alta. **Ojo con la antigüedad:** su línea tiene "
                    "forma de U (las casas nuevas y las muy antiguas valen más; las de 40 a 80 años, menos). Por eso su r es "
                    "casi 0: r solo mide relaciones en línea recta, y una U no lo es. Una r cercana a 0 no siempre significa que "
                    "no haya ningún patrón.")

        st.markdown("#### ¿Por qué hay casas caras con pocos baños? La ubicación")
        st.info(_esc(self.respuesta_atipicas(d)))
        c1, c2 = st.columns(2)
        self._grafico(self.figura_p1_atipicas(d), c1)
        self._grafico(self.figura_p1_mapa_atipicas(d), c2)
        comparacion = self.comparacion_atipicas(d)
        st.dataframe(comparacion, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.1f") for c in comparacion.columns
                                    if c not in ("Grupo", "Viviendas")})
        st.markdown("**Cómo leerlo:** una casa es *atípica* si su precio está muy por encima del de las demás casas con el mismo "
                    "número de baños (más de Q3 + 1,5 × RIC, la regla de Tukey). Casi todas están cerca de un centro comercial, en "
                    "Seattle o el Eastside. Es decir: una casa con un solo baño puede ser cara porque está muy bien ubicada.")
        st.caption("Limitación: las atípicas también son, en general, más grandes y mejor calificadas; la ubicación pesa mucho, "
                   "pero no es la única razón. Subregión y anillo urbano vienen de las coordenadas y de NVIDIA Nemotron.")
        st.caption("Limitación: la correlación muestra asociación, no causa. Una casa grande también suele tener más baños y mejor "
                   "calificación. En los puntos se muestra una muestra de 3.000 casas, con un pequeño desplazamiento horizontal en "
                   "calificación y baños para que no se tapen; el eje de precio llega hasta el percentil 99.")
        self._tabla(d, "Categoría de tamaño", "por tamaño", ORDEN_TAMANOS)

    def tab_p2(self, d) -> None:
        self._encabezado("P2", d)
        c1, c2 = st.columns(2)
        self._grafico(self.figura_p2_mediana(d), c1)
        self._grafico(self.figura_p2_conteo(d), c2)
        self._grafico(self.figura_p2_grado(d))
        st.markdown("#### Dónde están las casas según su calificación")
        c3, c4 = st.columns(2)
        self._grafico(self.figura_p2_mapa(d), c3)
        self._grafico(self.figura_p2_mapa_baja(d), c4)
        self._grafico(self.figura_p2_subregion(d))
        texto = ""
        if d["Bien calificada"].any() and (~d["Bien calificada"]).any():
            alta = d[d["Bien calificada"]]["Municipio / Subregión"].value_counts(normalize=True)
            baja = d[~d["Bien calificada"]]["Municipio / Subregión"].value_counts(normalize=True)
            texto = (f" Con la selección actual, la subregión con más casas de calificación 8 o más es {alta.index[0]} "
                     f"({alta.iloc[0] * 100:.0f} % del grupo); la de más casas con calificación menor a 8 es "
                     f"{baja.index[0]} ({baja.iloc[0] * 100:.0f} % del grupo).")
        st.markdown("**Cómo leerlo:** el precio típico sube en cada grado, pero los grados altos tienen pocas casas. "
                    "El mapa de la izquierda muestra las casas con calificación 8 o más (azul oscuro = mejor calificación) y el de "
                    "la derecha las de calificación menor a 8 (naranja oscuro = Baja). Las barras comparan en qué subregión está "
                    "cada grupo." + texto)
        st.caption("Limitación: la calificación es una nota del condado; se mezcla con tamaño y ubicación.")
        self._tabla(d, "Nivel de calificación", "por nivel de calificación", ORDEN_CALIFICACION)

    def tab_p3(self, d) -> None:
        self._encabezado("P3", d)
        c1, c2, c3 = st.columns(3)
        self._grafico(self.figura_p3_conteo(d), c1)
        self._grafico(self.figura_p3_mediana(d), c2)
        self._grafico(self.figura_p3_m2(d), c3)
        st.markdown("**Cómo leerlo:** la primera barra muestra que las casas con vista al agua son muy pocas; por eso no cambian "
                    "el precio de todo el mercado. Las otras dos muestran cuánto más cuestan, en total y por metro cuadrado.")

        if (d["Vista al mar o lago"] == ORDEN_VISTA[0]).all():
            st.warning("Con los filtros actuales no hay casas con vista al agua.")
        else:
            st.markdown("#### ¿La vista al agua se paga por sí sola? Comparar casas parecidas")
            prima = self.prima_por_tamano(d)
            if not prima.empty:
                minimo, maximo = (f"{v:.1f}".replace(".", ",") for v in (prima["Veces"].min(), prima["Veces"].max()))
                st.info(_esc(f"Comparando casas del mismo tamaño, las que tienen vista al agua cuestan entre "
                             f"{minimo} y {maximo} veces lo que cuestan las que no la tienen."))
            c4, c5 = st.columns(2)
            self._grafico(self.figura_p3_por_tamano(d), c4)
            self._grafico(self.figura_p3_por_subregion(d), c5)
            st.markdown("**Cómo leerlo:** cada par de barras compara casas parecidas (del mismo tamaño o de la misma subregión). "
                        "Si la barra naranja sigue siendo más alta, la diferencia no se explica solo por el tamaño o la zona.")
            self._grafico(self.figura_p3_caja(d))
            st.markdown("**Cómo leer la caja:** la línea del medio es la mediana; la caja va del 25 % al 75 % de los precios; los "
                        "puntos sueltos son casas atípicas. La escala es logarítmica para poder ver juntos precios de USD 100.000 "
                        "y de USD 7 millones. Las cajas con vista están más arriba y son más anchas: su precio varía mucho más.")
            perfil = self.perfil_p3(d)
            st.markdown("**Perfil de las casas con vista frente a las demás**")
            st.dataframe(perfil, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(c, format="%.1f") for c in perfil.columns
                                        if c not in ("Grupo", "Viviendas")})
            st.caption("Las casas con vista también son más grandes y mejor calificadas: parte de su precio viene de eso.")
        self._grafico(self.figura_p3_mapa(d))
        st.markdown("**Cómo leer el mapa:** cada punto es una casa con vista (azul: lago o río; naranja: mar).")
        st.caption("Limitación: la clasificación mar/lago la hizo NVIDIA Nemotron y no está validada; hay casos dudosos "
                   "(por ejemplo, viviendas marcadas como mar cuyo cuerpo de agua cercano es un lago). Con tan pocas casas, "
                   "unas pocas ventas cambian el resultado.")
        self._tabla(d, "Vista al mar o lago", "por tipo de vista", ORDEN_VISTA)

    def tab_p4(self, d, n_top: int) -> None:
        self._encabezado("P4", d)
        c1, c2 = st.columns(2)
        self._grafico(self.figura_p4_subregion(d), c1)
        self._grafico(self.figura_p4_anillo(d), c2)
        st.markdown("#### Cruce de zona y tamaño (mapa de calor)")
        medida = st.radio("Qué mostrar en cada casilla:", list(self.MEDIDAS_CALOR), horizontal=True, key="p4_calor")
        self._grafico(self.figura_p4_calor(d, medida))
        st.markdown("**Cómo leer el cuadro:** busca la fila de la zona y la columna del tamaño de la casa; el número de la "
                    "casilla es el valor elegido arriba. Entre más oscura la casilla, más alto el valor. Las casillas grises "
                    f"tienen menos de {self.N_MINIMO_CASILLA} casas, así que su valor no es confiable. Pasa el cursor para ver "
                    "las tres medidas a la vez.")
        explicacion = self.explicar_calor(d)
        if explicacion:
            st.info(_esc(explicacion))
        st.markdown("**Qué muestra:** el precio por m² depende de la zona (las filas de arriba son más oscuras) y del tamaño: "
                    "en casi todas las zonas el m² de una casa pequeña cuesta más que el de una grande. En cambio, el precio "
                    "total de la casa sí sube con el tamaño (elige *Precio típico de la casa* para verlo).")
        self._grafico(self.figura_p4_mapa_m2(d))
        st.markdown("**Cómo leer el mapa:** cada punto es una vivienda; rojo es m² caro y amarillo m² barato. "
                    "La escala va del 5 % al 95 % de los precios para que los extremos no apaguen los colores.")
        st.markdown("#### El entorno: transporte, comercio y parques")
        self._grafico(self.figura_p4_entorno(d))
        st.markdown("**Cómo leerlo:** cada barra es el precio típico de las casas con ese nivel de entorno; debajo dice cuántas "
                    "casas son. La línea punteada es la mediana de todas las casas: las barras **azules** quedan por encima y las "
                    "**naranjas** por debajo. Pasa el cursor sobre una barra para ver en qué zona están esas casas.")
        for frase in self.explicar_entorno(d):
            st.markdown(f"- {frase}")
        st.markdown("**Conclusión:** el entorno por sí solo no explica el precio. Un nivel sale caro cuando sus casas están en "
                    "una zona cara (por ejemplo, el Eastside) y son más grandes y mejor calificadas. Por eso transporte 'Medio' "
                    "puede salir más caro que 'Alto': muchas casas de transporte 'Alto' están en zonas de precio medio de Seattle "
                    "y el sur del condado.")
        st.caption("Limitación: transporte, comercio y parques son etiquetas inferidas por NVIDIA Nemotron por cuadrante de "
                   "~4–5 km, no mediciones; se muestran como orientación.")
        top = self.tabla_p4_barrios(d, n_top)
        st.markdown(f"**Top {n_top} barrios por precio típico** (con al menos {N_MINIMO_BARRIO} ventas)")
        st.dataframe(
            top[["Barrio / Sector", "Municipio / Subregión", "viviendas", "mediana", "mediana_m2"]],
            hide_index=True, width="stretch",
            column_config={
                "viviendas": st.column_config.NumberColumn("Viviendas", format="%d"),
                "mediana": st.column_config.ProgressColumn("Mediana (USD)", format="%.0f", min_value=0,
                                                           max_value=float(top["mediana"].max()) if len(top) else 1.0),
                "mediana_m2": st.column_config.NumberColumn("USD/m²", format="%.0f"),
            },
        )
        st.caption("Limitación: barrio, municipio y entorno son inferidos por NVIDIA Nemotron por cuadrante; son orientativos.")
        self._tabla(d, "Municipio / Subregión", "por subregión")

    def tab_p5(self, d) -> None:
        self._encabezado("P5", d)
        c1, c2 = st.columns([1, 2])
        self._grafico(self.figura_p5_torta(d), c1)
        self._grafico(self.figura_p5_decadas(d), c2)
        st.markdown("**Cómo leerlo:** la torta muestra que las remodeladas son pocas. En las barras, para cada década de construcción "
                    "se compara la casa remodelada (naranja) con la no remodelada (azul). Pasa el cursor para ver cuántas casas hay en cada barra.")
        st.caption("Limitación: desde los años noventa hay muy pocas casas remodeladas (6 en los 90, 1 en los 2000 y ninguna en los 2010), "
                   "así que esas barras no son comparables.")
        self._tabla(d, ["Década de construcción", "Condición de remodelación"], "por década y remodelación")

    def tab_p6(self, d) -> None:
        self._encabezado("P6", d)
        c1, c2 = st.columns(2)
        self._grafico(self.figura_p6_ventas(d), c1)
        self._grafico(self.figura_p6_mediana(d), c2)
        c3, c4 = st.columns(2)
        self._grafico(self.figura_p6_variacion(d), c3)
        self._grafico(self.figura_p6_apilado(d), c4)
        st.markdown("**Cómo leerlo:** los picos de la primera línea son meses con más ventas; la segunda muestra si el precio típico "
                    "sube o baja frente a la línea punteada (mediana del periodo); las barras de variación muestran cuánto cambió frente al mes anterior.")
        st.caption("Limitación: mayo de 2015 es parcial. Un solo año no separa estacionalidad de tendencia.")
        self._tabla(d, "Periodo de venta", "por mes")

    def tab_proyecto(self) -> None:
        st.subheader("Necesidad")
        st.write("Los peritos de una compañía de avalúos (escenario hipotético) necesitan rangos de referencia objetivos para "
                 "el precio de venta de una vivienda según sus características, en lugar de depender solo de su experiencia.")
        st.subheader("Dataset")
        st.write("`reg_cat_house_sales` del Tabular Benchmark (INRIA, Hugging Face), con origen en el King County Department "
                 "of Assessments. Cada fila es una vivienda vendida entre mayo de 2014 y mayo de 2015. El precio viene como "
                 "ln(USD + 1) y se convierte a USD con `expm1`.")
        st.subheader("Recorrido del proyecto (reuso de las entregas)")
        st.dataframe(pd.DataFrame({
            "Entrega": ["AE1", "AE2", "AE3", "AE4"],
            "Qué aportó": ["Clases de carga y reporte (HFDatasetLoader, Reportes) y ejecución automática en GitHub Actions.",
                           "Necesidad, preguntas de negocio y diccionario inicial del dataset de viviendas.",
                           "Limpieza con auditoría, 25 variables nuevas (USD, m², distancias, Nemotron) y tablas P1–P6.",
                           "Este dashboard: reutiliza el dataset de AE3, la función de tablas de Reportes, la auditoría y el diccionario."],
        }), hide_index=True, width="stretch")
        st.subheader("Transformación: antes y después")
        if self.auditoria is not None:
            st.dataframe(self.auditoria, hide_index=True, width="stretch")
            st.caption("Generado con PreprocesadorHouseSales.comparar_antes_despues (AE3) desde el notebook de AE4.")
            if self.auditoria_columnas is not None:
                with st.expander("Detalle por columna (43 variables)"):
                    st.dataframe(self.auditoria_columnas, hide_index=True, width="stretch")
        else:
            st.dataframe(pd.DataFrame({
                "Indicador": ["Filas", "Columnas", "Valores nulos", "Filas duplicadas"],
                "Antes (crudo)": [21613, 18, 0, 0],
                "Después (preparado)": [21521, 43, 0, 0],
            }), hide_index=True, width="stretch")
        if self.diccionario is not None:
            with st.expander("Diccionario de las variables creadas en AE3"):
                st.dataframe(self.diccionario, hide_index=True, width="stretch")
        st.subheader("Conclusiones y recomendaciones")
        st.markdown("- El tamaño, la calificación constructiva y la ubicación son los factores más asociados al precio.\n"
                    "- La vista al agua marca diferencias grandes, pero en menos del 1 % de las viviendas: es un caso aparte.\n"
                    "- La remodelación se asocia con mayor precio en casi todas las décadas de construcción.\n"
                    "- Recomendación: rangos por tamaño y calificación; resultados por barrio como orientativos hasta validarlos.")
        st.subheader("Limitaciones")
        st.markdown("- Análisis descriptivo: no establece causas ni predice precios.\n"
                    "- Barrio, municipio, entorno y vista mar/lago vienen de NVIDIA Nemotron y no están validados.\n"
                    "- Los datos son de 2014–2015; mayo de 2015 es parcial.")

    def tab_datos(self, d) -> None:
        st.subheader("Datos filtrados")
        columnas = ["Fecha de venta", "Precio (USD)", "Precio por m2 (USD)", "Área habitable (m2)", "Habitaciones", "Baños",
                    "Calificación", "Nivel de calificación", "Categoría de tamaño", "Década de construcción",
                    "Condición de remodelación", "Vista al mar o lago", "Municipio / Subregión", "Barrio / Sector"]
        st.dataframe(d[columnas], hide_index=True, width="stretch", height=420)
        st.download_button("Descargar CSV filtrado", d[columnas].to_csv(index=False).encode("utf-8"),
                           file_name="viviendas_filtradas.csv", mime="text/csv")

    # ---------- página ----------
    SECCIONES = ["🎯 Respuestas", "P1 Tamaño", "P2 Calificación", "P3 Vista al agua", "P4 Ubicación",
                 "P5 Remodelación", "P6 Mensual", "📘 Proyecto", "🗂 Datos"]

    def _limites(self) -> dict:
        d = self.datos
        decadas = sorted(d["Década de construcción"].unique())
        return {
            "fechas": (d["Fecha de venta"].min().date(), d["Fecha de venta"].max().date()),
            "decadas": (decadas[0], decadas[-1]),
            "precio": (int(d["Precio (USD)"].min()), int(d["Precio (USD)"].max())),
            "habitaciones": (int(d["Habitaciones"].min()), int(d["Habitaciones"].max())),
        }

    def _restablecer(self) -> None:
        lim = self._limites()
        st.session_state["f_tamanos"] = ORDEN_TAMANOS
        st.session_state["f_calificacion"] = ORDEN_CALIFICACION
        st.session_state["f_municipios"] = sorted(self.datos["Municipio / Subregión"].dropna().unique())
        st.session_state["f_fechas"] = lim["fechas"]
        st.session_state["f_top"] = 10
        st.session_state["f_vista"] = ORDEN_VISTA
        st.session_state["f_remodelacion"] = ["No remodelada", "Remodelada"]
        st.session_state["f_anillo"] = ORDEN_ANILLO
        st.session_state["f_decadas"] = lim["decadas"]
        st.session_state["f_precio"] = lim["precio"]
        st.session_state["f_habitaciones"] = lim["habitaciones"]

    def render(self) -> None:
        municipios = sorted(self.datos["Municipio / Subregión"].dropna().unique())
        fmin, fmax = self.datos["Fecha de venta"].min().date(), self.datos["Fecha de venta"].max().date()
        with st.sidebar:
            st.header("Filtros")
            tamanos = st.multiselect("Tamaño habitable", ORDEN_TAMANOS, default=ORDEN_TAMANOS, key="f_tamanos")
            calif = st.multiselect("Calificación constructiva", ORDEN_CALIFICACION, default=ORDEN_CALIFICACION, key="f_calificacion")
            muni = st.multiselect("Municipio / subregión", municipios, default=municipios, key="f_municipios")
            fechas = st.date_input("Rango de fechas de venta", (fmin, fmax), min_value=fmin, max_value=fmax,
                                   format="DD/MM/YYYY", key="f_fechas")
            lim = self._limites()
            with st.expander("Más filtros"):
                vista = st.multiselect("Vista al agua", ORDEN_VISTA, default=ORDEN_VISTA, key="f_vista")
                remodelacion = st.multiselect("Condición de remodelación", ["No remodelada", "Remodelada"],
                                              default=["No remodelada", "Remodelada"], key="f_remodelacion")
                anillo = st.multiselect("Anillo urbano (distancia al centro comercial)", ORDEN_ANILLO,
                                        default=ORDEN_ANILLO, key="f_anillo")
                decadas = st.select_slider("Década de construcción", sorted(self.datos["Década de construcción"].unique()),
                                           value=lim["decadas"], key="f_decadas")
                precio = st.slider("Rango de precio (USD)", lim["precio"][0], lim["precio"][1], lim["precio"],
                                   step=10_000, key="f_precio")
                habitaciones = st.slider("Habitaciones", lim["habitaciones"][0], lim["habitaciones"][1],
                                         lim["habitaciones"], key="f_habitaciones")
            n_top = st.slider("Barrios en el ranking (P4)", 5, 25, 10, key="f_top")
            st.button("Restablecer filtros", on_click=self._restablecer, width="stretch")
            st.divider()
            st.caption("Fuente: King County Department of Assessments, vía Tabular Benchmark (INRIA) en Hugging Face. "
                       "Ventas reales de mayo 2014 a mayo 2015. Barrio, municipio, entorno y vista mar/lago fueron "
                       "inferidos con NVIDIA Nemotron y no están validados.")

        st.title("Factores asociados al precio de venta de viviendas en King County")
        st.info(f"**Pregunta de negocio:** {PREGUNTA_NEGOCIO}")
        with st.expander("Cómo usar este dashboard"):
            st.markdown("1. Ajusta los filtros de la izquierda (abre **Más filtros** para vista, remodelación, década, precio y habitaciones); todos los indicadores, gráficos y respuestas se recalculan.\n"
                        "2. Elige una sección en la barra de botones de arriba. **Respuestas** resume las seis preguntas con su decisión.\n"
                        "3. Cada sección **P1–P6** tiene los gráficos de una pregunta, cómo leerlos y su tabla estadística.\n"
                        "4. **Proyecto** explica la necesidad, el dataset y la transformación; **Datos** permite descargar la selección.\n"
                        "5. Pasa el cursor sobre los gráficos para ver los valores exactos.")

        if not (isinstance(fechas, tuple) and len(fechas) == 2):
            st.info("Selecciona fecha de inicio y de fin en el filtro.")
            return
        d = self.filtrar(tamanos, calif, muni, fechas, extra={
            "vista": vista, "remodelacion": remodelacion, "anillo": anillo, "decadas": decadas,
            "precio": precio, "habitaciones": habitaciones})
        if d.empty:
            st.warning("Ningún registro cumple los filtros actuales.")
            return
        st.caption(f"Periodo: {fechas[0]:%d/%m/%Y} – {fechas[1]:%d/%m/%Y} · {numero(len(d))} viviendas · "
                   f"{d['Municipio / Subregión'].nunique()} subregiones")

        indicadores = list(self.kpis(d).items())
        for fila in (indicadores[:3], indicadores[3:]):
            for col, (nombre, (valor, delta)) in zip(st.columns(3), fila):
                col.metric(nombre, valor, delta=delta, delta_color="off", border=True)

        # Selector de sección en lugar de pestañas: solo se dibuja la sección elegida. Así los mapas se inicializan
        # con su tamaño real (en una pestaña oculta salen vacíos) y la página responde más rápido.
        seccion = st.segmented_control("Sección", self.SECCIONES, default=self.SECCIONES[0], key="seccion",
                                       label_visibility="collapsed") or self.SECCIONES[0]
        st.divider()
        if seccion == self.SECCIONES[0]:
            st.subheader("Seis preguntas, seis respuestas (calculadas con la selección actual)")
            for r in self.respuestas(d):
                with st.container(border=True):
                    st.markdown(f"**{r['id']}. {r['pregunta']}**")
                    st.markdown(_esc(r["respuesta"]))
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"📏 **Indicador:** {r['indicador']}")
                    c2.markdown(f"📊 **Cómo se ve:** {r['visual']}")
                    c3.markdown(f"✅ **Decisión:** {r['decision']}")
        elif seccion == self.SECCIONES[1]:
            self.tab_p1(d)
        elif seccion == self.SECCIONES[2]:
            self.tab_p2(d)
        elif seccion == self.SECCIONES[3]:
            self.tab_p3(d)
        elif seccion == self.SECCIONES[4]:
            self.tab_p4(d, n_top)
        elif seccion == self.SECCIONES[5]:
            self.tab_p5(d)
        elif seccion == self.SECCIONES[6]:
            self.tab_p6(d)
        elif seccion == self.SECCIONES[7]:
            self.tab_proyecto()
        else:
            self.tab_datos(d)
