"""Aplicación Streamlit de la AE4: métricas y gráficos por pregunta de negocio.

Lee las tablas de resultados de AE3 (carpeta datos/). No recalcula el análisis:
las cifras salen de los CSV que genera el notebook / script de la AE3.

Ejecutar desde esta carpeta:
    pip install -r requirements_ae4.txt
    streamlit run app_streamlit.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

DATOS = Path(__file__).parent / "datos"


@st.cache_data
def cargar(nombre: str) -> pd.DataFrame:
    return pd.read_csv(DATOS / nombre)


def usd(x: float) -> str:
    return f"USD {x:,.0f}".replace(",", ".")


st.set_page_config(page_title="Precios de vivienda - King County", layout="wide")
st.title("Factores asociados al precio de venta de viviendas en King County")
st.caption("Proyecto Integrador III · Analítica de datos · Ventas mayo 2014 – mayo 2015 · 21.521 viviendas")

# ---------- 1. Necesidad y dataset ----------
st.header("1. Necesidad y dataset")
st.write(
    "Los peritos de la compañía necesitan rangos de referencia objetivos para el precio de venta de una vivienda, "
    "según sus características. Escenario hipotético."
)
st.write(
    "Pregunta de análisis: ¿cómo varía el precio de venta según tamaño, calificación de construcción, "
    "vista al agua, antigüedad y remodelación?"
)
st.write(
    "Dataset: `reg_cat_house_sales` (Tabular Benchmark, INRIA). Cada fila es una vivienda vendida. "
    "Se excluyen 92 registros con 0 habitaciones, menos de 1 baño o 33 habitaciones (21.613 → 21.521)."
)

# ---------- 2. Pregunta 1 ----------
st.header("2. Pregunta 1 · Tamaño y capacidad")
st.markdown(
    "**Métrica:** mediana del precio de venta (USD) por categoría de área habitable y por número de habitaciones. "
    "Unidad: USD y viviendas (n)."
)
p1 = cargar("tabla_p1_tamano_vivienda.csv")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Mediana por tamaño habitable")
    st.bar_chart(p1.set_index("Categoría de tamaño")["Mediana (USD)"])
with c2:
    p1b = cargar("tabla_p1b_habitaciones.csv")
    st.subheader("Mediana por número de habitaciones")
    st.bar_chart(p1b.set_index("Habitaciones")["Mediana (USD)"])
st.markdown(
    f"**Interpretación:** la mediana sube de {usd(p1['Mediana (USD)'].iloc[0])} en viviendas pequeñas "
    f"a {usd(p1['Mediana (USD)'].iloc[-1])} en las de 280 m² o más. "
    "**Decisión:** rangos de referencia por tamaño para una primera estimación."
)
st.caption(
    "Limitación: el precio por m² de las muy grandes no crece igual que el precio total; la mediana por "
    "habitaciones mezcla tamaño y distribución de la casa."
)

# ---------- 3. Pregunta 2 ----------
st.header("3. Pregunta 2 · Calificación constructiva")
st.markdown("**Métrica:** mediana del precio de venta (USD) por nivel de calificación (escala 4–13 agrupada en 4 estratos).")
p2 = cargar("tabla_p2_calificacion_constructiva.csv")
st.bar_chart(p2.set_index("Nivel de calificación")["Mediana (USD)"])
st.markdown(
    "**Interpretación:** del estrato Estándar (7) al de Alta/Lujo (10–13) la mediana pasa de USD 375.000 a USD 1.038.000. "
    "**Decisión:** la calificación es un criterio central para asignar rangos. "
    "**Limitación:** es una asociación descriptiva; no separa el efecto de tamaño o ubicación."
)

# ---------- 4. Pregunta 3 ----------
st.header("4. Pregunta 3 · Vista al agua")
st.markdown("**Métrica:** mediana del precio (USD) y del precio por m² según tipo de vista.")
p3 = cargar("tabla_p3_vista_mar_lago.csv")
st.bar_chart(p3.set_index("Vista al mar o lago")["Mediana (USD)"])
st.markdown(
    "**Interpretación:** con vista al mar (n = 76) la mediana es USD 908.250; con vista a lago interior (n = 81), USD 1.955.000; "
    "sin vista directa (n = 21.364), USD 450.000. "
    "**Limitación:** muestra muy pequeña en vista al agua; la clasificación mar/lago viene de un modelo de lenguaje sin validar."
)

# ---------- 5. Pregunta 4 ----------
st.header("5. Pregunta 4 · Ubicación, barrios y entorno")
st.markdown("**Métrica:** mediana del precio (USD) por subregión y por barrio (barrios con n ≥ 50).")
p4 = cargar("tabla_p4_subregiones_barrios.csv")
st.bar_chart(p4.set_index("Municipio / Subregión")["Mediana (USD)"])
p4b = cargar("tabla_p4b_barrios_top15.csv")
st.subheader("Top 15 barrios por mediana")
st.bar_chart(p4b.set_index("Barrio / Sector")["Mediana (USD)"])
st.markdown(
    "**Interpretación:** Medina / Clyde Hill (USD 1.020.000; USD 4.479,17/m²) y Downtown Bellevue (USD 987.000) "
    "encabezan la lista; Seattle tiene la mediana por m² más alta de las subregiones (USD 3.470,95/m²). "
    "**Limitación:** barrios, municipio, transporte, parques y comercio son inferidos por NVIDIA Nemotron a partir de coordenadas; "
    "hay que validarlos contra una fuente pública antes de tomar decisiones."
)

# ---------- 6. Pregunta 5 ----------
st.header("6. Pregunta 5 · Antigüedad y remodelación")
st.markdown("**Métrica:** mediana del precio (USD) según condición de remodelación.")
p5b = cargar("tabla_p5b_resumen_remodelacion.csv")
st.bar_chart(p5b.set_index("Condición de remodelación")["Mediana (USD)"])
st.markdown(
    f"**Interpretación:** las viviendas remodeladas (4,23 %, n = 910) tienen una mediana de "
    f"{usd(p5b['Mediana (USD)'].iloc[0])} frente a {usd(p5b['Mediana (USD)'].iloc[1])} de las no remodeladas. "
    "**Limitación:** no se muestra por década de construcción en esta versión; la comparación entre cohortes antiguas y "
    "nuevas queda pendiente de respaldo."
)

# ---------- 7. Pregunta 6 ----------
st.header("7. Pregunta 6 · Dinámica mensual")
st.markdown("**Métrica:** número de ventas y mediana del precio (USD) por mes.")
p6 = cargar("tabla_p6_periodo_ventas.csv")
st.line_chart(p6.set_index("Periodo de venta")["Mediana (USD)"])
st.bar_chart(p6.set_index("Periodo de venta")["Viviendas (n)"])
st.markdown(
    "**Interpretación:** la mediana se mantiene entre USD 428.000 y USD 478.250 en casi todo el periodo. "
    "**Limitación:** mayo de 2015 es parcial (datos hasta el 27 de mayo; 642 ventas), por eso no se compara directamente."
)

# ---------- 8. Conclusiones ----------
st.header("8. Conclusiones, recomendaciones y limitaciones")
st.markdown(
    "- El tamaño y la calificación constructiva son los factores descriptivos más asociados al precio.\n"
    "- La vista al agua y el barrio marcan diferencias grandes, pero sobre muestras pequeñas o atributos inferidos.\n"
    "- Recomendación: usar rangos por tamaño y calificación; tratar los resultados por barrio como orientativos.\n"
    "- Limitación: análisis descriptivo; no establece causas ni predice precios."
)
