"""Aplicación web de la AE4 (precios de vivienda en King County).

Ejecutar en local, desde la raíz del repositorio:
    streamlit run src/AE4/app.py

En Streamlit Community Cloud: Main file path = src/AE4/app.py
"""
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Precios de vivienda · King County", page_icon="🏠", layout="wide")  # primera llamada

AQUI = Path(__file__).resolve().parent              # src/AE4
sys.path.insert(0, str(AQUI))
from dashboard_ae4 import DashboardAE4              # noqa: E402

CARPETA_DATOS = AQUI / "datos"


@st.cache_resource(show_spinner="Cargando 21.521 viviendas…")
def cargar() -> DashboardAE4:
    return DashboardAE4.desde_carpeta(CARPETA_DATOS)


cargar().render()
