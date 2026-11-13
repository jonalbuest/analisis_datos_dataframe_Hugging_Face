"""Dashboard de la AE4: precios de vivienda en King County con Streamlit + Plotly.

La clase :class:`DashboardAE4` toma el dataset enriquecido de AE3 y:

1. aplica filtros (barra lateral) y recalcula cada métrica sobre la selección;
2. dibuja cada gráfico como figura de Plotly, con título, etiquetas y unidades;
3. redacta la respuesta de cada pregunta de negocio con cifras calculadas, no escritas a mano;
4. explica en cada pestaña qué se ve, cómo leerlo y qué decisión apoya.

Con los filtros en su valor inicial, las cifras coinciden con las tablas de AE3
(mediana global USD 450.000, 21.521 viviendas).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

N_MINIMO_BARRIO = 50          # mismo criterio que la tabla P4 de AE3
UMBRAL_BIEN_CALIFICADA = 8    # estratos Media-Alta (8–9) y Alta/Lujo (10–13)

ORDEN_TAMANOS = ["1. Pequeña (<130 m²)", "2. Mediana (130–199 m²)", "3. Grande (200–279 m²)", "4. Muy grande (≥280 m²)"]
ORDEN_CALIFICACION = ["1. Baja (4–6)", "2. Estándar (7)", "3. Media-Alta (8–9)", "4. Alta / Lujo (10–13)"]
ORDEN_VISTA = ["Sin vista directa al agua", "Vista a lago / río interior", "Vista al mar (Puget Sound)"]
COLORES_CALIFICACION = {ORDEN_CALIFICACION[0]: "#9ecae1", ORDEN_CALIFICACION[1]: "#6baed6",
                        ORDEN_CALIFICACION[2]: "#2171b5", ORDEN_CALIFICACION[3]: "#08306b"}
COLOR_PRINCIPAL = "#2171b5"


def usd(x: float) -> str:
    return f"USD {x:,.0f}".replace(",", ".")


def numero(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def fuerza_corr(r: float) -> str:
    a = abs(r)
    if a >= 0.7:
        return "fuerte"
    if a >= 0.4:
        return "moderada"
    if a >= 0.2:
        return "débil"
    return "muy débil o nula"


class DashboardAE4:
    def __init__(self, datos: pd.DataFrame) -> None:
        self.datos = datos.copy()
        self.datos["Fecha de venta"] = pd.to_datetime(self.datos["Fecha de venta"])
        self.datos["Bien calificada"] = self.datos["Calificación"] >= UMBRAL_BIEN_CALIFICADA

    @classmethod
    def desde_csv(cls, ruta: str | Path) -> "DashboardAE4":
        return cls(pd.read_csv(ruta))

    # ---------- filtros ----------
    def filtrar(self, tamanos, calificaciones, municipios, fechas) -> pd.DataFrame:
        d = self.datos
        d = d[d["Categoría de tamaño"].isin(tamanos)]
        d = d[d["Nivel de calificación"].isin(calificaciones)]
        d = d[d["Municipio / Subregión"].isin(municipios)]
        ini, fin = fechas
        return d[(d["Fecha de venta"] >= pd.Timestamp(ini)) & (d["Fecha de venta"] <= pd.Timestamp(fin))]

    # ---------- métricas ----------
    @staticmethod
    def resumen(d: pd.DataFrame, columna: str, orden: list[str] | None = None) -> pd.DataFrame:
        r = d.groupby(columna).agg(
            viviendas=("Precio (USD)", "size"),
            mediana=("Precio (USD)", "median"),
            media=("Precio (USD)", "mean"),
            mediana_m2=("Precio por m2 (USD)", "median"),
        ).reset_index()
        if orden is not None:
            r = r.set_index(columna).reindex([o for o in orden if o in set(r[columna])]).reset_index()
        return r

    @staticmethod
    def correlacion(d: pd.DataFrame, a: str, b: str) -> float:
        return float(d[a].corr(d[b], method="pearson"))

    @staticmethod
    def _estilo(fig: go.Figure, alto: int = 400) -> go.Figure:
        fig.update_layout(height=alto, margin=dict(l=10, r=10, t=60, b=10), template="plotly_white")
        return fig

    def barras(self, df, x, y, titulo, etiqueta, horizontal=False, orden_desc=False, colores=None) -> go.Figure:
        if horizontal:
            fig = px.bar(df, x=y, y=x, orientation="h", text=y, title=titulo,
                         color=colores, color_discrete_sequence=[COLOR_PRINCIPAL] if colores is None else None)
            fig.update_layout(xaxis_title=etiqueta, yaxis_title="")
            fig.update_yaxes(categoryorder="total ascending" if orden_desc else "trace")
        else:
            fig = px.bar(df, x=x, y=y, text=y, title=titulo,
                         color=colores, color_discrete_sequence=[COLOR_PRINCIPAL] if colores is None else None)
            fig.update_layout(xaxis_title="", yaxis_title=etiqueta)
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        return self._estilo(fig)

    def mapa_puntos(self, df, color_col, titulo, colores, orden) -> go.Figure:
        """Mapa sencillo: longitud en horizontal y latitud en vertical. No necesita mapa base."""
        fig = px.scatter(
            df, x="Longitud", y="Latitud", color=color_col, opacity=0.6,
            color_discrete_map=colores, category_orders={color_col: orden},
            hover_data={"Precio (USD)": ":,.0f", "Municipio / Subregión": True,
                        "Longitud": False, "Latitud": False},
            title=titulo, labels={"Longitud": "Longitud (este →)", "Latitud": "Latitud (norte ↑)"},
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(legend_title_text="", height=480, margin=dict(l=10, r=10, t=60, b=10),
                          template="plotly_white")
        return fig

    # ---------- respuestas de negocio (calculadas) ----------
    def respuesta_1(self, d) -> str:
        r = self.resumen(d, "Categoría de tamaño", ORDEN_TAMANOS)
        if r.empty:
            return "Sin datos con los filtros actuales."
        r_ = self.correlacion(d, "Precio (USD)", "Área habitable (m2)")
        return (
            f"La mediana sube de {usd(r['mediana'].iloc[0])} a {usd(r['mediana'].iloc[-1])} entre la categoría "
            f"más pequeña y la más grande. La correlación entre precio y área habitable es r = {r_:.2f} "
            f"({fuerza_corr(r_)}, positiva). Decisión: rangos de referencia por tamaño."
        )

    def respuesta_2(self, d) -> str:
        bien = d[d["Bien calificada"]]
        resto = d[~d["Bien calificada"]]
        if bien.empty or resto.empty:
            return "Se necesitan viviendas bien calificadas y otras que no lo estén para comparar."
        return (
            f"{numero(len(bien))} viviendas ({len(bien) / max(len(d), 1):.0%}) están bien calificadas (8 o más). "
            f"Su mediana es {usd(bien['Precio (USD)'].median())}, frente a {usd(resto['Precio (USD)'].median())} "
            f"del resto. La calificación se relaciona con el área (r = {self.correlacion(d, 'Calificación', 'Área habitable (m2)'):.2f}). "
            "Decisión: buscar las mejores calificaciones en las zonas del mapa."
        )

    def respuesta_3(self, d) -> str:
        con_vista = int((d["Vista al mar o lago"] != "Sin vista directa al agua").sum())
        return (
            f"Solo {con_vista} de {numero(len(d))} viviendas ({con_vista / max(len(d), 1):.2%}) tienen vista al agua. "
            "Por eso es un caso aparte, no una tendencia general del mercado."
        )

    def respuesta_4(self, d) -> str:
        r = self.resumen(d, "Municipio / Subregión").sort_values("mediana_m2", ascending=False)
        if r.empty:
            return "Sin datos con los filtros actuales."
        return (
            f"La subregión con mayor precio por m² es {r['Municipio / Subregión'].iloc[0]} "
            f"({usd(r['mediana_m2'].iloc[0])}/m²). La más barata es {r['Municipio / Subregión'].iloc[-1]} "
            f"({usd(r['mediana_m2'].iloc[-1])}/m²). Decisión: priorizar la ubicación al cotizar."
        )

    def respuesta_5(self, d) -> str:
        r = self.resumen(d, "Condición de remodelación")
        rem = r[r["Condición de remodelación"] == "Remodelada"]
        no = r[r["Condición de remodelación"] == "No remodelada"]
        if rem.empty or no.empty:
            return "La selección incluye solo un tipo de condición; no hay comparación."
        pct = rem["viviendas"].iloc[0] / max(r["viviendas"].sum(), 1)
        return (
            f"Solo el {pct:.1%} de las viviendas está remodelado ({int(rem['viviendas'].iloc[0])}). Su mediana es "
            f"{usd(rem['mediana'].iloc[0])}, frente a {usd(no['mediana'].iloc[0])} de las no remodeladas."
        )

    def respuesta_6(self, d) -> str:
        r = self.resumen(d, "Periodo de venta").sort_values("Periodo de venta")
        if r.empty:
            return "Sin datos con los filtros actuales."
        mx = r.loc[r["viviendas"].idxmax()]
        mn = r.loc[r["viviendas"].idxmin()]
        return (
            f"El mes con más ventas es {mx['Periodo de venta']} ({int(mx['viviendas'])}); el de menos, "
            f"{mn['Periodo de venta']} ({int(mn['viviendas'])}). Mayo de 2015 es parcial (datos hasta el 27)."
        )

    # ---------- pestañas ----------
    def tab_pregunta_1(self, d) -> None:
        st.markdown(
            "**Qué muestra:** cuánto cuesta la casa típica (la mediana) según su tamaño y su número de cuartos, "
            "y qué tanto se parecen el precio y el área."
        )
        r = self.resumen(d, "Categoría de tamaño", ORDEN_TAMANOS)
        rh = self.resumen(d, "Habitaciones").query("Habitaciones.between(1, 6)")
        c1, c2 = st.columns(2)
        c1.plotly_chart(self.barras(r, "Categoría de tamaño", "mediana",
                                    "Precio típico según tamaño de la casa", "Mediana del precio (USD)",
                                    horizontal=True, orden_desc=True), use_container_width=True)
        c2.plotly_chart(self.barras(rh, "Habitaciones", "mediana",
                                    "Precio típico según número de cuartos (1 a 6)", "Mediana del precio (USD)"),
                        use_container_width=True)
        st.markdown(
            "**Cómo leerlo:** cada barra es un grupo de casas. Entre más alta la barra, más cuesta la casa típica de ese grupo."
        )

        cor = pd.DataFrame({
            "Característica": ["Área habitable", "Calificación", "Baños", "Habitaciones", "Área del lote", "Antigüedad"],
            "r": [self.correlacion(d, "Precio (USD)", c) for c in
                  ["Área habitable (m2)", "Calificación", "Baños", "Habitaciones", "Área del lote (m2)", "Antigüedad (años)"]],
        })
        cor["Relación"] = cor["r"].apply(lambda v: "Sube junto con el precio" if v > 0 else "Va en sentido contrario")
        fig = px.bar(cor.sort_values("r"), x="r", y="Característica", orientation="h", text="r",
                     color="Relación", color_discrete_map={"Sube junto con el precio": COLOR_PRINCIPAL,
                                                           "Va en sentido contrario": "#e6550d"},
                     title="Qué tanto se parece el precio a cada característica (correlación r)",
                     labels={"r": "Correlación r (de -1 a 1)", "Característica": ""})
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[-1, 1])
        st.plotly_chart(self._estilo(fig, 380), use_container_width=True)
        st.markdown(
            "**Cómo leerlo (sin tecnicismos):** la correlación va de -1 a 1. Si está cerca de 1, el precio y esa "
            "característica suben juntos. Si está cerca de 0, no tienen que ver. Si es negativa, cuando una sube la otra baja. "
            f"El área habitable (r = {cor.loc[cor['Característica'] == 'Área habitable', 'r'].iloc[0]:.2f}) es lo que más se parece al precio."
        )
        st.caption("Limitación: la correlación muestra asociación, no causa. Una casa grande también suele tener más baños y mejor calificación.")

    def tab_pregunta_2(self, d) -> None:
        st.markdown(
            "**Qué muestra:** cuánto cuesta la casa típica según su calificación, cuántas casas hay en cada nivel "
            "y dónde están las bien calificadas (8 o más)."
        )
        r = self.resumen(d, "Nivel de calificación", ORDEN_CALIFICACION)
        c1, c2 = st.columns(2)
        c1.plotly_chart(self.barras(r, "Nivel de calificación", "mediana",
                                    "Precio típico según nivel de calificación", "Mediana del precio (USD)"),
                        use_container_width=True)
        c2.plotly_chart(self.barras(r, "Nivel de calificación", "viviendas",
                                    "Cuántas casas hay en cada nivel", "Viviendas (n)"),
                        use_container_width=True)
        st.markdown(
            "**Cómo leerlo:** la primera gráfica dice cuánto cuesta la casa típica de cada nivel. "
            "La segunda dice cuántas casas hay en cada nivel: los niveles altos son muy pocos."
        )

        bien = d[d["Bien calificada"]]
        st.plotly_chart(self.mapa_puntos(bien, "Nivel de calificación",
                                         f"Dónde están las viviendas bien calificadas (8 o más): {numero(len(bien))}",
                                         COLORES_CALIFICACION, ORDEN_CALIFICACION),
                        use_container_width=True)
        r_ca = self.correlacion(d, "Calificación", "Área habitable (m2)")
        st.markdown(
            f"**Cómo leerlo:** cada punto es una casa. Los puntos más oscuros son las calificaciones más altas. "
            f"Se ven más hacia la derecha (este), sobre todo en el Eastside. La calificación y el área tienen r = {r_ca:.2f}: "
            "las casas bien calificadas también suelen ser más grandes."
        )
        st.caption("Limitación: la calificación es una nota del condado; no mide por sí sola la calidad del barrio.")

    def tab_pregunta_3(self, d) -> None:
        st.markdown(
            "**Qué muestra:** cuántas casas tienen vista al agua, cuánto cuestan en la mediana y dónde están."
        )
        r = self.resumen(d, "Vista al mar o lago", ORDEN_VISTA)
        con_vista = int((d["Vista al mar o lago"] != "Sin vista directa al agua").sum())
        partes = pd.DataFrame({"Grupo": ["Con vista al agua", "Sin vista directa"],
                               "Viviendas": [con_vista, len(d) - con_vista]})
        c1, c2 = st.columns(2)
        c1.plotly_chart(self.barras(partes, "Grupo", "Viviendas", "Cuántas casas tienen vista al agua",
                                    "Viviendas (n)"), use_container_width=True)
        c2.plotly_chart(self.barras(r, "Vista al mar o lago", "mediana", "Precio típico según vista",
                                    "Mediana del precio (USD)"), use_container_width=True)
        st.plotly_chart(self.mapa_puntos(d, "Vista al mar o lago", "Dónde están las casas, según vista al agua",
                                         {"Sin vista directa al agua": "#d9d9d9",
                                          "Vista a lago / río interior": "#2171b5",
                                          "Vista al mar (Puget Sound)": "#e6550d"}, ORDEN_VISTA),
                        use_container_width=True)
        st.markdown(
            "**Cómo leerlo:** los puntos de colores son las casas con vista al agua. Son pocos frente a los grises. "
            "Por eso no cambian el precio de todo el mercado: es un caso aparte."
        )
        st.caption("Limitación: la clasificación mar/lago la hizo NVIDIA Nemotron y no está validada. Con 157 casas, unas pocas ventas muy caras cambian el promedio.")

    def tab_pregunta_4(self, d, n_top: int) -> None:
        st.markdown(
            "**Qué muestra:** cuánto cuesta un metro cuadrado en cada subregión, un cuadro de colores "
            "por subregión y tamaño, y los barrios más caros."
        )
        r = self.resumen(d, "Municipio / Subregión").sort_values("mediana_m2", ascending=False)
        st.plotly_chart(self.barras(r, "Municipio / Subregión", "mediana_m2",
                                    "Precio por m² en cada subregión (de mayor a menor)",
                                    "Mediana del precio (USD/m²)", horizontal=True, orden_desc=True),
                        use_container_width=True)

        pivote = (d.pivot_table(index="Municipio / Subregión", columns="Categoría de tamaño",
                                values="Precio por m2 (USD)", aggfunc="median")
                  .reindex(columns=[c for c in ORDEN_TAMANOS if c in d["Categoría de tamaño"].unique()]))
        pivote = pivote.reindex(r["Municipio / Subregión"])
        heat = px.imshow(pivote.round(0), text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
                         title="Precio por m² (USD) según subregión (filas) y tamaño de casa (columnas)",
                         labels={"x": "Tamaño de la casa", "y": "Subregión", "color": "USD por m²"})
        st.plotly_chart(self._estilo(heat, 460), use_container_width=True)
        st.markdown(
            "**Cómo leerlo del cuadro:** cada cuadro es el precio por m² de una subregión en un tamaño de casa. "
            "Entre más oscuro, más caro el metro cuadrado. Se lee por filas: Seattle arriba es la más cara; "
            "el sur del condado, abajo, la más barata. Por columnas, el precio por m² **baja** al pasar de casas "
            "pequeñas a grandes en casi todas las subregiones: una casa pequeña cuesta más por cada metro."
        )

        barrios = self.resumen(d, "Barrio / Sector").query("viviendas >= @N_MINIMO_BARRIO")
        top = barrios.sort_values("mediana", ascending=False).head(n_top)
        st.plotly_chart(self.barras(top, "Barrio / Sector", "mediana",
                                    f"Los {n_top} barrios con el precio típico más alto (con al menos {N_MINIMO_BARRIO} casas)",
                                    "Mediana del precio (USD)", horizontal=True, orden_desc=True),
                        use_container_width=True)
        st.caption("Limitación: barrio, municipio y transporte son inferidos por NVIDIA Nemotron a partir de coordenadas; son orientativos.")

    def tab_pregunta_5(self, d) -> None:
        st.markdown(
            "**Qué muestra:** qué parte de las casas está remodelada y cuánto cuesta la casa típica en cada caso."
        )
        r = self.resumen(d, "Condición de remodelación")
        c1, c2 = st.columns(2)
        pie = px.pie(r, names="Condición de remodelación", values="viviendas", hole=0.35,
                     title="Parte de las casas según remodelación",
                     color="Condición de remodelación",
                     color_discrete_map={"Remodelada": "#e6550d", "No remodelada": "#9ecae1"})
        pie.update_traces(textinfo="percent+label")
        c1.plotly_chart(self._estilo(pie), use_container_width=True)
        c2.plotly_chart(self.barras(r, "Condición de remodelación", "mediana",
                                    "Precio típico según remodelación", "Mediana del precio (USD)"),
                        use_container_width=True)
        st.markdown(
            "**Cómo leerlo:** la torta muestra cuántas casas hay de cada tipo (la parte naranja es poca). "
            f"La barra muestra el precio típico. {self.respuesta_5(d)}"
        )
        st.caption("Limitación: no comparamos por década de construcción en esta versión; la comparación de cohortes queda pendiente.")

    def tab_pregunta_6(self, d) -> None:
        st.markdown(
            "**Qué muestra:** cuántas ventas hay cada mes, cómo se reparten por calificación y cómo cambia el precio típico."
        )
        r = self.resumen(d, "Periodo de venta").sort_values("Periodo de venta")
        c1, c2 = st.columns(2)
        c1.plotly_chart(self._estilo(px.line(r, x="Periodo de venta", y="viviendas", markers=True,
                                             title="Ventas por mes", labels={"viviendas": "Ventas (n)", "Periodo de venta": "Mes"})),
                        use_container_width=True)
        c2.plotly_chart(self._estilo(px.line(r, x="Periodo de venta", y="mediana", markers=True,
                                             title="Precio típico por mes", labels={"mediana": "Mediana del precio (USD)", "Periodo de venta": "Mes"})),
                        use_container_width=True)
        apilado = px.bar(
            d.groupby(["Periodo de venta", "Nivel de calificación"]).size().reset_index(name="Ventas"),
            x="Periodo de venta", y="Ventas", color="Nivel de calificación",
            category_orders={"Nivel de calificación": ORDEN_CALIFICACION, "Periodo de venta": sorted(d["Periodo de venta"].unique())},
            color_discrete_map=COLORES_CALIFICACION, title="Ventas por mes, según nivel de calificación",
            labels={"Ventas": "Ventas (n)", "Periodo de venta": "Mes"},
        )
        st.plotly_chart(self._estilo(apilado, 420), use_container_width=True)
        st.markdown(
            "**Cómo leerlo:** en la primera gráfica, los picos son meses con más ventas. La segunda muestra si el precio típico "
            "sube o baja. En la última, cada color es un nivel de calificación. "
            f"{self.respuesta_6(d)}"
        )
        st.caption("Limitación: mayo de 2015 es parcial. Un año no separa estacionalidad de tendencia.")

    # ---------- página ----------
    def render(self) -> None:
        with st.sidebar:
            st.header("Filtros")
            tamanos = st.multiselect("Tamaño habitable", ORDEN_TAMANOS, default=ORDEN_TAMANOS)
            calif = st.multiselect("Calificación constructiva", ORDEN_CALIFICACION, default=ORDEN_CALIFICACION)
            municipios = sorted(self.datos["Municipio / Subregión"].dropna().unique())
            muni = st.multiselect("Municipio / subregión", municipios, default=municipios)
            fmin, fmax = self.datos["Fecha de venta"].min().date(), self.datos["Fecha de venta"].max().date()
            fechas = st.date_input("Rango de fechas de venta", (fmin, fmax), min_value=fmin, max_value=fmax)
            n_top = st.slider("Barrios en el ranking (P4)", 5, 25, 10)

        if not (isinstance(fechas, tuple) and len(fechas) == 2):
            st.info("Selecciona fecha de inicio y de fin en el filtro.")
            return
        d = self.filtrar(tamanos, calif, muni, fechas)
        st.metric("Viviendas en la selección", numero(len(d)))
        if d.empty:
            st.warning("Ningún registro cumple los filtros actuales.")
            return

        tabs = st.tabs(["🎯 Respuestas de negocio", "P1 Tamaño", "P2 Calificación", "P3 Vista al agua",
                        "P4 Ubicación", "P5 Remodelación", "P6 Mensual"])
        with tabs[0]:
            st.subheader("Respuestas de negocio (calculadas con la selección actual)")
            for titulo, fn in [
                ("1. Tamaño y capacidad", self.respuesta_1),
                ("2. Calificación constructiva", self.respuesta_2),
                ("3. Vista al agua", self.respuesta_3),
                ("4. Ubicación y barrios", self.respuesta_4),
                ("5. Antigüedad y remodelación", self.respuesta_5),
                ("6. Dinámica mensual", self.respuesta_6),
            ]:
                st.markdown(f"**{titulo}.** {fn(d)}")
        with tabs[1]:
            self.tab_pregunta_1(d)
        with tabs[2]:
            self.tab_pregunta_2(d)
        with tabs[3]:
            self.tab_pregunta_3(d)
        with tabs[4]:
            self.tab_pregunta_4(d, n_top)
        with tabs[5]:
            self.tab_pregunta_5(d)
        with tabs[6]:
            self.tab_pregunta_6(d)
