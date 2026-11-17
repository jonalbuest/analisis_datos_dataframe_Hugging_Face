# Proyecto Integrador III – Analítica de Datos con Hugging Face

Repositorio del **Proyecto Integrador III – Analítica de Datos** (IU Digital) del **Grupo 1 Canvas**. Reúne las entregas incrementales **AE1**, **AE2**, **AE3** y **AE4**, articuladas sobre una arquitectura modular orientada a objetos en Python (`src/clases/`). Las clases gestionan la descarga de datasets desde Hugging Face, la depuración y auditoría de calidad, el enriquecimiento analítico y espacial con **NVIDIA Nemotron LLM** y la generación de reportes y tablas estadísticas. Un workflow de **GitHub Actions** ejecuta los notebooks en cada `push` y publica los artefactos. La **AE4** presenta los resultados en un dashboard de **Streamlit**, construido primero en un notebook y reutilizando las clases y los datos de las entregas anteriores.

- **Repositorio:** <https://github.com/jonalbuest/analisis_datos_dataframe_Hugging_Face>
- **Aplicación (AE4):** <https://housesforsale.streamlit.app/> (Streamlit Community Cloud) · <https://analisis-datos-dataframe-hugging-face.onrender.com/> (Render)

## Integrantes

- Jonathan Alvarez Bustamante
- Yusep Alexander Ruiz Agudelo
- Yonier Alexis Quiceno Rodriguez

**Docente:** Andres Felipe Callejas

---

## Resumen de las entregas

| Entrega | Objetivo | Dataset | Notebook / App | Artefactos y reportes |
|---|---|---|---|---|
| **AE1** – Carga y creación de dataset | Infraestructura base: clases de carga y reporte, notebook de transformación y CI/CD en GitHub Actions | [`cs-uche/car_dealership`](https://huggingface.co/datasets/cs-uche/car_dealership) (split `train`) | `src/AE1/sincronicas/transformaciones.ipynb` | `src/AE1/sincronicas/reportes/car_dealership.txt` |
| **AE2** – Ideación, diseño y documentación de la necesidad | Problema de negocio inmobiliario, ficha técnica, unidad de análisis y diccionario inicial (18 variables) | [`inria-soda/tabular-benchmark`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/viewer/reg_cat_house_sales), config. `reg_cat_house_sales` | `src/AE2/notebooks/transformaciones.ipynb` | `src/AE2/resultados/reg_cat_house_sales.txt`<br>`src/AE2/documento/EA2_Propuesta_HouseSales.docx` |
| **AE3** – Transformación y enriquecimiento de datos | Depuración con auditoría antes vs. después, variables derivadas (m², USD, distancias), enriquecimiento con **NVIDIA Nemotron** (43 variables) y 6 preguntas de negocio | Mismo dataset | `src/AE3/notebooks/transformaciones.ipynb` | `src/AE3/resultados/*.txt`<br>`src/AE3/documento/EA3_Transformacion_Enriquecimiento_HouseSales.docx`<br>Tablas P1–P6 en `src/AE3/datos/` |
| **AE4** – Métricas y visualización de datos | Métricas y gráficos por pregunta de negocio, dashboard en Streamlit con filtros y documento final en PDF | Dataset enriquecido de AE3 | `src/AE4/notebooks/visualizacion_ae4.ipynb`<br>`src/AE4/app.py` (Streamlit) | `src/AE4/dashboard_ae4.py`<br>`src/AE4/datos/`, `src/AE4/resultados/`<br>`src/AE4/documento/` (PDF final) |

> El dataset de AE1 sirvió para validar la infraestructura inicial. **Desde AE2, el proyecto trabaja con las ventas de viviendas del Condado de King (Seattle, WA).**

---

## Estructura del repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ejecutar.yml                  # CI/CD: ejecuta los notebooks de AE1 a AE4 y commitea resultados
├── .streamlit/
│   └── config.toml                       # Tema visual de la app de AE4
├── src/
│   ├── clases/                           # Paquete modular reutilizable (AE1–AE4)
│   │   ├── __init__.py
│   │   ├── hf_loader.py                  # HFDatasetLoader: descarga desde Hugging Face
│   │   ├── preprocesador.py              # PreprocesadorHouseSales: calidad, depuración y auditoría
│   │   ├── enriquecedor_nemotron.py      # EnriquecedorNemotron: variables derivadas, Haversine y NVIDIA Nemotron
│   │   └── reportes.py                   # Reportes: reportes .txt y tablas estadísticas por grupo
│   ├── AE1/
│   │   └── sincronicas/                  # datos/, reportes/, transformaciones.ipynb
│   ├── AE2/                              # datos/, documento/, notebooks/, resultados/
│   ├── AE3/
│   │   ├── datos/                        # crudo, enriquecido, catálogo Nemotron y tablas P1–P6
│   │   ├── documento/                    # EA3_Transformacion_Enriquecimiento_HouseSales.docx
│   │   ├── notebooks/transformaciones.ipynb
│   │   └── resultados/                   # auditoría y resumen en .txt
│   └── AE4/                              # Métricas y visualización
│       ├── notebooks/
│       │   └── visualizacion_ae4.ipynb   # Taller de gráficos: construye, verifica y prueba el dashboard
│       ├── app.py                        # Punto de entrada de Streamlit (Main file path)
│       ├── dashboard_ae4.py              # Clase DashboardAE4: filtros, KPI, figuras y respuestas de negocio
│       ├── requirements.txt              # Dependencias livianas de la app (Streamlit Cloud)
│       ├── datos/
│       │   ├── reg_cat_house_sales_enriquecido.csv   # Copia del dataset de AE3 para la app
│       │   ├── auditoria_global.csv                  # Antes vs. después (PreprocesadorHouseSales)
│       │   ├── auditoria_columnas.csv                # Detalle por columna
│       │   └── diccionario_variables.csv             # Variables creadas en AE3 (EnriquecedorNemotron)
│       ├── resultados/                   # Reportes .txt (clase Reportes)
│       └── documento/                    # Documento final del proyecto en PDF
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Paquete de clases en Python (`src/clases/`)

### 1. `HFDatasetLoader` (`hf_loader.py`)
Descarga conjuntos de datos de Hugging Face y los entrega como `pandas.DataFrame`, con opción de exportar a CSV o Parquet.

### 2. `PreprocesadorHouseSales` (`preprocesador.py`)
- `diagnosticar_calidad(...)`: evalúa duplicados, nulos y anomalías (habitaciones < 1 o > 15, baños < 1).
- `limpiar_y_validar(...)`: descarta las 92 filas con incoherencias físicas (0,43 %) y conserva una copia sin modificar.
- `comparar_antes_despues(...)`: tablas de auditoría global y por columna. **Reutilizada en AE4** para la sección Proyecto.

### 3. `EnriquecedorNemotron` (`enriquecedor_nemotron.py`)
- `calcular_variables_derivadas(...)`: precio en USD con `expm1` ($e^p - 1$, redondeado a dólares), áreas en m², USD/m², antigüedad, década y remodelación.
- `calcular_distancias_polos(...)`: distancias Haversine a Downtown Seattle, Bellevue Comercial y el Aeropuerto Sea-Tac.
- `enriquecer_con_nemotron(...)`: cuadrantes de $0.05^\circ$ (~4–5 km) enriquecidos con **NVIDIA NIM** (`nvidia/nemotron-3.5-lightning-30b-a3b`): municipio, barrio, entorno, transporte, comercio, parques, cuerpo de agua y tipo de vista al agua. Usa la caché `catalogo_barrios_nemotron.csv`.
- `obtener_diccionario_variables_nuevas()`: diccionario de las 25 variables creadas. **Reutilizado en AE4**.

### 4. `Reportes` (`reportes.py`)
- `generar_reporte()` / `guardar_reporte()`: reporte descriptivo en `.txt`. **Reutilizado en AE4** (`resumen_datos_dashboard_ae4.txt`).
- `tabla_estadistica_grupo()`: n, %, media, mediana, moda, desviación, CV, Q1, Q3, RIC, mínimo, máximo y mediana USD/m² por grupo. **Reutilizada en AE4**: el dashboard calcula sus tablas con esta misma función, por eso coinciden con las de AE3.

---

## AE3 – Transformación y enriquecimiento de datos

1. **Auditoría antes vs. después:** de 21.613 ventas se excluyeron 92 (0,43 %) por inconsistencias físicas (0 habitaciones, menos de 1 baño o 33 habitaciones en 150 m²). Se conserva el 99,57 % y la suma de ventas de USD 11.644.232.456.
2. **Diccionario ampliado:** de 18 a **43 variables**.
3. **NVIDIA Nemotron:** barrios, transporte, comercio, parques y tipo de frente de agua inferidos por cuadrante.

### Hallazgos de las 6 preguntas de negocio

| Pregunta | Dimensión | Hallazgo clave |
|---|---|---|
| **P1** | Tamaño y capacidad | Mediana de USD 320.000 en viviendas pequeñas (<130 m²) frente a USD 869.950 en muy grandes (≥280 m²). Las pequeñas tienen el mayor USD/m² (3.118,48 frente a 2.621,96). Correlación precio–área habitable: r = 0,70. |
| **P2** | Calificación constructiva | Mediana de USD 270.000 (Baja, 4–6), 375.000 (Estándar, 7), 562.000 (Media-Alta, 8–9) y **1.038.000 (Alta/Lujo, 10–13)**: 3,84×. |
| **P3** | Frente al agua | Solo 0,73 % ($n = 157$). Lago / río interior ($n = 81$): **USD 1.955.000**; mar en Puget Sound ($n = 76$): USD 908.250; sin vista: USD 450.000. Clasificación inferida con Nemotron. |
| **P4** | Subregión, barrio y entorno | Eastside tiene la mediana más alta (USD 640.000); Seattle el mayor USD/m² (3.470,95 frente a 1.774,88 en South King County). Barrios: Medina / Clyde Hill (USD 1.020.000), Downtown Bellevue (987.000), University District / Laurelhurst (805.000). |
| **P5** | Década y remodelación | Remodeladas (4,23 %, $n = 910$): mediana USD 600.500 frente a 449.500 (+33,6 %). En AE4: la remodelada tiene mayor precio típico en las 9 décadas con al menos 10 casas por grupo. |
| **P6** | Dinámica mensual | Más ventas en julio de 2014 (2.205) y abril de 2015 (2.224); menos en enero de 2015 (973). Mediana más alta en abril de 2015 (USD 478.250). Mayo de 2015 es parcial. |

---

## AE4 – Métricas y visualización de datos

### Flujo de trabajo

1. **Notebook** `src/AE4/notebooks/visualizacion_ae4.ipynb`: reutiliza los datos y las clases de AE1–AE3, exporta la copia de datos para la app, genera la auditoría y el diccionario, muestra cada tabla y cada gráfico por pregunta, verifica que las tablas coincidan con las de AE3 y prueba la app con `AppTest`.
2. **Clase** `src/AE4/dashboard_ae4.py` (`DashboardAE4`): contiene las métricas, las figuras y las respuestas. El notebook y la app usan la misma clase, así que muestran las mismas cifras.
3. **App** `src/AE4/app.py`: publica el dashboard en Streamlit.

### Qué se reutiliza de las entregas anteriores

| Origen | Elemento | Uso en AE4 |
|---|---|---|
| AE1 | `Reportes.guardar_reporte`, GitHub Actions | Reporte `.txt` de los datos del dashboard; el workflow ejecuta también el notebook de AE4 |
| AE2 | Necesidad y pregunta de negocio | Encabezado del dashboard y sección Proyecto |
| AE3 | Dataset enriquecido (21.521 × 43) | Datos del dashboard (`src/AE4/datos/`) |
| AE3 | `Reportes.tabla_estadistica_grupo` | Tablas estadísticas de cada pregunta (filtrables y descargables) |
| AE3 | `PreprocesadorHouseSales.comparar_antes_despues` | Auditoría antes vs. después en la sección Proyecto |
| AE3 | `EnriquecedorNemotron.obtener_diccionario_variables_nuevas` | Diccionario de variables en la sección Proyecto |

### Contenido del dashboard

- **Encabezado:** pregunta de negocio, guía de uso y **6 KPI** (viviendas, mediana del precio, mediana USD/m², % bien calificadas, % con vista al agua, % remodeladas). Con filtros, cada KPI muestra su diferencia frente al total.
- **Barra lateral:** tamaño, calificación, municipio o subregión y rango de fechas; en **Más filtros**: vista al agua, remodelación, anillo urbano, década de construcción, rango de precio y habitaciones. Incluye el tamaño del ranking de barrios, un botón para restablecer filtros y la fuente de los datos.
- **Navegación:** una barra de botones elige la sección. Solo se dibuja la sección elegida, lo que hace la app más rápida y evita que los mapas se dibujen vacíos (problema de las pestañas ocultas).
- **🎯 Respuestas de negocio:** para cada pregunta, la respuesta calculada, el indicador, cómo se ve y la decisión que apoya.
- **P1 Tamaño:** mediana por tamaño, habitaciones y baños; correlación (r de Pearson) de cada característica con el precio; nubes de puntos de área, calificación, baños y antigüedad, coloreables por anillo urbano o subregión; líneas de medianas; **casas atípicas** (regla de Tukey por número de baños: 736 casas, el 88 % a menos de 10 km de un centro comercial frente al 34 % del resto) con su mapa.
- **P2 Calificación:** mediana y conteo por nivel; mediana por grado (4–13); **mismo tamaño, distinta calificación** (en casas muy grandes, USD 550.000 con Estándar frente a 1.119.000 con Alta/Lujo); mapas de las viviendas bien calificadas y de las de baja calificación; reparto por subregión.
- **P3 Vista al agua:** cuántas casas tienen vista; mediana en USD y USD/m²; caja de precios; **prima en casas del mismo tamaño** (1,7 a 2,7 veces más); comparación por subregión; perfil de las casas con vista (más grandes y mejor calificadas); mapa.
- **P4 Ubicación:** USD/m² por subregión y anillo urbano; mapa de calor subregión × tamaño con medida seleccionable y su lectura explicada; mapa del precio por m²; **precio por m² según la distancia al centro comercial** (r = −0,58: USD 4.008/m² a menos de 5 km frente a 1.794 a 20 km o más); entorno (transporte, comercio, parques) frente a la mediana general; ranking de barrios.
- **P5 Remodelación:** torta de participación; mediana por década, remodelada vs. no remodelada; **misma subregión** (más cara en 4 de 5 subregiones); **años desde la remodelación** (máximo de USD 744.000 a los 5–9 años; a los 30 años o más, igual que una no remodelada); perfil de las remodeladas.
- **P6 Mensual:** ventas por mes, mediana mensual, variación frente al mes anterior, ventas por nivel de calificación, mediana mensual por subregión y **mapa de calor día de la semana × mes** (98 % de las ventas se registra de lunes a viernes).
- **📘 Proyecto:** necesidad, dataset, recorrido AE1–AE4, auditoría antes vs. después, diccionario, conclusiones y limitaciones.
- **🗂 Datos:** tabla filtrada y descarga en CSV.

Cada sección de pregunta incluye cómo leer los gráficos, una explicación calculada con los filtros actuales, la limitación y la tabla estadística de AE3 para la selección actual.

### Ejecutar en local

Desde la raíz del repositorio, con Python 3.12:

```bash
pip install -r requirements.txt
streamlit run src/AE4/app.py
```

Regenerar los datos de la app y revisar los gráficos:

```bash
jupyter nbconvert --to notebook --execute --inplace src/AE4/notebooks/visualizacion_ae4.ipynb
```

La última sección del notebook prueba la app con `streamlit.testing` (AppTest), sin filtros y con filtros, antes de publicar.

### Publicación

La app está publicada en [Streamlit Community Cloud](https://housesforsale.streamlit.app/) y en [Render](https://analisis-datos-dataframe-hugging-face.onrender.com/); ambas se actualizan con cada `push` a `main`.

- Repositorio: este repositorio · Rama: `main`
- Main file path: `src/AE4/app.py`
- Python: 3.12
- Dependencias: `requirements.txt` de la raíz (incluye `streamlit>=1.50` y `plotly>=5.24`).
- Los mapas usan el fondo público `carto-positron` (no requiere clave; sí conexión a internet).

---

## Ejecución local de los notebooks

### 1. Preparación del entorno

```bash
git clone https://github.com/jonalbuest/analisis_datos_dataframe_Hugging_Face.git
cd analisis_datos_dataframe_Hugging_Face

python -m venv .venv

# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# En Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configuración opcional de la API Key de NVIDIA

El repositorio incluye `src/AE3/datos/catalogo_barrios_nemotron.csv`, así que los notebooks **no llaman a la API de NVIDIA** para ejecutarse. La descarga inicial del dataset desde Hugging Face sí requiere internet.

Para regenerar el catálogo consultando **NVIDIA Nemotron**, crea un archivo `.env` en la raíz (no se sube al repositorio):

```env
NVIDIA_API_KEY=nvapi-tu_api_key_aqui
```

### 3. Ejecución de los notebooks

```bash
# AE1:
cd src/AE1/sincronicas
jupyter nbconvert --to notebook --execute --inplace transformaciones.ipynb

# AE2:
cd ../../AE2/notebooks
jupyter nbconvert --to notebook --execute --inplace transformaciones.ipynb

# AE3:
cd ../../AE3/notebooks
jupyter nbconvert --to notebook --execute --inplace transformaciones.ipynb

# AE4:
cd ../../AE4/notebooks
jupyter nbconvert --to notebook --execute --inplace visualizacion_ae4.ipynb
```

---

## Automatización con GitHub Actions

El archivo `.github/workflows/ejecutar.yml` se ejecuta en cada `push` a `main`:
1. Configura un runner Ubuntu con Python 3.12 e instala `requirements.txt`.
2. Limpia los datos temporales de AE1–AE3, preservando el catálogo espacial.
3. Ejecuta los notebooks de AE1, AE2 y AE3.
4. Ejecuta el notebook de AE4: actualiza la copia de datos de la app, la auditoría y el diccionario, y **prueba la app** con `AppTest`. Si la app falla, el workflow falla.
5. Hace commit y push de los resultados con `github-actions[bot]`.

---

## Preguntas de negocio y tablas de resultados (AE3)

Rutas relativas a la raíz del repositorio.

| Pregunta | Tablas | Nivel de detalle |
|---|---|---|
| P1. Tamaño y capacidad | `src/AE3/datos/tabla_p1_tamano_vivienda.csv`, `tabla_p1b_habitaciones.csv` | Rango de área habitable (m²), habitaciones |
| P2. Calificación constructiva | `tabla_p2_calificacion_constructiva.csv`, `tabla_p2b_calificacion_detalle.csv` | Nivel y grado de calificación |
| P3. Frente al agua | `tabla_p3_vista_mar_lago.csv` | Mar (Puget Sound), lago interior, sin vista |
| P4. Subregión, barrio y entorno | `tabla_p4_subregiones_barrios.csv`, `tabla_p4b_barrios_top15.csv`, `tabla_p4c_transporte_comercio_parques.csv`, `tabla_p4d_zonas_comunes_parques.csv` | Municipio, barrio (n ≥ 50), transporte, comercio, parques (**inferidos con Nemotron**) |
| P5. Remodelación y década | `tabla_p5_antiguedad_remodelacion.csv`, `tabla_p5b_resumen_remodelacion.csv` | Década de construcción y condición de remodelación |
| P6. Periodo de venta | `tabla_p6_periodo_ventas.csv` | Mes (2014-05 a 2015-05) |

**Otros archivos:**

- Dataset preparado y enriquecido (43 variables): `src/AE3/datos/reg_cat_house_sales_enriquecido.csv`
- Copia sin modificar: `src/AE3/datos/reg_cat_house_sales_raw.csv`
- Catálogo espacial (cuadrantes de 0,05°): `src/AE3/datos/catalogo_barrios_nemotron.csv`
- Auditoría antes vs. después: `src/AE3/resultados/auditoria_antes_despues.txt`
- Resumen descriptivo: `src/AE3/resultados/resumen_reg_cat_house_sales_enriquecido.txt`

---

## Datos y permisos de publicación

- **Fuente:** `inria-soda/tabular-benchmark`, configuración `reg_cat_house_sales`, split `train`. Conjunto público de Hugging Face, con origen en el King County Department of Assessments.
- **Licencia:** la ficha de Hugging Face no declara licencia. Kaggle y OpenML indican CC0 (dominio público). Ver la ficha en `src/AE2/documento/EA2_Propuesta_HouseSales.docx`.
- **Datos personales:** no contiene. Incluye coordenadas de cada vivienda vendida, sin nombres ni direcciones.
- **Reproducción:** el repositorio publica el CSV preparado y los resultados. Para reproducir desde cero, ejecutar los notebooks de AE3 y AE4.

---

## Limitaciones

- **Análisis descriptivo:** las diferencias entre grupos no son efectos causales. Tamaño, calificación y ubicación están relacionados entre sí.
- **Atributos inferidos con NVIDIA Nemotron:** barrio, municipio, transporte, parques, comercio, zonas comunes, cuerpo de agua y tipo de vista se asignan por cuadrante de ~4–5 km, sin fuente verificada. Son orientativos; hay casos dudosos (por ejemplo, viviendas marcadas como "mar" cuyo cuerpo de agua cercano es un lago). Validación pendiente contra una fuente pública (King County GIS).
- **Frente al agua:** submuestra pequeña (n = 81 lago y n = 76 mar).
- **Mayo de 2015 es parcial:** datos hasta el 27 de mayo (642 ventas).
- **Periodo y cobertura:** ventas de mayo de 2014 a mayo de 2015; no representan el mercado actual ni otra región.
- **Registros excluidos:** 92 viviendas (0,43 %) por consistencia física. Ver la auditoría.

---

## Referencias

- American Psychological Association. (2020). *Publication manual of the American Psychological Association* (7th ed.). APA.
- Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.
- harlfoxem. (2016). *House Sales in King County, USA* [Conjunto de datos]. Kaggle. <https://www.kaggle.com/datasets/harlfoxem/housesalesprediction>
- INRIA-SODA. (2023). *tabular-benchmark* [Conjunto de datos]. Hugging Face. <https://huggingface.co/datasets/inria-soda/tabular-benchmark>
- King County Department of Assessments. (2024). *Residential property appraisal reports and definitions*. King County, Washington.
- NVIDIA Corporation. (2024). *Nemotron language models and NVIDIA NIM inference microservices*. NVIDIA Developer. <https://build.nvidia.com/>
- Plotly Technologies Inc. (2025). *Plotly Python graphing library*. <https://plotly.com/python/>
- Streamlit Inc. (2025). *Streamlit documentation*. <https://docs.streamlit.io/>
