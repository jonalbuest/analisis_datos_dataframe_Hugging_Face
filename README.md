# Proyecto Integrador III – Analítica de Datos con Hugging Face

Repositorio del **Proyecto Integrador III – Analítica de Datos** (IU Digital) del **Grupo 1 Canvas**. Reúne las entregas incrementales **AE1**, **AE2** y **AE3**, articuladas sobre una arquitectura modular orientada a objetos en Python (`src/clases/`). Dichas clases gestionan la descarga de datasets desde Hugging Face, la depuración y auditoría de calidad, el enriquecimiento analítico y espacial con **NVIDIA Nemotron LLM**, y la generación de reportes y tablas estadísticas en formato APA. Un workflow continuo de **GitHub Actions** valida y ejecuta los notebooks en cada `push` publicando los artefactos analíticos.

## Integrantes

- Jonathan Alvarez Bustamante
- Yusep Alexander Ruiz Agudelo
- Yonier Alexis Quiceno Rodriguez

**Docente:** Andres Felipe Callejas

---

## Resumen de las entregas

| Entrega | Objetivo | Dataset | Notebook | Artefactos y Reportes |
|---|---|---|---|---|
| **AE1** – Carga y creación de dataset | Montar la infraestructura base: clases de carga y reporte, notebook de transformación y CI/CD en GitHub Actions | [`cs-uche/car_dealership`](https://huggingface.co/datasets/cs-uche/car_dealership) (split `train`) | `src/AE1/sincronicas/transformaciones.ipynb` | `src/AE1/sincronicas/reportes/car_dealership.txt` |
| **AE2** – Ideación, diseño y documentación de la necesidad | Definir el problema de negocio inmobiliario, documentar la ficha técnica, unidad de análisis y diccionario inicial (18 vars) | [`inria-soda/tabular-benchmark`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/viewer/reg_cat_house_sales), config. `reg_cat_house_sales` | `src/AE2/notebooks/transformaciones.ipynb` | `src/AE2/resultados/reg_cat_house_sales.txt`<br>`src/AE2/documento/EA2_Propuesta_HouseSales.docx` |
| **AE3** – Transformación y enriquecimiento de datos | Depuración con auditoría Antes vs. Después, variables derivadas ($m^2$, USD, distancias), enriquecimiento espacial con **NVIDIA Nemotron LLM** (43 vars) y resolución de 6 preguntas de negocio | [`inria-soda/tabular-benchmark`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/viewer/reg_cat_house_sales), config. `reg_cat_house_sales` | `src/AE3/notebooks/transformaciones.ipynb` | `src/AE3/resultados/auditoria_antes_despues.txt`<br>`src/AE3/resultados/resumen_reg_cat_house_sales_enriquecido.txt`<br>`src/AE3/documento/EA3_Transformacion_Enriquecimiento_HouseSales.docx`<br>Tablas P1–P6 en `src/AE3/datos/` |

> El dataset de AE1 sirvió para validar la infraestructura inicial. **A partir de AE2 y a lo largo de AE3, el proyecto trabaja de lleno con las transacciones residenciales del Condado de King (Seattle, WA).**

---

## Estructura del repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ejecutar.yml                  # CI/CD: Ejecuta los notebooks de AE1, AE2 y AE3 y commitea resultados
├── src/
│   ├── clases/                           # Paquete modular reutilizable
│   │   ├── __init__.py                   # Exporta HFDatasetLoader, PreprocesadorHouseSales, EnriquecedorNemotron, Reportes
│   │   ├── hf_loader.py                  # Descarga desde Hugging Face → pandas DataFrame / CSV
│   │   ├── preprocesador.py              # PreprocesadorHouseSales: Reglas de calidad, depuración y auditoría
│   │   ├── enriquecedor_nemotron.py      # EnriquecedorNemotron: Transformaciones físicas, geodésicas y LLM NVIDIA
│   │   └── reportes.py                   # Reportes: Resúmenes descriptivos y tablas estadísticas grupales APA
│   ├── AE1/
│   │   └── sincronicas/
│   │       ├── datos/
│   │       ├── reportes/
│   │       └── transformaciones.ipynb
│   ├── AE2/
│   │   ├── datos/
│   │   ├── documento/
│   │   ├── notebooks/
│   │   └── resultados/
│   └── AE3/                              # Actividad Evaluativa 3
│       ├── datos/                        # Datasets (raw, enriquecido), catálogo espacial y tablas P1-P6
│       │   ├── reg_cat_house_sales_raw.csv
│       │   ├── reg_cat_house_sales_enriquecido.csv
│       │   ├── catalogo_barrios_nemotron.csv
│       │   └── tabla_p*.csv              # Tablas de resultados estadísticos P1 a P6
│       ├── documento/                    # Documento formal consolidado de avance
│       │   └── EA3_Transformacion_Enriquecimiento_HouseSales.docx
│       ├── notebooks/
│       │   └── transformaciones.ipynb    # Notebook ejecutable integral con explicaciones markdown
│       └── resultados/                   # Auditorías y reportes automatizados en .txt
│           ├── auditoria_antes_despues.txt
│           └── resumen_reg_cat_house_sales_enriquecido.txt
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Paquete de clases en Python (`src/clases/`)

### 1. `HFDatasetLoader` (`hf_loader.py`)
Descarga conjuntos de datos desde el ecosistema Hugging Face Datasets y los entrega como `pandas.DataFrame`. Permite configuración de split, tokens privados y exportación directa a CSV/Parquet.

### 2. `PreprocesadorHouseSales` (`preprocesador.py`)
Asegura la gobernanza y calidad de los datos antes de cualquier transformación:
- `verificar_calidad(df)`: Evalúa duplicados, campos nulos, valores negativos y anomalías estructurales (habitaciones < 1 o > 15, baños < 1).
- `depurar(df, conservar_original=True)`: Aplica reglas determinísticas que descartan 92 filas con incoherencias físicas (0.43% de la muestra), garantizando la reproducibilidad y preservando una copia raw inmutable.
- `renombrar_columnas(df)`: Homologa la nomenclatura técnica hacia español estándar.
- `tipar_fechas(df)`: Decodifica `date_year` (0/1 a 2014/2015) y genera fechas aproximadas de venta.
- `comparar_antes_despues(df_antes, df_despues)`: Construye la matriz de auditoría técnica con recuentos, porcentajes de retención y control de nulos.

### 3. `EnriquecedorNemotron` (`enriquecedor_nemotron.py`)
Encapsula el enriquecimiento matemático, geodésico y de inteligencia artificial generativa:
- `calcular_variables_derivadas(df)`: Revierte la escala logarítmica del precio ($e^p - 1$) preservando centavos, convierte pies cuadrados a metros cuadrados ($1\text{ m}^2 \approx 10.7639\text{ sqft}$), calcula el precio unitario ($\text{USD}/\text{m}^2$), y genera la antigüedad de la propiedad, década de construcción y condición de remodelación.
- `calcular_distancias_geodesicas(df)`: Calcula distancias en kilómetros mediante la fórmula de Haversine a nodos estratégicos: Downtown Seattle, Downtown Bellevue y el Aeropuerto Internacional Sea-Tac.
- `construir_catalogo_espacial(df)`: Discretiza las coordenadas en cuadrantes de $0.05^\circ \times 0.05^\circ$ (~4 a 5 km) anclados a centros de gravedad urbanos del Condado de King (`ANCLAS_KING_COUNTY`).
- `enriquecer_con_nemotron(df)`: Consulta la API de **NVIDIA NIM** (`nvidia/nemotron-3.5-lightning-30b-a3b`) para enriquecer cada cuadrante con: municipio oficial, nombre de barrio/sector, entorno urbano, acceso a transporte, nivel de comercio, zonas comunes y dotación de parques. Gestiona caché local (`catalogo_barrios_nemotron.csv`) para ejecuciones ultrarrápidas y reproduce una unión `many_to_one` validada que no altera filas ni totales monetarios.
- `clasificar_tipo_frente_agua(df)`: Diferencia propiedades con frente a lago interior (Lake Washington, Lake Sammamish) frente a costa marina de agua salada (Puget Sound).

### 4. `Reportes` (`reportes.py`)
Generador estandarizado de reportes analíticos y tablas estadísticas:
- `generar_reporte()` / `guardar_reporte()`: Genera archivos de texto enriquecidos con marcas de tiempo ISO, resumen numérico y resumen categórico (cardinalidad y modas).
- `tabla_estadistica_grupo()`: Calcula para cualquier variable de segmentación la distribución estadística completa: $n$, porcentaje, media, mediana, moda, desviación estándar, coeficiente de variación ($CV$), primer cuartil ($Q_1$), tercer cuartil ($Q_3$), rango intercuartílico ($IQR$), valores mínimos y máximos, además de la mediana y dispersión del precio unitario ($\text{USD}/\text{m}^2$).

---

## AE3 – Transformación y Enriquecimiento de Datos

### Resumen del Sprint
En esta etapa se completó la limpieza profunda, enriquecimiento multidimensional y análisis descriptivo de las 21,521 transacciones inmobiliarias válidas del Condado de King:
1. **Auditoría Antes vs. Después:** De 21,613 transacciones iniciales, se eliminaron 92 registros (0.43%) debido a inconsistencias físicas (viviendas con 0 baños, 0 habitaciones o anomalías extremas como 33 habitaciones en 150 m²). La depuración preservó intacto el 99.57% del volumen y la suma exacta de transacciones en USD 11,644,232,456.
2. **Ampliación del Diccionario:** El conjunto pasó de 18 columnas iniciales a **43 variables finales** estructuradas en dimensiones físicas, financieras, geodésicas y de entorno contextual.
3. **Integración de NVIDIA Nemotron LLM:** Mediante agregación espacial y anclaje geodésico, el modelo enriqueció el dataset con nombres de barrios representativos (Medina, Bellevue, Mercer Island, Laurelhurst, Kirkland, Kent, Renton), infraestructura de transporte, áreas comerciales, parques y tipología de frentes de agua.

### Hallazgos de las 6 Preguntas Analíticas de Negocio

| Pregunta | Dimensión Analizada | Hallazgo Clave |
|---|---|---|
| **P1** | Tamaño habitable y capacidad ($m^2$, hab., baños) | El precio total aumenta con el área: viviendas pequeñas (<130 $m^2$) presentan una mediana de USD 320,000 vs. USD 869,950 en viviendas muy grandes (≥280 $m^2$). Sin embargo, las viviendas pequeñas tienen el costo unitario más alto (**USD 3,118.48/$m^2$** vs. USD 2,621.96/$m^2$ en las muy grandes), impulsado por la alta densidad y el valor del suelo en el casco urbano central de Seattle. |
| **P2** | Calificación constructiva (*grade*) | El estándar de construcción es el multiplicador más determinante del valor: el precio mediano asciende desde USD 270,000 en calificación Baja (grados 4–6) a USD 375,000 en Estándar (grado 7), USD 562,000 en Media-Alta (grados 8–9) y **USD 1,038,000 en Alta/Lujo (grados 10–13)**, representando una prima acumulada de **3.84×**. |
| **P3** | Tipo de frente acuático (lago vs. mar) | Las viviendas frente al agua representan solo el 0.73% del mercado ($n=157$), pero muestran una valoración extrema: las propiedades sobre **lagos interiores** (Lake Washington y Lake Sammamish, $n=81$) alcanzan una mediana récord de **USD 1,955,000** (**USD 6,575.26/$m^2$**), superando en más del doble a las ubicadas sobre agua salada en Puget Sound (USD 908,250 median) y cuadruplicando a las propiedades sin frente acuático (USD 450,000). |
| **P4** | Subregiones, barrios y entorno (Nemotron) | La subregión **Eastside** ostenta el precio mediano más elevado (USD 640,000), pero la ciudad de **Seattle** lidera en costo por metro cuadrado (**USD 3,470.95/$m^2$** vs. USD 1,774.88/$m^2$ en South King County). Los enclaves con mayor plusvalía son **Medina** (mediana USD 1,020,000), **Downtown Bellevue** (USD 987,000) y **U-District / Laurelhurst** (USD 805,000). |
| **P5** | Antigüedad y remodelación | Las viviendas remodeladas (4.23% del mercado, $n=910$) obtienen un precio mediano de **USD 600,500** frente a USD 449,500 en las no remodeladas, reflejando una **prima bruta de modernización de +33.6% (+USD 151,000)**. |
| **P6** | Estacionalidad mensual de transacciones | El mercado sigue un ciclo estacional marcado: los volúmenes alcanzan sus máximos en **julio de 2014** (2,205 ventas) y **abril de 2015** (2,224 ventas), mientras que el valle se presenta en **enero de 2015** (973 ventas). El pico máximo de precios se registró en abril de 2015 con una mediana de USD 478,250. |

---

## Ejecución local

### 1. Preparación del entorno

```bash
git clone <URL-del-repositorio>
cd analisis_datos_dataframe_Hugging_Face

python -m venv .venv

# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# En Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configuración opcional de API Key de NVIDIA

El repositorio incluye el archivo `src/AE3/datos/catalogo_barrios_nemotron.csv`, así que el notebook de AE3 **no llama a la API de NVIDIA** para ejecutarse. La descarga inicial del dataset desde Hugging Face sí requiere conexión a internet.

Si deseas regenerar el catálogo espacial consultando directamente el modelo **NVIDIA Nemotron** (`nvidia/nemotron-3.5-lightning-30b-a3b`), crea un archivo `.env` en la raíz del proyecto:

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
```

---

## Automatización con GitHub Actions

El archivo `.github/workflows/ejecutar.yml` orquesta la integración continua en cada `push` a la rama `main`:
1. Configura un runner Ubuntu con Python 3.12 y descarga las dependencias del proyecto.
2. Limpia los datos temporales preservando el catálogo espacial local.
3. Ejecuta secuencialmente los tres notebooks (`AE1`, `AE2` y `AE3`) con `jupyter nbconvert`.
4. Realiza commit y push automático de los datasets generados, tablas de resumen y reportes `.txt` utilizando el bot oficial `github-actions[bot]`.

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

**Otros archivos de salida:**

- Dataset preparado y enriquecido (43 variables): `src/AE3/datos/reg_cat_house_sales_enriquecido.csv`
- Copia sin modificar para comparar: `src/AE3/datos/reg_cat_house_sales_raw.csv`
- Catálogo espacial (cuadrantes de 0,05°): `src/AE3/datos/catalogo_barrios_nemotron.csv`
- Auditoría antes vs. después: `src/AE3/resultados/auditoria_antes_despues.txt`
- Resumen descriptivo del DataFrame final (generado por el notebook): `src/AE3/resultados/resumen_reg_cat_house_sales_enriquecido.txt`

---

## Datos y permisos de publicación

- **Fuente:** `inria-soda/tabular-benchmark`, configuración `reg_cat_house_sales`, split `train`. Es un conjunto público de Hugging Face.
- **Licencia:** la ficha de Hugging Face no declara licencia. Kaggle y OpenML indican CC0 (dominio público). Ver la ficha en `src/AE2/documento/EA2_Propuesta_HouseSales.docx`.
- **Datos personales:** no contiene. Incluye coordenadas geográficas de cada vivienda vendida, sin nombres ni direcciones.
- **Reproducción:** el repositorio publica el CSV preparado y los resultados. Para reproducir desde cero, ejecutar el notebook de AE3 (la descarga usa HFDatasetLoader y no requiere token). Si el docente o la plataforma restringen la publicación del CSV, el procedimiento es el mismo: descargar el dataset original desde Hugging Face y ejecutar el notebook.

---

## Limitaciones de AE3

- **Análisis descriptivo:** las diferencias entre grupos no son efectos causales. Tamaño, calificación y ubicación están relacionados entre sí.
- **Atributos inferidos con NVIDIA Nemotron:** barrio, municipio, transporte, parques, comercio, zonas comunes y cuerpo de agua se asignan por cuadrante de ~4–5 km, sin una fuente verificada. Tomarlos como orientativos. Validación pendiente contra una fuente pública.
- **Frente al agua:** la submuestra es pequeña (n = 81 lago interior y n = 76 mar). La clasificación depende de la misma inferencia.
- **Mayo de 2015 es parcial:** los datos llegan hasta el 27 de mayo (642 ventas). No comparar ese mes con los completos sin ajustar.
- **Periodo y cobertura:** ventas de mayo de 2014 a mayo de 2015. No representan el mercado actual ni otra región.
- **Registros excluidos:** 92 viviendas (0,43 %) por consistencia física. Ver la auditoría.

---

## Referencias

- American Psychological Association. (2020). *Publication manual of the American Psychological Association* (7th ed.). APA.
- Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.
- harlfoxem. (2016). *House Sales in King County, USA* [Conjunto de datos]. Kaggle. <https://www.kaggle.com/datasets/harlfoxem/housesalesprediction>
- INRIA-SODA. (2023). *tabular-benchmark* [Conjunto de datos]. Hugging Face. <https://huggingface.co/datasets/inria-soda/tabular-benchmark>
- King County Department of Assessments. (2015). *Residential property appraisal reports and definitions*. King County, Washington.
- NVIDIA Corporation. (2024). *Nemotron Language Models and NVIDIA NIM Inference Microservices*. NVIDIA Developer. <https://build.nvidia.com/>
