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
| **P1** | Tamaño habitable y capacidad ($m^2$, hab., baños) | El precio total escala fuertemente con el área ($r > 0.70$): viviendas pequeñas (<130 $m^2$) presentan una mediana de USD 319,950 vs. USD 850,000 en viviendas grandes (>270 $m^2$). Sin embargo, las viviendas pequeñas tienen el costo unitario más alto (**USD 3,118.48/$m^2$** vs. USD 2,431.31/$m^2$ en grandes), impulsado por la alta densidad y el valor del suelo en el casco urbano central de Seattle. |
| **P2** | Calificación constructiva (*grade*) | El estándar de construcción es el multiplicador más determinante del valor: el precio mediano asciende desde USD 270,000 en calificación Baja (grados 4–6) a USD 450,000 en Media (7–8), USD 715,000 en Alta (9) y **USD 1,038,000 en Lujo (10–13)**, representando una prima acumulada de **3.84×**. |
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

El repositorio incluye el archivo `src/AE3/datos/catalogo_barrios_nemotron.csv`, por lo que el pipeline y los notebooks corren **de forma autónoma y sin dependencias externas**.

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

## Referencias

- American Psychological Association. (2020). *Publication manual of the American Psychological Association* (7th ed.). APA.
- Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.
- harlfoxem. (2016). *House Sales in King County, USA* [Conjunto de datos]. Kaggle. <https://www.kaggle.com/datasets/harlfoxem/housesalesprediction>
- INRIA-SODA. (2023). *tabular-benchmark* [Conjunto de datos]. Hugging Face. <https://huggingface.co/datasets/inria-soda/tabular-benchmark>
- King County Department of Assessments. (2015). *Residential property appraisal reports and definitions*. King County, Washington.
- NVIDIA Corporation. (2024). *Nemotron Language Models and NVIDIA NIM Inference Microservices*. NVIDIA Developer. <https://build.nvidia.com/>
