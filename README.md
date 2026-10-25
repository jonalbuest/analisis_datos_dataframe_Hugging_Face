# Proyecto Integrador III – Analítica de Datos con Hugging Face

Repositorio del **Proyecto Integrador III – Analítica de Datos** (IU Digital) del **Grupo 1 Canvas**. Reúne las entregas incrementales **AE1** y **AE2**, que comparten un paquete de clases en Python (`src/clases/`). Esas clases descargan datasets de Hugging Face, los convierten en DataFrames de pandas y generan automáticamente un resumen en `.txt`. Un workflow de **GitHub Actions** ejecuta los notebooks en cada `push` y publica los resultados.

## Integrantes

- Jonathan Alvarez Bustamante
- Yusep Alexander Ruiz Agudelo
- Yonier Alexis Quiceno Rodriguez

**Docente:** Andres Felipe Callejas

---

## Resumen de las entregas

| Entrega | Objetivo | Dataset | Notebook | Resultado `.txt` |
|---|---|---|---|---|
| **AE1** – Carga y creación de dataset | Montar la infraestructura: dos clases (carga y resumen), un notebook con transformaciones y automatización en GitHub | [`cs-uche/car_dealership`](https://huggingface.co/datasets/cs-uche/car_dealership) (split `train`) | `src/AE1/sincronicas/transformaciones.ipynb` | `src/AE1/sincronicas/reportes/car_dealership.txt` |
| **AE2** – Ideación, diseño y documentación de la necesidad | Definir el dataset definitivo, documentar la necesidad analítica y describir los datos | [`inria-soda/tabular-benchmark`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/viewer/reg_cat_house_sales), config. `reg_cat_house_sales` (split `train`) | `src/AE2/notebooks/transformaciones.ipynb` | `src/AE2/resultados/reg_cat_house_sales.txt` |

> El dataset de AE1 se usó solo para construir y probar la infraestructura. **A partir de AE2, el proyecto trabaja con el dataset de viviendas del Condado de King.**

---

## Estructura del repositorio

```text
.
├── .github/
│   └── workflows/
│       └── ejecutar.yml                  # Ejecuta los notebooks de AE1 y AE2 y hace commit de los resultados
├── src/
│   ├── clases/                           # Clases compartidas por todas las entregas
│   │   ├── __init__.py
│   │   ├── hf_loader.py                  # HFDatasetLoader: Hugging Face → DataFrame (y CSV opcional)
│   │   └── reportes.py                   # Reportes: DataFrame → resumen en pantalla y en .txt
│   ├── AE1/
│   │   └── sincronicas/
│   │       ├── datos/
│   │       ├── reportes/
│   │       └── transformaciones.ipynb
│   └── AE2/                             # Actividad 2
│       ├── datos/                       # CSV descargado automáticamente (reg_cat_house_sales.csv)
│       ├── documento/                   # Documento inicial del proyecto (.docx)
│       ├── notebooks/
│       │   └── transformaciones.ipynb    # Reconocimiento, transformaciones y reporte (house_sales)
│       ├── datos/                        # CSV descargado (lo genera el notebook)
│       └── resultados/                   # Resumen .txt (lo genera el notebook)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Clases en Python (`src/clases/`)

### `HFDatasetLoader` (`hf_loader.py`)

Descarga un dataset de Hugging Face y lo devuelve como `pandas.DataFrame`.

| Parámetro | Descripción |
|---|---|
| `dataset_name` | Identificador del repositorio en Hugging Face (obligatorio) |
| `config` | Configuración o subconjunto, si el dataset tiene varias |
| `split` | División a cargar (por defecto, `train`) |
| `token` | Token de Hugging Face, solo para datasets privados |
| `max_rows` | Límite de filas, para trabajar con una muestra |
| `filename` / `file_type` | Guarda el DataFrame en `csv`, `json`, `parquet` o `xlsx` |

Métodos: `load()` descarga y devuelve el DataFrame; `info()` devuelve el dataset, la configuración, el split y el número de filas y columnas.

### `Reportes` (`reportes.py`)

Recibe un DataFrame (no se conecta a Hugging Face) y genera un resumen con:

- identificador del dataset, configuración, split y cobertura o muestra (opcionales);
- cantidad de filas, columnas y duplicados;
- nombres y tipos de las columnas;
- valores nulos por columna;
- estadísticas descriptivas de las columnas numéricas (`describe()`).

Métodos: `generar_reporte()` devuelve el texto, `imprimir_reporte()` lo muestra y `guardar_reporte()` lo escribe en UTF-8, creando la carpeta si no existe.

### Ejemplo de uso

```python
from clases import HFDatasetLoader, Reportes

loader = HFDatasetLoader(
    dataset_name="inria-soda/tabular-benchmark",
    config="reg_cat_house_sales",
    split="train",
    filename="../datos/reg_cat_house_sales.csv",
)
df = loader.load()

Reportes().guardar_reporte(
    df,
    "../resultados/reg_cat_house_sales.txt",
    dataset_name=loader.dataset_name,
    config=loader.config,
    split=loader.split,
)
```

---

## AE1 – Carga y creación de dataset

- **Dataset:** `cs-uche/car_dealership`, split `train`. Son 15.389 registros y 11 columnas de vehículos usados: año, impuesto, kilometraje, cilindraje, fabricante, modelo, combustible, interior de cuero, tipo de caja, categoría y precio.
- **Flujo del notebook:** importar las clases → cargar el DataFrame → revisar su estructura (`info()`) → transformar → generar y guardar el resumen `.txt`.
- **Transformaciones:**
  1. Eliminación de filas duplicadas: de 15.389 a 12.810 filas (se eliminan 2.579).
  2. Renombrado de las 11 columnas al español.
- **Resultado:** `src/AE1/sincronicas/reportes/car_dealership.txt`, que resume el DataFrame transformado.

---

## AE2 – Ideación, diseño y documentación de la necesidad

### Necesidad analítica

- **Escenario (hipotético):** una compañía de consultoría y avalúos inmobiliarios quiere estandarizar sus criterios de valoración con datos, en lugar de depender solo de la experiencia de cada perito.
- **Beneficiarios:** los peritos avaluadores y el coordinador de avalúos (principales) y los asesores comerciales (secundarios).
- **Pregunta de análisis:**

  > ¿Cómo varía el precio de venta de las viviendas vendidas en el Condado de King entre mayo de 2014 y mayo de 2015 según su tamaño (área habitable, área del lote, habitaciones y baños), su calificación de construcción, su ubicación frente al agua y su antigüedad o remodelación?

- **Tipo de análisis:** descriptivo y comparativo. No predice el precio de una vivienda concreta ni establece relaciones de causa y efecto.

### Ficha del dataset

| Campo | Valor |
|---|---|
| Repositorio | `inria-soda/tabular-benchmark` |
| Configuración | `reg_cat_house_sales` (archivo `reg_cat/house_sales.csv`) |
| Split | `train` (único disponible) |
| Revisión consultada | commit `8d0ff91` (4 de septiembre de 2023) |
| Fecha de consulta | 22 de octubre de 2026 |
| Tamaño | 21.613 filas × 18 columnas (≈ 2 MB) |
| Muestreo | No: se usa el dataset completo |
| Fuente original | *House Sales in King County, USA* (Kaggle, harlfoxem, 2016), curado en OpenML (ID 44066) por Grinsztajn et al. (2022) |
| Licencia | No indicada en la ficha de Hugging Face; **CC0 (dominio público)** en Kaggle y OpenML |
| Datos sintéticos | No (datos reales de ventas, según OpenML y Kaggle) |
| Datos personales | No contiene |

**Unidad de análisis:** cada fila representa **una vivienda vendida** en el Condado de King (Seattle, Washington, EE. UU.) entre mayo de 2014 y mayo de 2015.

### Notas para interpretar los datos

- `price` está en **escala logarítmica**: `price = ln(precio_USD + 1)`.
- `date_year` está codificado: `0` = 2014 y `1` = 2015.
- `waterfront`: `1` = frente al agua, `0` = no.
- `yr_renovated = 0` significa que la vivienda nunca fue remodelada.
- Las áreas están en **pies cuadrados** (1 m² ≈ 10,764 sqft).
- La versión curada **no incluye** el estado de conservación, la vista, el número de pisos ni el código postal del dataset original.

### Flujo del notebook

1. Importar `HFDatasetLoader` y cargar el DataFrame desde Hugging Face.
2. Mostrar registros (`head()`), dimensiones, columnas, tipos y valores nulos, y explicar qué representa cada fila y qué variables respaldan la necesidad.
3. Detectar registros inconsistentes: viviendas con 0 habitaciones, con menos de 1 baño o con 33 habitaciones.
4. Aplicar las transformaciones.
5. Generar, guardar y verificar el resumen `.txt` con el identificador del dataset, la configuración, el split y la cobertura.

### Transformaciones aplicadas

| # | Transformación | Resultado |
|---|---|---|
| 1 | Eliminación de duplicados | 0 duplicados (21.613 filas) |
| 2 | Filtro de registros inconsistentes (`bedrooms` entre 1 y 15 y `bathrooms` ≥ 1) | 21.521 filas (se descartan 92) |
| 3 | Renombrado de columnas al español | 18 columnas en español |
| 4 | Decodificación (`Vista al agua` → Sí/No; `Año de venta` → 2014/2015) y cálculo de `Precio (USD)` | 19 columnas |

### Documento

El documento inicial del proyecto (portada, introducción, necesidad, justificación, objetivos, marco teórico y legal, metodología con ficha y diccionario de variables, reconocimiento del dataset y referencias) está en `src/AE2/documento/` y se adjunta también en Canvas.

---

## Automatización con GitHub Actions

El workflow `.github/workflows/ejecutar.yml` se ejecuta en cada `push` a `main` y:

1. prepara un servidor Ubuntu con Python 3.12;
2. instala las dependencias de `requirements.txt` (además de `pyarrow` y `openpyxl`);
3. borra los datos y reportes anteriores de AE1 y AE2;
4. ejecuta los dos notebooks con `jupyter nbconvert --execute`;
5. hace commit y push de los resultados con el usuario **`github-actions[bot]`**.

> Para comprobar que la automatización funciona, el último commit del historial debe ser de `github-actions[bot]`.

---

## Ejecución local

```bash
git clone <URL-del-repositorio>
cd analisis_datos_dataframe_Hugging_Face

python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install pyarrow openpyxl   # opcional: exportar a parquet o xlsx
```

Para ejecutar los notebooks, ábrelos en Jupyter o VS Code y ejecuta todas las celdas en orden. También se pueden ejecutar desde la consola:

```bash
# AE1
cd src/AE1/sincronicas
jupyter nbconvert --to notebook --execute --inplace transformaciones.ipynb

# AE2 (desde la raíz del repositorio)
cd src/AE2/notebooks
jupyter nbconvert --to notebook --execute --inplace transformaciones.ipynb
```

Archivos que se generan:

| Entrega | Datos | Resumen |
|---|---|---|
| AE1 | `src/AE1/sincronicas/datos/car_dealership.csv` | `src/AE1/sincronicas/reportes/car_dealership.txt` |
| AE2 | `src/AE2/datos/reg_cat_house_sales.csv` | `src/AE2/resultados/reg_cat_house_sales.txt` |

### Token de Hugging Face (opcional)

Los dos datasets son públicos y no requieren token. Para evitar el aviso de solicitudes sin autenticación:

- **Local:** `hf auth login`
- **GitHub Actions:** crear el secreto `HF_TOKEN` en *Settings → Secrets and variables → Actions* y agregarlo a los pasos que ejecutan los notebooks:

  ```yaml
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  ```

**Nunca escribas tokens ni contraseñas en el código ni en el repositorio.**

---

## Dependencias

`numpy`, `pandas`, `matplotlib`, `notebook` y `datasets` (ver `requirements.txt`), además de `pyarrow` y `openpyxl`, que se instalan en el workflow.

## Referencias

- cs-uche. (2024). *Car Dealership* [Conjunto de datos]. Hugging Face. <https://huggingface.co/datasets/cs-uche/car_dealership>
- Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.
- harlfoxem. (2016). *House Sales in King County, USA* [Conjunto de datos]. Kaggle. <https://www.kaggle.com/datasets/harlfoxem/housesalesprediction>
- INRIA-SODA. (2023). *tabular-benchmark* [Conjunto de datos]. Hugging Face. <https://huggingface.co/datasets/inria-soda/tabular-benchmark>
