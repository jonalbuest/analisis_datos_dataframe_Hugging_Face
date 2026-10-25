# Proyecto Integrador III - Analítica de Datos con Hugging Face

Repositorio del **Proyecto Integrador** de la asignatura *Proyecto Integrado III - Analítico de Datos*, que reúne las entregas incrementales (**AE1** y **AE2**) utilizando un paquete transversal de clases en Python (`src/clases/`) para la carga, exploración, transformación y generación automática de reportes sobre datasets de Hugging Face.

## Integrantes - Grupo 1 Canvas

- Jonathan Alvarez Bustamante
- Yusep Alexander Ruiz Agudelo
- Yonier Alexis Quiceno Rodriguez

---

## Avance AE2: Ideación, Diseño y Documentación de la Necesidad

- **Propósito:** Definir y documentar una necesidad de analítica descriptiva sobre el mercado inmobiliario residencial a partir de un dataset tabular de Hugging Face, realizando su reconocimiento estructural, depuración inicial y generación automática de su resumen en texto plano (`.txt`).
- **Necesidad analítica:** Comprender cómo se distribuyen y varían los precios de venta de viviendas residenciales en función de sus características físicas (habitaciones, baños, área habitable, área del lote), calidad constructiva (`grade`), vista al agua (`waterfront`) y ubicación geográfica para apoyar la toma de decisiones de valoración comercial e inversión inmobiliaria.
- **Dataset seleccionado:** [`inria-soda/tabular-benchmark`](https://huggingface.co/datasets/inria-soda/tabular-benchmark/viewer/reg_cat_house_sales)
  - **Identificador en Hugging Face:** `inria-soda/tabular-benchmark`
  - **Configuración (`config`):** `reg_cat_house_sales`
  - **División (`split`):** `train` (21,613 registros originales y 18 columnas; 21,521 registros tras depuración de inconsistencias).

---

## Estructura del repositorio

```text
├── .github/
│   └── workflows/
│       └── ejecutar.yml                 # Flujo CI/CD que ejecuta los notebooks de AE1 y AE2
├── src/
│   ├── clases/                          # Clases reutilizables compartidas por todas las entregas
│   │   ├── __init__.py
│   │   ├── hf_loader.py                 # Clase HFDatasetLoader (descarga desde Hugging Face)
│   │   └── reportes.py                  # Clase Reportes (generación y exportación de resumen .txt)
│   ├── AE1/                             # Entrega Actividad Evaluativa 1
│   │   └── sincronicas/
│   │       ├── datos/
│   │       ├── reportes/
│   │       └── transformaciones.ipynb
│   └── AE2/                             # Actividad 2
│       ├── datos/                       # CSV descargado automáticamente (reg_cat_house_sales.csv)
│       ├── documento/                   # Documento inicial del proyecto (.docx)
│       ├── notebooks/
│       │   └── transformaciones.ipynb   # Notebook descriptivo y de transformación de AE2
│       └── resultados/                  # Resumen automático del DataFrame (reg_cat_house_sales.txt)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Pasos de ejecución local

1. **Crear y activar el entorno virtual:**
   ```bash
   python -m venv .venv
   # En Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # En Linux / macOS:
   source .venv/bin/activate
   ```

2. **Instalar las dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Ejecutar el notebook de AE2:**
   Abrir y ejecutar todas las celdas de `src/AE2/notebooks/transformaciones.ipynb` en VS Code o Jupyter, o ejecutarlo por consola:
   ```bash
   cd src/AE2/notebooks
   jupyter nbconvert --to notebook --execute --inplace transformaciones.ipynb
   ```
   Al finalizar la ejecución, se generarán automáticamente:
   - `src/AE2/datos/reg_cat_house_sales.csv`
   - `src/AE2/resultados/reg_cat_house_sales.txt`
