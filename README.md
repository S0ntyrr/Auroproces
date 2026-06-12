<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=250&section=header&text=Auroproces&fontSize=72&fontAlign=50&fontAlignY=30&animation=twinkling" />
</p>

<p align="center">
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Orbitron&size=24&pause=1000&color=5865F2&center=true&width=650&lines=Analisis+operativo+de+procesos+de+clientes;KPIs+de+tiempo%2C+calidad+y+SLA;Visualizacion+de+hallazgos+de+negocio" alt="Typing SVG" /></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/estado-analisis_en_desarrollo-yellow" />
<img src="https://img.shields.io/badge/tecnologia-Python-blue" />
<img src="https://img.shields.io/badge/dataset-Excel-success" />
<img src="https://img.shields.io/badge/enfoque-operaciones%20y%20clientes-important" />
</p>

<p align="center">
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Altair-1F77B4?style=for-the-badge&logo=vega&logoColor=white" />
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

## Auroproces

Análisis operativo de procesos de clientes orientado a identificar cuellos de botella, reprocesos, riesgos de incumplimiento de SLA y oportunidades de mejora en la experiencia del cliente.

## Descripción

El proyecto toma como entrada `dataset_procesos_clientes_01.xlsx`, limpia los datos, calcula KPIs ejecutivos, clasifica el riesgo de cada caso y genera visualizaciones en HTML con Altair.

## Objetivo del análisis

- Medir el volumen de casos y el tiempo promedio de atención.
- Identificar procesos con mayor tasa de error.
- Evaluar el cumplimiento de SLA por tipo de proceso.
- Detectar cargas operativas elevadas por cliente o negocio.
- Encontrar estacionalidad en la demanda mensual.

## Estructura del repositorio

- `analisis_procesos_clientes.py`: script principal del análisis.
- `analisis_procesos_clientes_notebook.ipynb`: notebook con el flujo documentado paso a paso.
- `dataset_procesos_clientes_01.xlsx`: archivo de entrada.
- `outputs/`: carpeta donde se guardan las visualizaciones HTML generadas al ejecutar el script.

## Requisitos

- Python 3.x
- `pandas`
- `numpy`
- `altair`
- `openpyxl`

## Instalación y ejecución

1. Coloca `dataset_procesos_clientes_01.xlsx` en la raíz del proyecto.
2. Instala dependencias.

```bash
pip install pandas numpy altair openpyxl
```

3. Ejecuta el análisis.

```bash
python analisis_procesos_clientes.py
```

4. Revisa la carpeta `outputs/` para abrir las visualizaciones generadas.

## Notebook

El archivo `analisis_procesos_clientes_notebook.ipynb` contiene la misma lógica del script, organizada en celdas para facilitar revisión, validación y modificación del análisis.

## Resultados generados

Al ejecutar el script se producen archivos HTML interactivos en `outputs/` con las visualizaciones del análisis. No se incluyen capturas en este README porque no existen artefactos estáticos dentro del repositorio.

## Notas técnicas

- La columna `fecha` se convierte con `pd.to_datetime(..., dayfirst=True)`.
- Se limpian nulos, valores inválidos y duraciones extremas.
- Se calcula un indicador de riesgo por caso basado en duración y error.

## Uso profesional del proyecto

Este repositorio está enfocado en mostrar capacidad de análisis de datos, limpieza, generación de KPIs y comunicación de hallazgos de negocio de forma clara y reproducible.

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" />
</p>
