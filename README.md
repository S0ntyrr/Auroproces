# Auroproces

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
