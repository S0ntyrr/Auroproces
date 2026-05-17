# Auroproces — Análisis Operativo de Procesos de Clientes

Este repositorio contiene un análisis exploratorio/operativo de datos de atención a clientes, enfocado en medir eficiencia, calidad y experiencia del cliente a partir de un dataset en Excel.

## ¿Qué hace este proyecto?
A partir del archivo `dataset_procesos_clientes_01.xlsx`, el script:

- Carga y valida la columna de fecha.
- Limpia datos (nulos, duraciones inválidas y tratamiento de outliers con “winsorizing”/clip).
- Calcula métricas (KPIs) como:
  - Volumen total de casos
  - Tiempo promedio de atención (min)
  - % de casos con error
  - % de estados desconocidos
  - % de cumplimiento de SLA
- Define límites de SLA por tipo de proceso y calcula cumplimiento.
- Clasifica el **riesgo** de cada caso (bajo/medio/alto) según duración vs SLA y presencia de error.
- Genera visualizaciones en HTML con Altair:
  - Duración por proceso (boxplot)
  - Tasa de error por proceso
  - Cumplimiento SLA por proceso
  - Carga operativa por cliente/negocio (Top 10)
  - Estacionalidad (casos por mes)
- Imprime insights y recomendaciones priorizadas.

## Estructura (principal)
- `analisis_procesos_clientes.py`: pipeline principal del análisis.
- `analisis_procesos_clientes_notebook.ipynb`: versión en notebook del análisis.
- `dataset_procesos_clientes_01.xlsx`: dataset de entrada.
- `outputs/`: carpeta donde se guardan las visualizaciones exportadas (HTML).

## Requisitos
- Python 3.x
- Paquetes típicos: `pandas`, `numpy`, `altair`, etc.

## Uso rápido (ejemplo)
1. Asegura que `dataset_procesos_clientes_01.xlsx` esté en la misma carpeta del script.
2. Ejecuta el script:

```bash
python analisis_procesos_clientes.py
```

3. Revisa los archivos generados en `outputs/`.

## Notas
Este análisis está pensado como un insumo para identificar cuellos de botella, procesos con alta tasa de error y oportunidades de automatización o rediseño operativo.
