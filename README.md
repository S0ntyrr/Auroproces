<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=250&section=header&text=Auroproces&fontSize=80&fontAlign=50&fontAlignY=30&animation=twinkling" />
</p>

<p align="center">
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Orbitron&size=26&pause=1000&color=5865F2&center=true&width=620&lines=Analisis+operativo+de+procesos+de+clientes;KPIs+de+tiempo%2C+calidad+y+SLA;Dashboards+interactivos+en+HTML" alt="Typing SVG" /></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/estado-analisis_en_desarrollo-yellow" />
<img src="https://img.shields.io/badge/tecnologia-Python-blue" />
<img src="https://img.shields.io/badge/dashboards-Altair%20%2B%20HTML-informational" />
<img src="https://img.shields.io/badge/dataset-Excel-success" />
<img src="https://img.shields.io/badge/enfoque-operaciones%20y%20experiencia%20de%20cliente-important" />
</p>

---

## 📊 Descripción del Proyecto

**Auroproces** es un análisis operativo de procesos de clientes orientado a identificar cuellos de botella, reprocesos, riesgos de incumplimiento de SLA y oportunidades de mejora en la experiencia del cliente.

El proyecto toma como entrada el archivo `dataset_procesos_clientes_01.xlsx`, limpia los datos, calcula KPIs ejecutivos, clasifica el riesgo de cada caso y genera dashboards interactivos en formato HTML con Altair.

### 🎯 Qué resuelve

| Área | Resultado |
|------|-----------|
| **Calidad de datos** | Depuración de nulos, estados desconocidos y duraciones inválidas |
| **Eficiencia operativa** | Tiempo promedio de atención y distribución por proceso |
| **Cumplimiento SLA** | Medición por tipo de proceso y detección de retrasos |
| **Riesgo operativo** | Clasificación bajo, medio y alto por caso |
| **Demanda** | Detección de picos mensuales y carga por cliente |
| **Visualización** | Dashboards HTML interactivos listos para abrir localmente |

---

## 🛠️ Tecnologías Utilizadas

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Altair-1F77B4?style=for-the-badge&logo=vega&logoColor=white" />
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

---

## 🧩 Estructura del Proyecto

- `analisis_procesos_clientes.py`: pipeline principal del análisis y generación de dashboards.
- `analisis_procesos_clientes_notebook.ipynb`: notebook con el mismo flujo de trabajo, útil para exploración y explicación paso a paso.
- `dataset_procesos_clientes_01.xlsx`: dataset de entrada.
- `outputs/`: carpeta donde se generan los dashboards HTML al ejecutar el script.

---

## 📈 Dashboards Generados

GitHub no renderiza de forma nativa los dashboards HTML interactivos dentro del README, por eso se enlazan como artefactos generados por el proyecto.

Al ejecutar el script se crean estos archivos:

| Dashboard | Archivo |
|-----------|---------|
| Duración por tipo de proceso | [`outputs/Kpi1_duracion_por_proceso.html`](outputs/Kpi1_duracion_por_proceso.html) |
| Tasa de error por proceso | [`outputs/Kpi2_error_por_proceso.html`](outputs/Kpi2_error_por_proceso.html) |
| Cumplimiento SLA por proceso | [`outputs/Kpi3_sla_por_proceso.html`](outputs/Kpi3_sla_por_proceso.html) |
| Carga operativa por cliente | [`outputs/Kpi4_casos_por_cliente.html`](outputs/Kpi4_casos_por_cliente.html) |
| Volumen de casos por mes | [`outputs/Kpi5_casos_por_mes.html`](outputs/Kpi5_casos_por_mes.html) |

Si quieres compartirlos en GitHub Pages, estos mismos HTML se pueden publicar como una demo estática.

---

## 🚀 Instalación y Ejecución

### Requisitos

- Python 3.x
- `pandas`
- `numpy`
- `altair`
- `openpyxl` para leer el Excel

### Paso a paso

1. Ubica el archivo `dataset_procesos_clientes_01.xlsx` en la raíz del proyecto.
2. Instala dependencias:

```bash
pip install pandas numpy altair openpyxl
```

3. Ejecuta el análisis:

```bash
python analisis_procesos_clientes.py
```

4. Abre los archivos generados en `outputs/` con tu navegador.

---

## 🧠 Qué hace el análisis

- Carga y valida la columna de fecha.
- Limpia nulos, estados inválidos y duraciones inconsistentes.
- Aplica tratamiento de outliers sobre la duración.
- Calcula KPIs de volumen, tiempos, error, estado desconocido y cumplimiento de SLA.
- Clasifica el riesgo de cada caso según duración y error.
- Genera insights y recomendaciones priorizadas.

---

## 📘 Notebook

El notebook [`analisis_procesos_clientes_notebook.ipynb`](analisis_procesos_clientes_notebook.ipynb) contiene el mismo flujo del script, pero dividido en celdas para facilitar la revisión, la explicación y la experimentación.

Úsalo si quieres:

- revisar el análisis paso a paso,
- validar la lógica de limpieza y KPIs,
- modificar el flujo sin tocar el script principal,
- documentar el razonamiento detrás de cada visualización.

---

## 🐛 Solución de Problemas

### No aparecen los dashboards

Ejecuta primero el script para regenerar la carpeta `outputs/`:

```bash
python analisis_procesos_clientes.py
```

Luego abre los HTML generados en tu navegador.

### Error al leer el Excel

Verifica que el archivo `dataset_procesos_clientes_01.xlsx` exista en la raíz del proyecto y que tenga la columna `fecha`.

### Problemas con dependencias

Si falta alguna librería, reinstala con:

```bash
pip install -U pandas numpy altair openpyxl
```

---

## 📄 Licencia

Este proyecto se distribuye con fines académicos y de análisis. Si deseas formalizar una licencia específica, puedes agregar un archivo `LICENSE` en el repositorio.

---

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" />
</p>
