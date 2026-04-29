# NOTEBOOK: Análisis Operativo - Procesos de Clientes
# Autor: Santiago Rodriguez
# Fecha: 2026
# Dataset: dataset_procesos_clientes_01.xlsx
# Objetivo: Analizar datos operativos de atención a clientes para identificar oportunidades de mejora en eficiencia, calidad y experiencia del cliente.

# 1. Configuracion del Entorno
"""
Se utilizan librerias esenciales para un analisis claro. 
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt

print("Librerias cargadas correctamente")


# 2. Carga del dataset
"""
Se define una ruta explicita y se asegura que la variable fecha se lea de forma correcta.
"""

DATA_PATH = Path("dataset_procesos_clientes_01.xlsx")
def cargar_dataset(ruta: Path) -> pd.DataFrame:
    df = pd.read_excel(ruta)

    # convertir fecha de forma segura
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')

    print(f"Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    return df


df_raw = cargar_dataset(DATA_PATH)

#Vereficicacion del dato "fecha"
print("Tipo de dato de fecha:", df_raw['fecha'].dtype)



# 3. Limpieza basica y variables operativas
"""
Se mantiene un pipeline simple, se eliminan inconsistencias y se crean variables clave para analisi.
"""

SLA_POR_PROCESO = {
    'Realizar pagos': 30,
    'Abrir productos': 60,
    'Reportar Fraudes': 45,
    'Realizar reclamaciones': 90,
    'Solicitar soportes o reportes transaccionales': 60,
}

ESTADOS_VALIDOS = {'exitoso', 'con_error'}

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=['fecha', 'duracion_minutos'])
    df['estado'] = df['estado'].fillna('desconocido').str.lower().str.strip()
    df = df[df['duracion_minutos'] > 0]

    # Winsor para evitar que pocos casos extremos distorsionen el promedio
    q1 = df['duracion_minutos'].quantile(0.25)
    q3 = df['duracion_minutos'].quantile(0.75)
    iqr = q3 - q1
    techo = q3 + 1.5 * iqr
    df['duracion_minutos'] = df['duracion_minutos'].clip(upper=techo)

    df['estado_valido'] = df['estado'].isin(ESTADOS_VALIDOS)
    df['con_error'] = (df['estado'] == 'con_error').astype(int)
    df['sla_limite'] = df['tipo_proceso'].map(SLA_POR_PROCESO)
    df['cumple_sla'] = df['duracion_minutos'] <= df['sla_limite']
    return df


def clasificar_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    def regla(row: pd.Series) -> str:
        sla = row['sla_limite'] if pd.notnull(row['sla_limite']) else 60
        if row['con_error'] == 1 and row['duracion_minutos'] > sla * 1.2:
            return 'alto'
        if row['con_error'] == 1 or row['duracion_minutos'] > sla:
            return 'medio'
        return 'bajo'

    df = df.copy()
    df['riesgo'] = df.apply(regla, axis=1)
    return df


df = limpiar_datos(df_raw)
df = clasificar_riesgo(df)


# 4. KPIs ejecutivos

def calcular_kpis(df: pd.DataFrame) -> dict:
    estados_validos = df['estado_valido'].sum()
    pct_error = round(df.loc[df['estado_valido'], 'con_error'].mean() * 100, 2) if estados_validos else 0.0
    return {
        'volumen_total': len(df),
        'tiempo_promedio_min': round(df['duracion_minutos'].mean(), 1),
        'pct_error': pct_error,
        'pct_estado_desconocido': round((~df['estado_valido']).mean() * 100, 2),
        'pct_cumplimiento_sla': round(df['cumple_sla'].mean() * 100, 2),
    }


kpis = calcular_kpis(df)
print("\nKPIs clave")
print(kpis)

# 5. Calidad de datos

print("\nDiagnóstico de calidad de datos")
print(df_raw.isnull().sum())

q1_raw = df_raw['duracion_minutos'].dropna().quantile(0.25)
q3_raw = df_raw['duracion_minutos'].dropna().quantile(0.75)
iqr_raw = q3_raw - q1_raw
techo_raw = q3_raw + 1.5 * iqr_raw

print("\nDiagnóstico de calidad procesado")
print({
    'registros_con_estado_desconocido': int((~df['estado_valido']).sum()),
    'registros_con_duracion_outlier': int((df_raw['duracion_minutos'].dropna() > techo_raw).sum()),
})

print("\nResumen de duración (dato vacio)")
print(df_raw['duracion_minutos'].describe())

# 6. Visualizacion de KPIs

"""
Cada grafico responde una decision operativa concreta.
"""

alt.data_transformers.disable_max_rows()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Kpi1) Duracion por proceso
Kpi1 = alt.Chart(df).mark_boxplot(extent='min-max', size=40).encode(
    x=alt.X('tipo_proceso:N', title='Tipo de Proceso',
            axis=alt.Axis(labelAngle=-30, labelLimit=200)),
    y=alt.Y('duracion_minutos:Q', title='Duracion (min)',
            scale=alt.Scale(zero=False)),
    color=alt.Color('tipo_proceso:N', legend=None,
                    scale=alt.Scale(scheme='tableau10')),
    tooltip=['tipo_proceso:N',
             alt.Tooltip('mean(duracion_minutos):Q', title='Promedio (min)', format='.1f')]
).properties(
    title='Duracion por tipo de proceso (identifica cuellos de botella)',
    width=700, height=350
)
Kpi1.save(OUTPUT_DIR / "Kpi1_duracion_por_proceso.html")


# Kpi 2) Tasa de error por proceso

error_proceso = df.groupby('tipo_proceso').agg(
    total=('con_error', 'count'),
    errores=('con_error', 'sum')
).reset_index()
error_proceso['pct_error'] = (error_proceso['errores'] / error_proceso['total'] * 100).round(1)

Kpi2 = alt.Chart(error_proceso).mark_bar().encode(
    x=alt.X('tipo_proceso:N', title='Tipo de Proceso',
            axis=alt.Axis(labelAngle=-30, labelLimit=200)),
    y=alt.Y('pct_error:Q', title='% Procesos con Error'),
    tooltip=['tipo_proceso:N', 'pct_error:Q', 'errores:Q', 'total:Q']
).properties(
    title='Tasa de error por proceso (donde priorizar reprocesos)',
    width=700, height=300
)
Kpi2.save(OUTPUT_DIR / "Kpi2_error_por_proceso.html")

# Kpi 3) Cumplimiento SLA por proceso

sla_resumen = df.groupby('tipo_proceso')['cumple_sla'].mean().mul(100).round(1).reset_index()
Kpi3 = alt.Chart(sla_resumen).mark_bar().encode(
    x=alt.X('cumple_sla:Q', title='% Cumplimiento SLA', scale=alt.Scale(domain=[0, 100])),
    y=alt.Y('tipo_proceso:N', title='Tipo de Proceso',
            sort=alt.EncodingSortField('cumple_sla', order='descending')),
    tooltip=['tipo_proceso:N', alt.Tooltip('cumple_sla:Q', title='% SLA', format='.1f')]
).properties(
    title='Cumplimiento SLA por proceso (control de tiempos)',
    width=550, height=280
)
Kpi3.save(OUTPUT_DIR / "Kpi3_sla_por_proceso.html")

# Kpi4) Casos por cliente / negocio

cliente_resumen = df.groupby('id_negocio').agg(
    total_casos=('id_negocio', 'count'),
    interacciones_promedio=('cantidad_interacciones', 'mean'),
    duracion_promedio=('duracion_minutos', 'mean')
).reset_index()

top_clientes = cliente_resumen.sort_values(
    ['total_casos', 'interacciones_promedio'],
    ascending=[False, False]
).head(10)

Kpi4 = alt.Chart(top_clientes).mark_bar().encode(
    x=alt.X('total_casos:Q', title='Total de casos'),
    y=alt.Y('id_negocio:N', title='Cliente / negocio', sort='-x'),
    color=alt.Color('interacciones_promedio:Q', title='Interacciones promedio',
                    scale=alt.Scale(scheme='oranges')),
    tooltip=[
        'id_negocio:N',
        'total_casos:Q',
        alt.Tooltip('interacciones_promedio:Q', title='Interacciones promedio', format='.1f'),
        alt.Tooltip('duracion_promedio:Q', title='Duración promedio (min)', format='.1f')
    ]
).properties(
    title='Carga operativa por cliente (casos e intensidad de interacción)',
    width=700, height=320
)
Kpi4.save(OUTPUT_DIR / "Kpi4_casos_por_cliente.html")

# 5. Estacionalidad

# Kpi 5) Volumen de casos por mes (estacionalidad)

df['mes'] = df['fecha'].dt.month
# Agrupar y ordenar correctamente
casospor_mes = df.groupby('mes').size().reset_index(name='total_casos').sort_values('mes')

mapa_meses = {
    1: 'Enero',
    2: 'Febrero',
    3: 'Marzo',
    4: 'Abril',
    5: 'Mayo',
    6: 'Junio',
    7: 'Julio',
    8: 'Agosto',
    9: 'Septiembre',
    10: 'Octubre',
    11: 'Noviembre',
    12: 'Diciembre',
}

casospor_mes['mes_nombre'] = casospor_mes['mes'].map(mapa_meses)

orden_meses = list(mapa_meses.values())

base_kpi5 = alt.Chart(casospor_mes).encode(
    x=alt.X('mes_nombre:N', sort=orden_meses, title='Mes', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('total_casos:Q', title='Cantidad de casos'),
    tooltip=[
        alt.Tooltip('mes_nombre:N', title='Mes'),
        alt.Tooltip('total_casos:Q', title='Casos', format=',')
    ]
)

bars_kpi5 = base_kpi5.mark_bar(size=34, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    color=alt.Color('total_casos:Q', legend=None, scale=alt.Scale(scheme='tealblues'))
)

labels_kpi5 = base_kpi5.mark_text(dy=-8, fontSize=11).encode(
    text=alt.Text('total_casos:Q', format=',')
)

Kpi5 = (bars_kpi5 + labels_kpi5).properties(
    title='Volumen de casos por mes (estacionalidad de la demanda)',
    width=680, height=320
)

Kpi5.save(OUTPUT_DIR / "Kpi5_casos_por_mes.html")

# 6. Insights clave 
    
"""
Insights clave
"""
mes_max = casospor_mes.sort_values('total_casos', ascending=False).iloc[0]

tiempo_por_proceso = df.groupby('tipo_proceso')['duracion_minutos'].mean().sort_values(ascending=False)
error_por_proceso = error_proceso.set_index('tipo_proceso')['pct_error'].sort_values(ascending=False)
sla_por_proceso = sla_resumen.set_index('tipo_proceso')['cumple_sla'].sort_values()
responsable_casos = df['responsable'].value_counts()
cliente_top = top_clientes.iloc[0]

print("\nINSIGHTS")
print(f"1) Mayor duracion promedio: {tiempo_por_proceso.index[0]} ({tiempo_por_proceso.iloc[0]:.1f} min)")
print(f"2) Mayor tasa de error: {error_por_proceso.index[0]} ({error_por_proceso.iloc[0]:.1f}%)")
print("Esto sugiere que este proceso requiere una intervención prioritaria por su impacto en tiempos y calidad.")
print(f"3) Cliente con mayor carga operativa: {cliente_top['id_negocio']} ({int(cliente_top['total_casos'])} casos, {cliente_top['interacciones_promedio']:.1f} interacciones promedio)")
print(f"4) Responsable con más casos: {responsable_casos.index[0]} ({responsable_casos.iloc[0]} casos)")
print(f"5) Realizar pagos presenta el menor cumplimiento de SLA {sla_por_proceso.index[0]} ({sla_por_proceso.iloc[0]:.1f}%), lo que sugiere retrasos sistemáticos en un proceso crítico para el cliente.")
print(f"6) Se observa un pico de demanda en el mes {mes_max['mes_nombre']} con {int(mes_max['total_casos'])} casos, lo que sugiere estacionalidad en la operación.")


# Recomendaciones
print("\nRECOMENDACIONES PRIORIZADAS")
recomendaciones = [
    '1) Priorizar el proceso con mayor duracion y/o mayor error para revisar guiones, tiempos de respuesta y pasos de resolucion.',
    '2) Investigar los estados desconocidos y definir una regla operativa para capturarlos o excluirlos del dataset.',
    '3) Atender los clientes con mayor carga operativa con automatizacion de autoservicio.',
    '4) Revisar la demanda del mes pico para reforzar capacidad en periodos de mayor volumen.',
]
for recomendacion in recomendaciones:
    print(recomendacion)

# 7. Supuestos 

"""
Supuesto que se pueden tomar con los datos, enfocadas en reducir reprocesos y mejorar tiempos.
"""

print("\nSUPUESTOS DE OPORTUNIDAD")
print("P1. Rediseñar el proceso de Reportar Fraudes, ya que combina alta duración y tasa de error.")
print("P2. Intervenir primero el proceso con mayor tasa de error para reducir el costo y el tiempo perdidos en reprocesos.")
print("P3. Activar alertas al 80% del tiempo límite para transicionar de una gestión reactiva a una prevención real de incumplimientos.")
print("P4. Analizar el comportamiento de sus 90 casos para diseñar un modelo de atención para que minimice su impacto en la carga general.")
print("P5. Distribuir la carga del responsable que tiene 1,978 casos asignados para eliminar cuellos de botella y recuperar la agilidad en la respuesta.")
print("P6. Planificar recursos según picos de demanda para evitar que el servicio sufra una sobrecarga en meses críticos.")

