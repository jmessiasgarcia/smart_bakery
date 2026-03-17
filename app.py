import os
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import logging
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from xgboost import XGBRegressor
from datetime import datetime


# Configuración de logs
logging.getLogger(
    "streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="🥐 Dashboard Rentabilidad", layout="wide")

# LOADING DATA


@st.cache_data
def load_data():
    # Carga de archivos físicos
    df_f = pd.read_csv('facturacion_limpia.csv')
    df_a = pd.read_csv('areas_limpia.csv')
    df_final = pd.read_csv('df_final.csv')
    df_margen_raw = pd.read_csv('margen_clean_2.csv')

    # Procesamiento de Clientes y Zonas
    df_f['Código cliente'] = df_f['Código cliente'].astype(str).str.strip()
    df_a['Código cliente'] = df_a['Código cliente'].astype(str).str.strip()

    df_facturacion_zonas = pd.merge(
        df_f, df_a[['Código cliente', 'Zona']], on='Código cliente', how='left')

    df_facturacion_zonas['Zona'] = df_facturacion_zonas['Zona'].fillna(
        'Sin Clasificar')

    df_facturacion_zonas['Fecha'] = pd.to_datetime(
        df_facturacion_zonas['Fecha'])

    df_final.columns = df_final.columns.str.strip()
    df_margen_raw.columns = df_margen_raw.columns.str.strip()

    cols_numericas = ['Multiplicador',
                      'Margen Bruto Unitario', 'Brecha de Marcación']
    for col in cols_numericas:
        if col in df_margen_raw.columns:
            df_margen_raw[col] = pd.to_numeric(
                df_margen_raw[col], errors='coerce')

    return df_facturacion_zonas, df_final, df_margen_raw, df_a


# --- ESTA LÍNEA ES LA CLAVE ---
# Aquí "recibimos" las 4 cosas que tu función devuelve (return)
df_facturacion_zonas, df_final, df_margen_raw, df_a = load_data()

# --- LÓGICA DE SEGMENTACIÓN ---


def definir_segmento(row):
    v23, v24, v25 = row['Euros 2023'], row['Euros 2024'], row['Euros 2025']
    total = v23 + v24 + v25
    if total <= 0:
        return "Nunca ha comprado"
    if v23 > 0 and v24 > 0 and v25 > 0:
        return "Cliente Fiel (3 años)"
    if v25 > 0 and v23 <= 0 and v24 <= 0:
        return "Cliente Nuevo (2025)"
    if v25 <= 0 and (v23 > 0 or v24 > 0):
        return "Cliente Perdido / Riesgo"
    return "Cliente Intermitente"


# --- LLAMADA A LA FUNCIÓN ---
df_fase1, df_final, df_margen_raw, df_areas = load_data()

#################################################################################################

st.title("SMART BAKERY: Panel de Inteligencia Estratégica")

st.markdown("""
###### Analista: José Messias Garcia da Silva Ferreira
""", unsafe_allow_html=True)

df_filtrado = df_fase1

# --- SECCIÓN 1 ---
st.header("I. Salud de Cartera y Pulso de Mercado")

OBJETIVO_VENTAS = 1500000  # nao utilizo
OBJETIVO_CLIENTES = 500    # nao utilizo


# --- BLOQUE 1: KPIs CON COMPARATIVA ---
st.markdown("######    Indicadores Clave de Rendimiento")
c1, c2, c4 = st.columns(3)

# 1. Ventas Totales y % sobre objetivo
total_vta = df_filtrado['Importe_Euros'].sum()
progreso_vta = (total_vta / OBJETIVO_VENTAS) * 100
falta_vta = OBJETIVO_VENTAS - total_vta
c1.metric(
    label="Revenue",
    value=f"{total_vta:,.0f} €",
    delta_color="normal",  # Verde si sube
    help="Suma total de los ingresos facturados en el periodo y zona seleccionados. Es el volumen bruto de ventas antes de gastos."
)

# 2. Clientes Activos vs Objetivo
num_clientes = df_filtrado['Código cliente'].nunique()

c2.metric(
    label="Clientes",
    value=f"{num_clientes}",

    delta_color="inverse"  # Rojo si falta
)


# 4. Volumen de Unidades
total_unidades = df_filtrado['Cantidad_Unidades'].sum()
c4.metric(
    label="Unidades Vendidas",
    value=f"{total_unidades:,.0f}",

)


# --- CÁLCULO DE DF_TEMPORAL ---
df_temporal = (
    df_filtrado
    .set_index('Fecha')
    .resample('ME')['Importe_Euros']  # 'ME' es Month End, más seguro
    .sum()
    .reset_index()
)
# --- CÁLCULO DE DF_TEMPORAL
df_temporal = (
    df_filtrado
    .set_index('Fecha')
    .resample('ME').agg({
        'Importe_Euros': 'sum',
        'Cantidad_Unidades': 'sum'
    })
    .reset_index()
)

df_temporal['Fecha'] = pd.to_datetime(df_temporal['Fecha'])

# --- CREACIÓN DEL GRÁFICO CON DOS EJES ---
fig_line = go.Figure()

# Línea de Ingresos (Eje Y principal)
fig_line.add_trace(go.Scatter(
    x=df_temporal['Fecha'],
    y=df_temporal['Importe_Euros'],
    name='Facturación (€)',
    mode='lines',
    line=dict(color='#2EC18E', width=2.5, shape='spline'),
    yaxis='y1'
))

# Línea de Unidades (Eje Y secundario)
fig_line.add_trace(go.Scatter(
    x=df_temporal['Fecha'],
    y=df_temporal['Cantidad_Unidades'],
    name='Unidades Vendidas',
    mode='lines',
    line=dict(color='#10BCF6', width=2, dash='dot', shape='spline'),
    yaxis='y2'
))

# --- CONFIGURACIÓN DE EJES Y LAYOUT ---
fig_line.update_layout(
    title='Evolución de Ingresos y Volumen por Mes',
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom",
                y=1.02, xanchor="right", x=1),

    # Eje Y Principal: Euros
    yaxis=dict(
        title=dict(
            text="Facturación (€)",
            font=dict(color="#2EC18E")  # Antes era titlefont
        ),
        tickfont=dict(color="#2EC18E"),
        showgrid=False
    ),

    # Eje Y Secundario: Unidades
    yaxis2=dict(
        title=dict(
            text="Unidades",
            font=dict(color="#10BCF6")  # Antes era titlefont
        ),
        tickfont=dict(color="#10BCF6"),
        anchor="x",
        overlaying="y",
        side="right",
        showgrid=False
    )
)

# Sombreado por años
start_year = df_temporal['Fecha'].min().year
end_year = df_temporal['Fecha'].max().year

for year in range(start_year, end_year + 1):
    color = "rgba(100, 149, 237, 0.05)" if year == 2023 else \
            "rgba(60, 179, 113, 0.05)" if year == 2024 else \
            "rgba(255, 165, 0, 0.05)"

    fig_line.add_vrect(
        x0=f"{year}-01-01", x1=f"{year}-12-31",
        fillcolor=color, layer="below", line_width=0,
        annotation_text=str(year), annotation_position="top left"
    )

# -MONTH
fig_line.update_xaxes(
    type="date",
    tickmode="linear",
    # Tick0 fuerza el inicio en el primer día del año para alinear las marcas
    tick0=f"{df_temporal['Fecha'].min().year}-01-01",
    dtick="M3",           # Una etiqueta cada 3 meses (Ene, Abr, Jul, Oct)
    tickformat="%b\n%Y",  # Mes arriba y Año abajo para que no se amontone
    ticklabelmode="period",  # Centra el texto en el periodo
    showgrid=False,
    anchor="y",
    side="bottom"
)

# Esto asegura que la línea vertical del cursor sea precisa
fig_line.update_layout(
    hovermode="x unified",
    xaxis=dict(spikethickness=1, spikedash="dot",
               spikecolor="#999999", spikesnap="data")
)

st.plotly_chart(fig_line, width='stretch')

# --- CÁLCULOS PARA KPIs DE CLIENTES ---

st.divider()
st.subheader("Estructura de Ventas")

# st.markdown("######    Indicadores Clave de Rendimiento")

# --- CÁLCULOS PARA KPIs DE ESTRUCTURA ---
limite_superior = df_filtrado['Importe_Euros'].quantile(0.9)
df_hist = df_filtrado[df_filtrado['Importe_Euros'] <= limite_superior]

# 1. ¿Cuál es el rango donde más vendemos? (La Moda)
bins = pd.cut(df_hist['Importe_Euros'], bins=50)
rango_top = bins.value_counts().idxmax()

# 2. % de pedidos "Pequeños" (por debajo de la mediana)
mediana_val = df_filtrado['Importe_Euros'].median()
pedidos_bajos = (df_filtrado['Importe_Euros'] < mediana_val).mean()


# --- CÁLCULOS ---
df_sorted = df_filtrado.sort_values('Importe_Euros', ascending=False)
top_20_count = int(len(df_sorted) * 0.2)
ingresos_top_20 = df_sorted.iloc[:top_20_count]['Importe_Euros'].sum()
total_ingresos = df_filtrado['Importe_Euros'].sum()
pareto_val = ingresos_top_20 / total_ingresos if total_ingresos > 0 else 0

# 1. Preparación de datos
df_hist_pos = df_hist[df_hist['Importe_Euros'] > 0]
mediana_ticket = df_hist_pos['Importe_Euros'].median()

# 2. Crear gráfico
fig_dist = px.histogram(
    df_hist_pos,
    x="Importe_Euros",
    nbins=250,
    marginal="box",
    title="Distribución y Outliers",
    labels={
        'Importe_Euros': 'Importe del Ticket (€)'},
    color_discrete_sequence=["#2EC18E"],
    opacity=1,

)

# 3. Configuramos la MEDIANA como un elemento de la LEYENDA
# Añadimos un "rastro" invisible solo para que aparezca en la leyenda
fig_dist.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='lines',
    line=dict(color='#FF4BE1', width=2, dash='dash'),
    name=f'Mediana: {mediana_ticket:.2f}€'
))

# 4. Añadimos la línea física (pero sin el texto que se duplica)
fig_dist.add_vline(
    x=mediana_ticket,
    line_dash="dash",
    line_color="#FF4BE1",
    line_width=2
)

# 5. LIMPIEZA TOTAL: Sin grids, sin etiquetas duplicadas en el BoxPlot
fig_dist.update_traces(
    hoverinfo='skip',
    marker_color="#397C92",
    selector=dict(type='box')
)

fig_dist.update_layout(
    template="plotly_dark",
    # Quitamos todos los grids
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        range=[0, df_hist_pos['Importe_Euros'].quantile(0.99)]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False
    ),
    # Configuramos la leyenda arriba
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    bargap=0.05,
    height=600,
    showlegend=True
)

fig_dist.update_yaxes(title_text="Frecuencia", showgrid=False, zeroline=False)

st.plotly_chart(fig_dist, width='stretch')

st.info("""
    * **La Mediana (Línea Roja):** Es el punto exacto donde se divide nuestros clientes. El 50% de los pedidos están por debajo de este valor y el otro 50% por encima.
    * **Impacto en el Desperdicio:** Entender el ticket típico nos ayuda a prever cuántas unidades de cada producto se llevan en promedio. Si el ticket mediano sube pero las unidades bajan, significa que vendemos productos más caros (más margen, menos merma logística).
""")


# ==========================================
# 📈 CALCULADORA DE IMPACTO (EL PORQUÉ DEL 8%)
# ==========================================
with st.expander("Potencial de crecimiento"):
    # Valores basados en tu gráfico
    ticket_medio_actual = 240
    aumento_propuesto = 20
    ticket_objetivo = ticket_medio_actual + aumento_propuesto

    # Cálculo del porcentaje de crecimiento del ticket
    # (20 / 240) * 100 = 8.33%
    porcentaje_crecimiento = (aumento_propuesto / ticket_medio_actual) * 100

    st.write(
        f"### ¿Por qué decimos que la facturación subiría un {porcentaje_crecimiento:.1f}%?")

    st.write(f"**Ticket Medio Actual:** {ticket_medio_actual}€")
    st.write(f"**Nuevo Ticket:** {ticket_objetivo}€")

    st.info(
        f"Subir solo **20€** en cada pedido de tus 1.366 clientes genera un impacto directo del **{porcentaje_crecimiento:.1f}%** en el total de ventas.")


st.divider()

# ==========================================
# 🔄 ANÁLISIS DE RETENCIÓN MENSUAL (MES A MES)
# ==========================================
st.subheader("Fidelidad Mes a Mes")

# 1. Preparar datos de facturación
df_facturacion_zonas['Mes_Año'] = df_facturacion_zonas['Fecha'].dt.to_period(
    'M')

# 2. Obtener lista de clientes únicos por cada mes
clientes_por_mes = df_facturacion_zonas.groupby(
    'Mes_Año')['Código cliente'].unique()

# 3. Calcular la retención entre los dos últimos meses cerrados
# Tomamos el último mes y el anterior
meses_disponibles = sorted(clientes_por_mes.index.tolist())

if len(meses_disponibles) >= 2:
    ultimo_mes = meses_disponibles[-1]
    mes_anterior = meses_disponibles[-2]

    set_clientes_ultimo = set(clientes_por_mes[ultimo_mes])
    set_clientes_anterior = set(clientes_por_mes[mes_anterior])

    # Clientes que compraron en AMBOS meses
    clientes_fieles_mes = set_clientes_ultimo.intersection(
        set_clientes_anterior)

    tasa_retencion_mensual = (
        len(clientes_fieles_mes) / len(set_clientes_anterior)) * 100

    # Mostrar el KPI Mensual
    col_m1, col_m2 = st.columns(2)

    col_m1.metric(
        label=f"Retención {mes_anterior} | {ultimo_mes}",
        value=f"{tasa_retencion_mensual:.1f}%",
        delta=f"{len(clientes_fieles_mes)} Clientes Repitieron",
        help="Porcentaje de clientes del mes pasado que han vuelto a comprar este mes."
    )

    col_m2.metric(
        label="Clientes Fugados (Churn)",
        value=len(set_clientes_anterior) - len(clientes_fieles_mes),
        delta="Revisar urgencia",
        delta_color="inverse",
        help="Clientes que compraron el mes pasado pero no han comprado en este."
    )
else:
    st.info("Faltan datos históricos mensuales para calcular la retención.")

    ###############


df_fidelidad = df_areas.copy()
df_fidelidad['Segmento'] = df_fidelidad.apply(definir_segmento, axis=1)
df_fidelidad['Valor Total'] = df_fidelidad['Euros 2023'] + \
    df_fidelidad['Euros 2024'] + df_fidelidad['Euros 2025']


mapa_colores = {
    # El color más oscuro y sólido (Representa estabilidad)
    "Cliente Fiel (3 años)": "#0F7091",  # Teal Oscuro Profundo

    # El color intermedio (Representa crecimiento y agua)
    "Cliente Nuevo (2025)": "#26A69A",   # Teal Medio Vibrante

    # Un color que resalte pero sin ser un rojo chillón (Contraste suave)
    # Menta muy pálido (Casi blanco/gris)
    "Cliente Perdido / Riesgo": "#15F2A8",

    # Un tono que conecta los dos verdes (Dinámico)
    "Cliente Intermitente": "#71CFAB",   # Teal Suave

    # El color base neutro para lo que no tiene datos
    "Nunca ha comprado": "#E0F2F1"       # Blanco Menta (Capa muy ligera)
}

col1, col2 = st.columns([1, 1.2])

with col1:
    conteo = df_fidelidad['Segmento'].value_counts().reset_index()
    fig_pie = px.pie(conteo, values='count', names='Segmento',
                     hole=0.7, color='Segmento', color_discrete_map=mapa_colores)

    fig_pie.update_layout(
        title=dict(
            text="Volumen de Clientes",
            x=0.05,               # <--- Posición cerca del borde izquierdo
            xanchor='left',       # <--- El punto de anclaje es el inicio del texto
            font=dict(size=15)    # Opcional: para que resalte más como título
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5
        ),
        # Aumentamos t (top) para que el título no se pegue al gráfico
        margin=dict(t=80, b=100, l=0)
    )

    st.plotly_chart(fig_pie, width='stretch')


with col2:
    # Agrupar por zona y sumar ventas
    df_zona = df_filtrado.groupby('Zona')['Importe_Euros'].sum().reset_index()

    # Crear gráfico de barras
    fig_bar = px.bar(
        df_zona,
        x='Zona',
        y='Importe_Euros',
        color='Importe_Euros',
        title='Facturación por Zona',
        color_continuous_scale='Tealgrn',
        labels={
            "Zona": "Región",          # nombre del eje X
            "Importe_Euros": "Facturación (€)"  # nombre del eje Y
        }
    )

    # Quitar grid y limpiar fondo
    fig_bar.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        # fondo de todo el canvas
    )

    # Mostrar en Streamlit)
    st.plotly_chart(fig_bar, width='stretch')


st.divider()
st.subheader("Análisis de Pareto: Concentración de Clientes")

# 1. Usamos el DataFrame que contiene el detalle por cliente
# AJUSTA AQUÍ: Cambia 'df_fase1' por el nombre de tu variable que carga 'facturacion_limpia.csv'
df_p = df_fase1.copy()

try:
    # Agrupamos por 'Código cliente' sumando su facturación total
    df_pareto = df_p.groupby('Código cliente')['Importe_Euros'].sum(
    ).sort_values(ascending=False).reset_index()

    # 2. Cálculos de porcentajes acumulados
    df_pareto['Ventas_Acum_Perc'] = (
        df_pareto['Importe_Euros'].cumsum() / df_pareto['Importe_Euros'].sum()) * 100
    df_pareto['Clientes_Acum_Perc'] = (
        (df_pareto.index + 1) / len(df_pareto)) * 100

    # 3. Gráfico de Pareto (Curva de Lorenz)
    fig_pareto = go.Figure()

    # Área de la curva
    fig_pareto.add_trace(go.Scatter(
        x=df_pareto['Clientes_Acum_Perc'],
        y=df_pareto['Ventas_Acum_Perc'],
        fill='tozeroy',
        name='Venta Acumulada',
        line=dict(color='#15F2A8', width=3),
        hovertemplate="<b>% de Clientes:</b> %{x:.1f}%<br><b>% de Ventas:</b> %{y:.1f}%<extra></extra>"
    ))

    # Línea del 80/20 (Punto crítico)
    fig_pareto.add_hline(y=80, line_dash="dot", line_color="#E433C7",
                         annotation_text="Límite 80% Ventas", annotation_position="bottom right")

    fig_pareto.add_vline(x=20, line_dash="dot",
                         line_color="rgba(255,255,255,0.5)")

    fig_pareto.update_layout(
        template="plotly_dark",
        xaxis_title="% de Clientes (Ordenados de mayor a menor gasto)",
        yaxis_title="% de Facturación Total",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_pareto, width='stretch')

    # 4. KPI de Interpretación
    v20 = df_pareto[df_pareto['Clientes_Acum_Perc']
                    <= 20]['Ventas_Acum_Perc'].max()

    st.info(f"""
    **Diagnóstico de Pareto:**
    El **20%** de los clientes actuales representan el **{v20:.1f}%** de tus ingresos totales.
    
    * **Si es cercano al 80%**: El negocio depende críticamente de unos pocos clientes (fidelización vital).
    * **Si es cercano al 30-40%**: El negocio está muy diversificado, lo cual es muy seguro ante bajas de clientes.
    """)

except Exception as e:
    st.error(
        f"Error: Asegúrate de que el DataFrame contiene la columna 'Código cliente'. Detalle: {e}")

    # ==========================================
# 💡 ESTRATEGIA PARA EL GRUPO B y C (El 80% de clientes)
# ==========================================
with st.expander("Recomendación Estratégica: Optimización del 77%"):
    st.markdown("""
    ### Objetivo: Maximizar la Eficiencia Operativa
    Para los clientes que generan el 23% de la facturación, el enfoque no es el crecimiento agresivo, sino la **rentabilidad por ahorro de tiempo**:
    
    1. **Estandarización de Pedidos:** Crear "Cestas Tipo" para estos clientes. Que no pierdan (ni nos hagan perder) tiempo eligiendo; que repitan su pedido habitual con un solo click.
    2. **Optimización Logística:** Agrupar las entregas de estos clientes en días específicos de la semana para que el camión siempre vaya lleno y la ruta sea circular.
    3. **Autogestión:** Fomentar que el pedido entre directamente por la App/Web sin llamadas telefónicas. Esto libera a tu equipo administrativo para cuidar al 20% de clientes VIP.
    4. **Umbral de Rentabilidad:** Revisar si los pedidos de menos de X euros cubren el coste de transporte. Si no es así, proponer entregas quincenales en lugar de semanales.
    """)
    st.info("*La rentabilidad no solo viene de vender más, sino de que atender al cliente pequeño no nos cueste dinero.*")

# 1. Preparamos los datos
top_10_clientes = df_pareto.head(10).copy()

# 2. Quitamos los decimales y ponemos formato moneda (punto para miles)
top_10_clientes['Importe_Euros'] = top_10_clientes['Importe_Euros'].apply(
    lambda x: f"{int(x):,}".replace(",", ".") + " €"
)

# 3. Limpiamos el nombre del NaN si existe
top_10_clientes['Código cliente'] = top_10_clientes['Código cliente'].fillna(
    'Venta No Identificada')

# 4. Lo mostramos en el expander
with st.expander("Nuestros 10 Clientes 'Imprescindibles'", expanded=False):
    st.table(top_10_clientes[['Código cliente', 'Importe_Euros']])

    st.warning(f"""
    El **Cliente 2117** es el motor de la fábrica, pero eso nos pone en una situación delicada: **si ellos dejan de comprar, el negocio sufre un golpe durísimo de la noche a la mañana.**

    **Recomendación:**

    1.  **Mima al Gigante, pero busca hermanos:** No dejes de cuidar al Cliente 2117, pero nuestra prioridad debería ser captar otros 2 o 3 clientes de ese perfil para repartir el peso de la facturación.
    2.  **Ponle cara a los 'desconocidos':** Tenemos **350.000 €** en ventas (el grupo 'Venta No Identificada') que no sabemos quiénes son. Es fundamental investigar de dónde vienen esas ventas para ver si podemos convertir a esos clientes anónimos en clientes VIP identificados.
    3.  **Diversificar es ganar salud:** Cuantos más clientes medianos tengamos menos poder tendrá un solo cliente sobre el futuro de tu panadería.

    """)

st.divider()

st.subheader(f"Detalle de Cliente")

# 1. Aseguramos que la columna sea numérica para evitar errores de comparación
df_p['Código cliente'] = pd.to_numeric(df_p['Código cliente'], errors='coerce')

# 2. Caja de entrada (usamos text_input para tener más control o number_input)
id_cliente_audit = st.number_input(
    "Escribe el código del cliente para investigar:", value=2117, step=1)

# 3. FILTRADO DINÁMICO (Forzamos la comparación a entero)
df_audit = df_p[df_p['Código cliente'] == id_cliente_audit].copy()

if not df_audit.empty:
    # --- MÉTRICAS ---
    total_v = df_audit['Importe_Euros'].sum()

    # Intentamos buscar la columna de facturas, si no existe, contamos registros
    if 'Factura' in df_audit.columns:
        pedidos_v = df_audit['Factura'].nunique()
        etiqueta_pedidos = "Nº de Facturas"
    else:
        pedidos_v = len(df_audit)
        etiqueta_pedidos = "Nº de Operaciones"

    col1, col2 = st.columns(2)
    col1.metric(f"Facturación Total",
                f"{int(total_v):,}".replace(",", ".") + " €")
    col2.metric(etiqueta_pedidos, pedidos_v)

    # --- GRÁFICO TENDENCIA CON SOMBRAS ANUALES ---
    st.subheader("Evolución de compras mensuales")

    df_tendencia = df_audit.groupby(df_audit['Fecha'].dt.to_period('M'))[
        'Importe_Euros'].sum().reset_index()
    df_tendencia['Fecha'] = df_tendencia['Fecha'].astype(str)

    # Crear la base del gráfico
    fig_tend = px.line(df_tendencia, x='Fecha', y='Importe_Euros',
                       markers=True, color_discrete_sequence=['#15F2A8'])

    # Extraer los años únicos para crear las sombras
    df_tendencia['Año'] = df_tendencia['Fecha'].str[:4]
    años_unicos = df_tendencia['Año'].unique()

    # Añadir sombras discretas alternas para cada año
    for i, anio in enumerate(años_unicos):
        # Sombreamos solo los años pares (o impares) para crear contraste
        if i % 2 == 0:
            # Buscamos el primer y último mes de ese año en los datos
            meses_anio = df_tendencia[df_tendencia['Año'] == anio]['Fecha']

            fig_tend.add_vrect(
                x0=meses_anio.iloc[0],
                x1=meses_anio.iloc[-1],
                fillcolor="white",
                opacity=0.05,  # Sombra muy discreta
                layer="below",
                line_width=0,
                annotation_text=anio,
                annotation_position="top left"
            )

    # Estética final
    fig_tend.update_layout(
        template="plotly_dark",
        xaxis_title="Meses",
        yaxis_title="Facturación (€)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_tend, width='stretch')


else:
    st.warning(
        f"No se han encontrado datos para el cliente {int(id_cliente_audit)}. Verifica si el código es correcto en la tabla superior.")

# ==========================================
# ✅ FICHA DE SEGUIMIENTO ESTRATÉGICO (CORREGIDA)
# ==========================================

# 1. Detectamos cómo se llama tu columna de dinero (Euros, Importe, etc.)
# Buscamos una columna que se llame 'Euros' o 'Importe' o la primera numérica que no sea el ID
posibles_nombres = ['Euros', 'Importe', 'Ventas', 'Total']
col_dinero = next((c for c in posibles_nombres if c in df_audit.columns),
                  df_audit.select_dtypes(include=['number']).columns[-1])

# 2. Ahora sí, calculamos con el nombre real
fact_cliente = df_audit[col_dinero].sum()
fact_total_empresa = df_p[col_dinero].sum()

st.subheader(f"Análisis Operativo: Cliente {id_cliente_audit}")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Estado de Actividad**")
    st.write("🟢 Activo")

with col2:
    st.write("**Peso en el Negocio**")
    if fact_total_empresa > 0:
        peso = (fact_cliente / fact_total_empresa) * 100
        st.write("Relevancia:", f"{peso:.2f}%")
    else:
        st.write("Calculando...")

with col3:
    st.write("**Acción Recomendada**")
    if fact_cliente > 50000:
        st.info("Prioridad en Logística")
    else:
        st.write("Mantener frecuencia estándar")


st.divider()
# --- SECCIÓN 2: OPTIMIZACIÓN DEL MARGEN Y RENTABILIDAD ---
st.header("II. Análisis de Matriz BCG (Boston Consulting Group)")
st.markdown("""
    ###### Productos con un **Margen > 30%**.
""")

total_vta_f2 = df_final['Importe_Euros'].sum()
total_beneficio = df_final['Beneficio_Real_Euros'].sum()
margen_medio_total = (total_beneficio / total_vta_f2) * \
    100 if total_vta_f2 != 0 else 0
st.markdown("######    Indicadores Clave de Rendimiento")

c1, c4 = st.columns(2)
c1.metric("Revenue", f"{total_vta_f2:,.0f} €")
# c3.metric("Margen Medio", f"{margen_medio_total:.1f}%")
c4.metric("Catálogo Activo",
          f"{df_final['Nombre Artículo'].nunique()} productos.")


df_resumen = df_final.groupby('Nombre Artículo').agg(
    {'Importe_Euros': 'sum', 'Beneficio_Real_Euros': 'sum', 'Cantidad_Unidades': 'sum'}).reset_index()
df_resumen = df_resumen[df_resumen['Importe_Euros'] != 0]
df_resumen['Margen'] = (
    df_resumen['Beneficio_Real_Euros'] / df_resumen['Importe_Euros']) * 100

fig_bubble = px.scatter(df_resumen, x='Importe_Euros', y='Margen', size='Cantidad_Unidades', color='Margen',
                        hover_name='Nombre Artículo', size_max=60, color_continuous_scale=["#F33939", "#15F2A8"], title="Matriz Volumen vs Margen",
                        labels={
                            'Importe_Euros': 'Ventas Totales', 'Margen': 'Margen (%)'}
                        )

fig_bubble.add_hline(y=0, line_dash="solid", line_color="red")


# --- 1. CÁLCULO DE MEDIAS ---
media_ventas = df_resumen['Importe_Euros'].mean()
media_margen = df_resumen['Margen'].mean()

# --- 2. GRÁFICO DE MATRIZ ESTRATÉGICA ---
fig_bubble = px.scatter(
    df_resumen,
    x='Importe_Euros',
    y='Margen',
    size='Cantidad_Unidades',
    color='Margen',
    hover_name='Nombre Artículo',
    size_max=60,
    color_continuous_scale=["#F33939", "#F6D258", "#15F2A8"],
    title="Análisis Estratégico de Cartera",
    labels={'Importe_Euros': 'Ventas Totales (€)', 'Margen': 'Margen (%)'}
)

# Líneas de Cruce (Medias)
fig_bubble.add_vline(x=media_ventas, line_dash="dot",
                     line_color="rgba(255,255,255,0.5)")
fig_bubble.add_hline(y=media_margen, line_dash="dot",
                     line_color="rgba(255,255,255,0.5)")

# --- 3. ANOTACIONES AJUSTADAS ---
# Usamos paper coordinates (xref="paper") para que las etiquetas se queden en las esquinas
# independientemente de cuánto crezca el gráfico.
fig_bubble.add_annotation(xref="paper", yref="paper", x=0.95, y=0.95,
                          text="⭐", showarrow=False, font_color="#15F2A8", font_size=35)
fig_bubble.add_annotation(xref="paper", yref="paper", x=0.05, y=0.95,
                          text="🐱", showarrow=False, font_color="#D7F610", font_size=35)
fig_bubble.add_annotation(xref="paper", yref="paper", x=0.95, y=0.05,
                          text="🐮", showarrow=False, font_color="#F6D258", font_size=35)
fig_bubble.add_annotation(xref="paper", yref="paper", x=0.05, y=0.05,
                          text="🐶", showarrow=False, font_color="#F33939", font_size=35)

# --- 4. AJUSTE DE TAMAÑO (AQUÍ ESTÁ LA CLAVE) ---
fig_bubble.update_layout(
    template="plotly_dark",
    height=650,  # Aumentamos la altura de los 450/600 por defecto a 800px
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, zeroline=False),
    # Ajustamos márgenes para aprovechar el espacio
    margin=dict(l=20, r=20, t=60, b=20)
)
fig_bubble.update_traces(marker=dict(line=dict(width=0)))
fig_bubble.add_hline(y=0, line_dash="solid", line_color="red")


# Renderizado ocupando todo el ancho disponible
st.plotly_chart(fig_bubble, width='stretch')


with st.expander("Análisis Detallado de Márgenes y Costes (Escandallos)"):
    st.write("### Control de Rentabilidad por Artículo")
    st.info("Esta tabla cruza tus costes de producción con los precios de venta para asegurar que cada producto sea rentable.")

    # 1. Preparar la tabla para mostrar (puedes elegir las columnas que más le importen a él)
    # Usamos df_margen_raw que ya cargaste
    df_mostrar_margen = df_margen_raw[[
        'Código Artículo', 'Nombre Artículo', 'Coste Producción Unitario',
        'Venta Neta Unitario', 'Margen Bruto Unitario', 'Margen Total Contribuido (€)'
    ]].copy()

    # 2. Añadir un semáforo visual de rentabilidad
    # Si el margen es menor al 30%, es una alerta para el dueño
    def resaltar_margen(val):
        color = 'red' if val < 0.30 else 'green'
        return f'color: {color}; font-weight: bold'

    # 3. Mostrar la tabla con estilo
    st.dataframe(
        df_mostrar_margen.style.format({
            'Coste Producción Unitario': '{:.2f} €',
            'Venta Neta Unitario': '{:.2f} €',
            'Margen Bruto Unitario': '{:.1%}',
            'Margen Total Contribuido (€)': '{:.2f} €'
        }).map(resaltar_margen, subset=['Margen Bruto Unitario']),
        use_container_width=True,
        hide_index=True
    )

    # 4. BOTÓN DE ACCIÓN: Análisis de Productos Críticos
    productos_criticos = df_margen_raw[df_margen_raw['Margen Bruto Unitario'] < 0.25]
    if not productos_criticos.empty:
        st.warning(
            f"⚠️ Atención : Tienes {len(productos_criticos)} productos con margen inferior al 25%. Deberíamos revisar sus costes o precios.")


with st.expander("Plan de Acción"):

    st.markdown(f"""
        <div style="border-left: 5px solid #15F2A8; padding: 5px 15px; margin-bottom: 15px;">
            <span style="font-weight: bold; color: #15F2A8; font-size: 18px;">⭐ Estrellas</span><br>
            <span style="color: #E0E0E0;">Prioridad absoluta: Asegurar stock y calidad constante. Son los pilares del beneficio neto.</span>
        </div>
        """, unsafe_allow_html=True)

    # 2. FAJA DILEMAS (Minimalista)
    st.markdown(f"""
        <div style="border-left: 5px solid #D7F610; padding: 5px 15px; margin-bottom: 15px;">
            <span style="font-weight: bold; color: #D7F610; font-size: 18px;">🐱 Dilemas</span><br>
            <span style="color: #E0E0E0;">Potencial de crecimiento: Incrementar visibilidad o marketing para escalar volumen de ventas.</span>
        </div>
        """, unsafe_allow_html=True)

    # 3. FAJA VACAS (Minimalista)
    st.markdown(f"""
        <div style="border-left: 5px solid #F6D258; padding: 5px 15px; margin-bottom: 15px;">
            <span style="font-weight: bold; color: #F6D258; font-size: 18px;">🐮 Vacas</span><br>
            <span style="color: #E0E0E0;">Optimización: Revisar costes operativos o ajustar precios para defender el margen.</span>
        </div>
        """, unsafe_allow_html=True)

    # 4. FAJA PERROS (Minimalista)
    st.markdown(f"""
        <div style="border-left: 5px solid #F33939; padding: 5px 15px; margin-bottom: 15px;">
            <span style="font-weight: bold; color: #F33939; font-size: 18px;">🐶 Perros</span><br>
            <span style="color: #E0E0E0;">Revisión crítica: Analizar descontinuación o sustitución por artículos de mayor rotación.</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()


st.header("III. Optimizador de Margen (Simulador Estratégico)")
# 1. Selección de Producto para el Simulador
prod_sim = st.selectbox("Seleccione un producto para simular:",
                        df_margen_raw['Nombre Artículo'].unique())
datos_orig = df_margen_raw[df_margen_raw['Nombre Artículo']
                           == prod_sim].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    # --- NUEVA PALANCA: Coste de Producción ---
    coste_base = float(datos_orig['Coste Producción Unitario'])
    # Permitimos simular variaciones de coste (desde un 50% menos hasta un 50% más)
    nuevo_coste_u = st.slider(
        "Ajustar Coste Producción (€)",
        round(coste_base * 0.5, 3),
        round(coste_base * 1.5, 3),
        coste_base,
        step=0.005,
        format="%.3f"
    )
    st.caption(f"Coste base original: {coste_base:.3f}€")

with col2:
    # Palanca 2: El Multiplicador de Precio
    nuevo_mult = st.slider("Ajustar Multiplicador (Precio)", 1.0, 15.0, float(
        datos_orig['Multiplicador']), step=0.1)
    # El precio bruto ahora depende del NUEVO coste
    nueva_venta_bruta = nuevo_coste_u * nuevo_mult

with col3:
    # Palanca 3: El Descuento
    nuevo_dto = st.slider("Ajustar Descuento (%)", 0, 100, int(
        datos_orig['Descuento Porcentual'] * 100)) / 100
    nueva_venta_neta = nueva_venta_bruta * (1 - nuevo_dto)

# --- RECALCULAR CON EL NUEVO COSTE ---
nuevo_margen_euro = nueva_venta_neta - nuevo_coste_u
nuevo_margen_porc = (nuevo_margen_euro /
                     nueva_venta_neta) if nueva_venta_neta > 0 else 0


# 3. Mostrar Resultado del Impacto
st.subheader("Resultado de la Simulación")
res1, res2, res3 = st.columns(3)

# Color del indicador
color_margin = "normal" if nuevo_margen_porc >= 0.3 else "inverse"

res1.metric("PVP Final Sugerido", f"{nueva_venta_neta:.3f} €",
            delta=f"{nueva_venta_neta - datos_orig['Venta Neta Unitario']:.3f} €")

res2.metric("Nuevo Margen %", f"{nuevo_margen_porc:.1%}",
            delta=f"{(nuevo_margen_porc - datos_orig['Margen Bruto Unitario']):.1%}",
            delta_color=color_margin)

res3.metric("Margen por Unidad (€)", f"{nuevo_margen_euro:.3f} €")


st.divider()

# Cargar los datos
df_rec = pd.read_excel('recomendation.xlsx')

st.header("IV. Estrategias de Venta Cruzada (Smart Bundles)")

# 1. Selector de producto (usa la columna Producto_1 de tu Excel)
producto_sel = st.selectbox("Si el cliente compra:",
                            df_rec['Producto_1'].unique())

# 2. Filtrado (cogemos los mejores del Excel)
resultado = df_rec[df_rec['Producto_1'] == producto_sel].sort_values(
    'Correlacion', ascending=False).head(3)

st.markdown(f"### Recomendaciones para aumentar el ticket")

if not resultado.empty:
    cols = st.columns(len(resultado))
    for i, (index, row) in enumerate(resultado.iterrows()):
        with cols[i]:
            with st.container(border=True):
                st.caption("✅ Producto Sugerido")
                st.subheader(row['Producto_2'])
                # Barra de probabilidad basada en la columna Correlacion
                st.progress(float(row['Correlacion']),
                            text=f"Aceptación: {row['Correlacion']:.1%}")
                st.markdown("**Argumento de Venta:**")
                st.info(f"{row['Recomendacion_Comercial']}")
else:
    st.warning("No hay asociaciones detectadas para este producto en el Excel.")

st.divider()

st.header("V. Smart Bakery App: Motor de Inteligencia Predictiva")

st.markdown("""
    Para garantizar que el modelo de Machine Learning aporte el máximo valor operativo, he aplicado un **filtro de viabilidad**:
    * **Selección Estratégica:** Se han analizado productos con un **Margen > 30%**.

""")


# ... (Selector de Producto y Métricas de Confianza) ...

# ... (Gráfico Predictivo) ...

# --- FASE 3: INTELIGENCIA PREDICTIVA (OPTIMIZADA PARA RANDOM FOREST) ---
try:
    st.markdown("### Motor de Proyección de Demanda")

    df_ml = df_final.copy()
    df_ml['Fecha'] = pd.to_datetime(df_ml['Fecha'])

    df_mensal = df_ml.groupby(
        [pd.Grouper(key='Fecha', freq='MS'),
         'Código artículo', 'Nombre Artículo']
    )['Cantidad_Unidades'].sum().reset_index().sort_values('Fecha')

    productos_dict = dict(
        zip(df_mensal['Código artículo'], df_mensal['Nombre Artículo']))

    top_ids = df_mensal.groupby('Código artículo')[
        'Cantidad_Unidades'].sum().nlargest(56).index.tolist()

    if not top_ids:
        st.warning("No hay datos suficientes para el entrenamiento.")
    else:
        id_sel = st.selectbox("Seleccione un producto para proyectar:", top_ids,
                              format_func=lambda x: f"{x} - {productos_dict[x]}")

        df_prod_ml = df_mensal[df_mensal['Código artículo'] == id_sel].copy()

        # JOAN
        for i in [1, 2, 3]:
            df_prod_ml[f'Lag_{i}'] = df_prod_ml['Cantidad_Unidades'].shift(i)

        df_prod_ml['Target'] = df_prod_ml['Cantidad_Unidades'].shift(-1)
        df_train_all = df_prod_ml.dropna()

        if len(df_train_all) < 5:
            st.error(
                "⚠️ Datos insuficientes para generar una predicción fiable en este artículo.")
        else:
            features = ['Lag_1', 'Lag_2', 'Lag_3']
            X_p, y_p = df_train_all[features], df_train_all['Target']

            # JOAN

            tscv = TimeSeriesSplit(n_splits=min(3, len(X_p)-1))

            best_mae, best_params = np.inf, {}
            grid_rf = [
                {'n_estimators': 600, 'max_depth': 6},
                {'n_estimators': 700, 'max_depth': 8},
                {'n_estimators': 800, 'max_depth': 9}
            ]

            for params in grid_rf:
                maes = []
                for train_idx, test_idx in tscv.split(X_p):
                    m = RandomForestRegressor(
                        **params, random_state=42).fit(X_p.iloc[train_idx], y_p.iloc[train_idx])
                    pred_v = m.predict(X_p.iloc[test_idx])
                    maes.append(mean_absolute_error(
                        y_p.iloc[test_idx], pred_v))

                avg_mae = np.mean(maes)
                if avg_mae < best_mae:
                    best_mae, best_params = avg_mae, params

            final_model = RandomForestRegressor(
                **best_params, random_state=42).fit(X_p, y_p)
            y_pred_h = final_model.predict(X_p)
            r2_p = r2_score(y_p, y_pred_h)

            st.subheader(f"Predicción para: {productos_dict[id_sel]}")
            m1, m2, m3, m4, m5, m6 = st.columns(6)

           # JOAN
            m1.metric(
                label="R² Score",
                value=f"{r2_p:.2%}",
                help="Indica cuánto de la variación de las ventas explica el modelo. > 70% es excelente, < 50% sugiere que las ventas son muy erráticas."
            )

            # 2. Stability
            m2.metric(
                label="Stability (Estabilidad)",
                value="Alta" if r2_p > 0.8 else "Media",
                help="Mide la fiabilidad del algoritmo ante nuevos datos. 'Alta' significa que el modelo es robusto para la toma de decisiones."
            )
            # JOAN

            best_mae = mean_absolute_error(y_p, y_pred_h)

            # 3. MAE
            m3.metric(
                label="MAE (Error Medio)",
                value=f"{best_mae:.0f} unidades",
                help="Error absoluto promedio. Si es 50 unidades, significa que la predicción suele fallar por unas 50 unidades arriba o abajo."
            )

            # 4. Max Depth
            m4.metric(
                label="Max Depth (Profundidad)",
                value=f"{best_params['max_depth']}",
                help="Niveles de los árboles. 5-8 es equilibrado y > 15 puede causar 'overfitting' (aprender de memoria el pasado en lugar de predecir)."
            )

            # 5. Estimators
            m5.metric(
                label="Estimators (Árboles)",
                value=f"{best_params['n_estimators']}",
                help="Cantidad de árboles en el bosque. 100 es el estándar; 200-300 da más estabilidad pero es más lento. Menos de 50 es poco fiable."
            )
            # Cálculo del MAPE (Error Porcentual)

            # 1. Calculamos la suma de los errores absolutos
            suma_error_absoluto = np.abs(y_p - y_pred_h).sum()

            # 2. Calculamos la suma de las ventas reales
            suma_ventas_reales = y_p.sum()

            # 3. Calculamos el WAPE (evitando división por cero)
            wape = (suma_error_absoluto /
                    suma_ventas_reales) if suma_ventas_reales != 0 else 0

            # Ahora lo añadimos a una de tus métricas (por ejemplo, en m3 junto al MAE o sustituyéndolo)
            m6.metric(
                label="WAPE (Error Global)",
                value=f"{wape:.1%}",
                help="Error ponderado por volumen. Es la métrica estándar en logística: mide cuánto fallamos sobre el total de kilos/unidades vendidos."
            )
            # JOAN

            # from sklearn.model_selection import cross_val_score
            # r2_cv = cross_val_score(
            #     final_model, X_p, y_p, cv=5, scoring='r2').mean()

            ultimo_dato = df_prod_ml.iloc[-1]
            futuro = pd.date_range(
                start=ultimo_dato['Fecha'] + pd.DateOffset(months=1), periods=6, freq='ME')
            std_error = np.std(y_p - y_pred_h)

            proyecciones = [{
                'Fecha': ultimo_dato['Fecha'],
                'Cantidad_Unidades': ultimo_dato['Cantidad_Unidades'],
                'Upper': ultimo_dato['Cantidad_Unidades'],
                'Lower': ultimo_dato['Cantidad_Unidades']
            }]

            lags_iter = [ultimo_dato['Cantidad_Unidades'],
                         ultimo_dato['Lag_1'], ultimo_dato['Lag_2']]

            for mes in futuro:
                prep_data = np.array(lags_iter).reshape(1, -1)
                pred = final_model.predict(prep_data)[0]
                proyecciones.append({
                    'Fecha': mes,
                    'Cantidad_Unidades': pred,
                    'Upper': pred + std_error,
                    'Lower': max(pred - std_error, 0)
                })
                lags_iter = [pred] + lags_iter[:2]

            df_proj = pd.DataFrame(proyecciones)

            fig = go.Figure()
            # --- RECUPERADO: SOMBREADO POR AÑOS (SHADOWING) ---
            anos = [2023, 2024, 2025, 2026]
            for ano in anos:
                # alternamos colores muy sutiles para diferenciar los años
                color_faja = "rgba(100, 149, 237, 0.05)" if ano % 2 == 0 else "rgba(255, 255, 255, 0.02)"

                fig.add_vrect(
                    x0=f"{ano}-01-01",
                    x1=f"{ano}-12-31",
                    fillcolor=color_faja,
                    layer="below",
                    line_width=0,
                    annotation_text=str(ano),
                    annotation_position="top left",
                    annotation_font=dict(
                        size=12, color="rgba(255,255,255,0.4)")
                )
            # --- NUEVO: SOMBREADO POR AÑOS DISCRETO (SHADOWING) ---
            anos = [2023, 2024, 2025, 2026]
            for ano in anos:

                color_faja = "rgba(100, 149, 237, 0.03)" if ano % 2 == 0 else "rgba(255, 255, 255, 0.02)"

                fig.add_vrect(
                    x0=f"{ano}-01-01", x1=f"{ano}-12-31",
                    fillcolor=color_faja,
                    layer="below",
                    line_width=0,
                    annotation_text=str(ano),
                    annotation_position="top left",
                    annotation_font=dict(
                        size=10, color="rgba(255,255,255,0.3)")
                )

            fig.add_trace(go.Scatter(
                x=df_prod_ml['Fecha'],
                y=df_prod_ml['Cantidad_Unidades'],
                name='Histórico',
                line=dict(color="#10BCF6", width=3),
                hovertemplate='%{y:.0f} unidades<extra>Histórico</extra>'
            ))

            # Proyección
            fig.add_trace(go.Scatter(
                x=df_proj['Fecha'],
                y=df_proj['Cantidad_Unidades'],
                name='Proyección RF',
                line=dict(color='#2EC18E', dash='dash', width=3),
                hovertemplate='%{y:.0f} unidades<extra>Predicción</extra>'
            ))

            # Área de Confianza de la Proyección (Sombreado verde muy suave)
            fig.add_trace(go.Scatter(
                x=pd.concat([df_proj['Fecha'], df_proj['Fecha'][::-1]]),
                y=pd.concat([df_proj['Upper'], df_proj['Lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(46,193,142,0.08)',  # Un poco más discreto aún
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.update_layout(
                title=f"Tendencia Predictiva: {productos_dict[id_sel]}",
                template="plotly_dark",
                height=550,
                hovermode="x unified",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1)
            )

            fig.update_xaxes(
                tickformat="%b %Y",
                hoverformat="%B %Y",
                showgrid=False  # Quitamos las líneas de cuadrícula para que luzca el sombreado
            )

            fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")

            st.plotly_chart(fig, width='stretch')

            with st.expander(f"Ver desglose de previsiones"):
                df_futuro_solo = df_proj.iloc[1:].copy()
                df_futuro_solo['Mes'] = df_futuro_solo['Fecha'].dt.strftime(
                    '%B %Y')
                df_futuro_solo['Previsto (Media)'] = df_futuro_solo['Cantidad_Unidades'].round(
                    0).astype(int)

                st.dataframe(
                    df_futuro_solo[['Mes', 'Previsto (Media)']],
                    hide_index=True,
                    width='content'
                )

except Exception as e:
    st.error(f"Error en el motor predictivo: {e}")

# --- SECCIÓN FINAL: CONCLUSIONES ESTRATÉGICAS ---


# --- CONFIGURACIÓN DE RUTAS SEGURAS ---
# Esto detecta automáticamente dónde está tu script y busca los archivos allí mismo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_FEEDBACK = os.path.join(BASE_DIR, "registro_feedback_smart_bakery.csv")


def registrar_feedback_local(id_prod, nombre_prod, pred, real, valoracion):
    try:
        ahora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nuevo_fb = pd.DataFrame([{
            "Timestamp": ahora,
            "ID_Articulo": id_prod,
            "Nombre": nombre_prod,
            "Prediccion_IA": round(pred, 2),
            "Venta_Real": real,
            "Feedback_Usuario": valoracion
        }])

        # Guardar usando la ruta segura PATH_FEEDBACK
        if not os.path.isfile(PATH_FEEDBACK):
            nuevo_fb.to_csv(PATH_FEEDBACK, index=False, sep=";")
        else:
            nuevo_fb.to_csv(PATH_FEEDBACK, mode='a',
                            index=False, sep=";", header=False)
        st.toast(f"Feedback '{valoracion}' guardado!", icon="💾")
    except Exception as e:
        st.error(f"Error al guardar: {e}")


# ==========================================
# 1️⃣ DICCIONARIS DE VENDES REALS
# ==========================================
ventas_reales_enero = {
    "101": 993, "131": 1470, "120": 633, "6": 400, "105": 301, "30": 533, "3": 310, "1121": 1176,
    "113": 332, "9": 486, "300": 334, "1123": 251, "103": 318, "303": 191, "330": 524, "1144": 229,
    "1141": 436, "124": 237, "1010": 222, "7": 124, "1122": 221, "4": 54, "36": 81, "31": 77,
    "1011": 171, "0000": 99, "402": 176, "410": 118, "1604": 215, "35": 55, "403": 182, "17": 18,
    "309": 85, "1310": 96, "140": 65, "404": 171, "138": 124, "161": 100, "331": 89, "1125": 49,
    "1001": 186, "405": 85, "1802": 152, "1013": 92, "2027": 44, "106": 94, "8": 134, "1134": 561,
    "107": 70, "401": 47, "25": 41, "142": 83, "1605": 94, "110": 64, "19": 35, "22": 53,
    "1309": 56, "1021": 63, "1016": 49, "321": 67, "1132": 128, "1114": 52, "818": 25, "604": 1,
    "1014": 37, "118": 25, "37": 28, "515": 20, "156": 41, "817": 5, "38": 25, "23": 11,
    "15": 34, "1308": 50, "1018": 32, "200": 53, "137": 59, "815": 9, "1112": 42, "100": 14,
    "308": 32, "332": 16, "210": 31, "307": 48, "700": 17, "181": 17, "114": 16, "1501": 39,
    "521": 11, "1314": 20, "314": 17, "108": 31, "517": 3, "202": 26, "333": 8, "104": 13,
    "516": 4, "201": 12, "208": 12, "518": 17, "313": 15, "1315": 15, "1316": 7, "816": 4,
    "522": 4, "502": 2, "1509": 22, "1508": 28, "820": 1, "1505": 6, "1500": 11, "1504": 4,
    "150": 9, "1313": 3, "1494": 4, "1507": 5, "315": 2, "211": 9, "350": 119, "351": 37,
    "334": 6, "326": 16, "324": 93, "323": 14, "327": 20, "0325": 91, "322": 69, "514": 2,
    "17": 1, "1606": -1, "IN400": 1, "26": 38, "28": 157
}

ventas_reales_febrero = {
    "131": 1716, "1121": 1356, "330": 1292, "0": 1077, "101": 1062, "120": 648, "1141": 618,
    "9": 582, "30": 515, "5136": 480, "113": 412, "6": 398, "331": 342, "1134": 332, "300": 317,
    "103": 315, "3": 312, "105": 306, "1123": 284, "1604": 269, "124": 265, "1001": 261,
    "1010": 229, "1144": 222, "1122": 217, "303": 189, "402": 172, "1011": 164, "8": 157,
    "1802": 157, "29": 156, "403": 145, "7": 143, "1605": 143, "404": 127, "1132": 127,
    "350": 114, "138": 113, "332": 110, "324": 104, "1013": 102, "161": 101, "0000": 100,
    "1310": 100, "405": 91, "28": 89, "36": 85, "106": 82, "142": 81, "309": 79, "325": 79,
    "107": 74, "110": 74, "321": 74, "1021": 73, "31": 72, "140": 72, "410": 68, "518": 67,
    "1125": 59, "156": 58, "22": 56, "1016": 56, "18": 55, "1308": 54, "351": 53, "1309": 53,
    "26": 50, "333": 50, "137": 49, "35": 47, "4": 46, "200": 46, "1501": 46, "308": 42,
    "2027": 40, "307": 39, "401": 39, "605": 39, "210": 38, "25": 36, "1114": 34, "17": 33,
    "19": 33, "108": 33, "37": 32, "1018": 31, "322": 30, "334": 30, "15": 29, "38": 29,
    "1014": 27, "100": 25, "208": 25, "181": 23, "604": 23, "700": 23, "326": 21, "314": 20,
    "1112": 19, "114": 18, "327": 18, "1314": 18, "118": 17, "323": 17, "211": 16, "104": 14,
    "313": 13, "361": 13, "515": 13, "12": 12, "150": 11, "362": 10, "1509": 10, "23": 9,
    "201": 9, "202": 8, "517": 8, "521": 8, "1316": 8, "1500": 8, "1508": 8, "315": 6,
    "522": 6, "516": 5, "1315": 5, "502": 3, "519": 3, "1505": 3, "1507": 3, "14": 2,
    "514": 2, "523": 2, "1504": 2, "IN400": 2, "406": 1, "535": 1, "1313": 1, "1494": 1, "IN409": 1
}

# ==========================================
# 2️⃣ SELECTOR DE MES (CORREGIDO)
# ==========================================
opciones_mes = ["Enero de 2026", "Febrero de 2026"]
mes_validacion = st.selectbox("Selecciona mes para validar:", opciones_mes)

# Lógica de asignación limpia
if mes_validacion == "Enero de 2026":
    diccionario_actual = ventas_reales_enero
    nombre_mes_real = "Venta Real Enero 2026"
    fila_idx = 0  # Primera fila de la predicción
else:
    diccionario_actual = ventas_reales_febrero
    nombre_mes_real = "Venta Real Febrero 2026"
    fila_idx = 1  # Segunda fila de la predicción

st.subheader(f"Validación Real: {mes_validacion}")

# ==========================================
# 3️⃣ LIMPIEZA DE CÓDIGO (SENSE 0 INICIAL)
# ==========================================
try:
    # Convertimos a string, quitamos espacios y ceros a la izquierda
    id_limpio = str(id_sel).strip().lstrip('0')
    if id_limpio == "":
        id_limpio = "0"
except:
    st.error("Error leyendo el código seleccionado.")
    st.stop()

# ==========================================
# 4️⃣ VALIDACIÓN Y COMPARATIVA
# ==========================================
if id_limpio in diccionario_actual:
    # Verificamos que df_proj exista y tenga datos
    if 'df_proj' in locals() and not df_proj.empty:
        try:
            # Intentamos sacar la predicción de la fila correspondiente
            pred_ia = float(df_proj["Cantidad_Unidades"].iloc[fila_idx])
            real_dato = diccionario_actual[id_limpio]

            diferencia = pred_ia - real_dato
            porcentaje_brecha = (diferencia / real_dato) * \
                100 if real_dato != 0 else 0

            # --- VISUALIZACIÓN ---
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Predicción IA", f"{pred_ia:.0f} unidades")
            with c2:
                st.metric(nombre_mes_real, f"{real_dato:.0f} unidades")
            with c3:
                # Delta inverso: si la IA predice de MÁS, es un aviso (rojo), si es de MENOS (verde/normal)
                st.metric("Desviación (Error)",
                          f"{diferencia:.0f} unidades",
                          delta=f"{porcentaje_brecha:.1f}%",
                          delta_color="inverse" if abs(porcentaje_brecha) > 15 else "normal")

            # Tabla resumen elegante
            df_comp = pd.DataFrame([{
                "Producto": productos_dict[id_sel],
                "Mes": mes_validacion,
                "Predicción": f"{pred_ia:.0f}",
                "Real": f"{real_dato}",
                "Diferencia": f"{diferencia:.0f}",
                "Precisión": f"{100 - abs(porcentaje_brecha):.1f}%"
            }])
            st.table(df_comp)

        except Exception as e:
            st.warning(
                f"Nota: No hay fila de predicción suficiente en el modelo para {mes_validacion}.")
    else:
        st.warning("Primero genera la proyección en la sección anterior.")
else:
    st.info(
        f"El producto {id_limpio} no tuvo ventas reales registradas en {mes_validacion}.")

# --- SECCIÓN: ACCESO A DATOS (RAW DATA) ---
st.markdown("---")
st.header("VI. Centro de Datos (Raw Data) y Información Clave para la IA")

with st.expander("Inspeccionar tablas de análisis y entrenamiento"):
    tab1, tab2, tab3 = st.tabs([
        "Histórico de Ventas",
        "Dataset Entrenamiento (Lags)",
        "Resumen por Producto"
    ])

    with tab1:
        st.subheader("Histórico Completo de Ventas")
        # Preparamos una copia para visualización
        df_vis_ventas = df_final.copy()

        # Limpiamos nombres y formateamos fecha
        df_vis_ventas.columns = df_vis_ventas.columns.str.strip()
        if 'Fecha' in df_vis_ventas.columns:
            df_vis_ventas['Fecha'] = pd.to_datetime(
                df_vis_ventas['Fecha']).dt.strftime('%d-%m-%Y')

        # Mostramos solo columnas clave para que no se vea desordenado
        cols_interes = ['Fecha', 'Código artículo',
                        'Nombre Artículo', 'Cantidad_Unidades', 'Importe_Euros']
        cols_reales = [c for c in cols_interes if c in df_vis_ventas.columns]

        st.dataframe(df_vis_ventas[cols_reales],
                     width='content', hide_index=True)

    with tab2:
        st.subheader("Datos Procesados para Random Forest")
        st.write(
            "Esta tabla muestra los **Lags** (ventas de meses anteriores) que la IA usa para aprender:")

        if 'df_train_all' in locals() or 'df_train_all' in globals():
            df_vis_train = df_train_all.copy()
            # Formateamos la fecha para que se vea el mes y el año claramente
            df_vis_train['Fecha'] = pd.to_datetime(
                df_vis_train['Fecha']).dt.strftime('%B %Y')

            st.dataframe(df_vis_train, width='content',
                         hide_index=True)
        else:
            st.warning(
                "Selecciona un producto arriba para generar los datos de entrenamiento.")

    with tab3:
        st.subheader("Consolidado de Métricas")
        if 'df_resumen' in locals() or 'df_resumen' in globals():
            st.dataframe(df_resumen, width='content', hide_index=True)
        else:
            st.info("El resumen se generará al procesar todos los productos.")

# --- TEXTO EXPLICATIVO DEBAJO DE LA TABLA DE DATOS ---
st.info("###### Inteligencia de Datos: ¿Cómo lee la IA esta tabla?")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.markdown("""
    **¿Qué son los Lags (Retardos)?**
    Son la **memoria** del modelo. Para predecir el futuro, la IA no mira una bola de cristal, sino que analiza:
    * **Lag_1:** Lo que vendiste el mes pasado.
    * **Lag_2 y Lag_3:** La tendencia de los meses previos.
    
    Esta "foto" histórica permite al algoritmo entender si las ventas están subiendo o bajando.
    """)

with col_exp2:
    st.markdown("""
    **¿Cómo deciden los Árboles?**
    Cada dato de la tabla pasa por cientos de **árboles de decisión**. 
    * El modelo se pregunta: *"Si el mes pasado vendimos X, y hace dos meses Y... ¿cuánto toca hoy?"*.
    * Al limitar la **Profundidad (Max Depth)**, obligamos a la IA a aprender patrones generales y no errores del pasado, garantizando esa **Robustez del 80%**.
    """)


st.divider()


# ==========================================
# 1. RECUPERAR LOS DATOS DE LA FUNCIÓN
# ==========================================
# Asegúrate de que este orden coincide exactamente con tu 'return'
df_facturacion_zonas, df_final, df_margen_raw, df_a = load_data()

# ==========================================
# 2. LÓGICA DEL CAMIÓN INTELIGENTE
# ==========================================
st.header("VII. Módulo de Logística Inteligente")

st.markdown("""
### ¿Es rentable enviar el camión hoy?
Este sistema calcula si el **beneficio neto** de los pedidos de una zona 
cubre los costes de envío (gasoil, chófer y mantenimiento).
""")

# Calculamos el margen promedio de la fábrica para saber cuánto ganamos por euro vendido
# Usamos la columna que ya limpiamos en tu función
margen_medio = df_margen_raw['Margen Bruto Unitario'].mean()

# Agrupamos la facturación por zona
resumen_logistica = df_facturacion_zonas.groupby('Zona').agg({
    'Importe_Euros': 'sum'
}).reset_index()

# Calculamos el beneficio estimado (Ventas * Margen)
resumen_logistica['Beneficio_Estimado'] = resumen_logistica['Importe_Euros'] * margen_medio

# --- INTERFAZ PARA --
col1, col2 = st.columns([1, 2])

with col1:
    # Slider  ajuste el coste real de mover un camión
    coste_camion = st.slider(
        "Coste fijo por ruta (€):",
        min_value=50,
        max_value=500,
        value=150,
        help="Gasoil + Chófer + Amortización del camión"
    )

# Calculamos el ROI (Dinero que queda tras pagar el camión)
resumen_logistica['Resultado_Neto'] = resumen_logistica['Beneficio_Estimado'] - coste_camion

# Función del semáforo para la decisión automática


def alerta_logistica(resultado):
    if resultado > 200:
        return "🟢 Ruta rentable"
    if resultado > 0:
        return "🟡 Margen justo"
    return "🔴 Agrupar pedido"


resumen_logistica['Decisión IA'] = resumen_logistica['Resultado_Neto'].apply(
    alerta_logistica)

# Ordenamos para que vea primero donde pierde dinero (lo que hay que automatizar)
resumen_logistica = resumen_logistica.sort_values(
    by='Resultado_Neto', ascending=True)

# --- MOSTRAR TABLA ---
st.dataframe(
    resumen_logistica[['Zona', 'Importe_Euros',
                       'Beneficio_Estimado', 'Resultado_Neto', 'Decisión IA']],
    use_container_width=True,
    hide_index=True
)


st.divider()

col_c1, col_c2 = st.columns([3, 1])

with col_c1:
    st.caption(
        "© 2026 **Smart Bakery Solutions** | Industrial Digital Transformation")
    st.caption("Developed by **José**  -  Data Specialist Student")

with col_c2:
    st.caption("v1.0.4-stable 🚀")
