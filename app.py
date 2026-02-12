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
    page_title="🥐 Dashboard Rentabilidad Alberto", layout="wide")

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
st.header("I. Diagnóstico de Salud de Cartera y Pulso de Mercado")

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

st.plotly_chart(fig_line, use_container_width=True)

# --- CÁLCULOS PARA KPIs DE CLIENTES ---

st.divider()
st.header("Análisis de la Estructura de Ventas")

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

# --- KPI ---
# Puedes ponerlo en una nueva fila de columnas
# ik1, ik2 = st.columns(2)

# ik1.metric(
#     label="Dependencia VIP (Pareto)",
#     value=f"{pareto_val:.1%}",
#     delta="Top 20% pedidos",
#     help="Indica qué porcentaje de tus ingresos totales provienen del 20% de los pedidos más caros."
# )

# ik2.metric(
#     label="Ticket Máximo Normal",
#     value=f"{limite_superior:,.0f} €",
#     delta="Excluyendo el 10% VIP",
#     help="El techo de gasto de un cliente estándar antes de entrar en la categoría de pedidos excepcionales."
# )


# --- TU GRÁFICO (FILTRADO Y CON MEDIANA) ---
# --- HISTOGRAMA + BOXPLOT (ANÁLISIS DE OUTLIERS) ---
# --- HISTOGRAMA + BOXPLOT: VERSIÓN FINAL CON LEYENDA ---


# 1. Preparación de datos
df_hist_pos = df_hist[df_hist['Importe_Euros'] > 0]
mediana_ticket = df_hist_pos['Importe_Euros'].median()

# 2. Crear gráfico
fig_dist = px.histogram(
    df_hist_pos,
    x="Importe_Euros",
    nbins=250,
    marginal="box",
    title="Análisis de Gasto: Distribución y Outliers",
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

st.plotly_chart(fig_dist, use_container_width=True)

st.info("""
    **Análisis Estratégico:**
    * **La Mediana (Línea Roja):** Es el punto exacto donde se divide tu clientela. El 50% de tus pedidos están por debajo de este valor y el otro 50% por encima.
    * **Impacto en el Desperdicio:** Entender el ticket típico nos ayuda a prever cuántas unidades de cada producto se llevan en promedio. Si el ticket mediano sube pero las unidades bajan, significa que vendemos productos más caros (más margen, menos merma logística).
""")

st.divider()
st.header("Retención y Fidelidad")


clientes_2025 = df_areas[df_areas['Euros 2025'] > 0]
num_clientes_25 = len(clientes_2025)

# Clientes que compraron en 2024
clientes_2024 = df_areas[df_areas['Euros 2024'] > 0]
num_clientes_24 = len(clientes_2024)

# Tasa de Retención: (Compraron en 24 Y en 25) / Compraron en 24
recurrentes = df_areas[(df_areas['Euros 2024'] > 0) &
                       (df_areas['Euros 2025'] > 0)]
tasa_retencion = (len(recurrentes) / num_clientes_24 *
                  100) if num_clientes_24 > 0 else 0

# Clientes Nuevos 2025: (Compraron en 25 pero no en 24 ni 23)
nuevos_25 = df_areas[(df_areas['Euros 2025'] > 0) & (
    df_areas['Euros 2024'] <= 0) & (df_areas['Euros 2023'] <= 0)]
num_nuevos = len(nuevos_25)

# Ticket Medio Histórico 2025 (Basado en df_areas)
ticket_medio_25 = clientes_2025['Euros 2025'].mean(
) if num_clientes_25 > 0 else 0

# --- BLOQUE NUEVO: KPIs DE CLIENTES ---

# st.markdown("######     Indicadores Clave de Rendimiento")
k1, k2, k3 = st.columns(3)

# KPI 1: Tasa de Retención
k1.metric(
    label="Tasa de Retención",
    value=f"{tasa_retencion:.1f}%",
    delta=f"{len(recurrentes)} Clientes Fieles",
    help="Porcentaje de clientes de 2024 que han vuelto a comprar en 2025."
)

# KPI 2: Captación de Nuevos
k2.metric(
    label="Nuevos Clientes 2025",
    value=num_nuevos,
    delta=f"{(num_nuevos/num_clientes_25*100):.1f}% del total" if num_clientes_25 > 0 else "0%",
    delta_color="normal"
)

# KPI 3: Valor Medio por Cliente (Anual)
# k3.metric(
#     label="Inversión Media Anual",
#     value=f"{ticket_medio_25:,.2f} €",
#     delta="Gasto por cliente/año"
# )

# KPI 4: Clientes Fantasma (Base Inactiva)
fantasmas = len(df_areas[df_areas.apply(
    lambda x: x['Euros 2023']+x['Euros 2024']+x['Euros 2025'] <= 0, axis=1)])
k3.metric(
    label="Clientes Inactivos",
    value=f"{fantasmas} clientes",
    delta="Oportunidad perdida",
    delta_color="inverse"
)


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

# col1, col2 = st.columns([1, 1.2])

# with col1:
#     conteo = df_fidelidad['Segmento'].value_counts().reset_index()
#     fig_pie = px.pie(conteo, values='count', names='Segmento',
#                      hole=0.7, color='Segmento', color_discrete_map=mapa_colores)

#     fig_pie.update_layout(
#         title=dict(
#             text="Volumen de Clientes",
#             x=0.05,               # <--- Posición cerca del borde izquierdo
#             xanchor='left',       # <--- El punto de anclaje es el inicio del texto
#             font=dict(size=15)    # Opcional: para que resalte más como título
#         ),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=-0.4,
#             xanchor="center",
#             x=0.5
#         ),
#         # Aumentamos t (top) para que el título no se pegue al gráfico
#         margin=dict(t=80, b=100, l=0)
#     )

#     st.plotly_chart(fig_pie, width='stretch')


# with col2:
#     # Agrupar por zona y sumar ventas
#     df_zona = df_filtrado.groupby('Zona')['Importe_Euros'].sum().reset_index()

#     # Crear gráfico de barras
#     fig_bar = px.bar(
#         df_zona,
#         x='Zona',
#         y='Importe_Euros',
#         color='Importe_Euros',
#         title='Facturación por Zona',
#         color_continuous_scale='Tealgrn',
#         labels={
#             "Zona": "Región",          # nombre del eje X
#             "Importe_Euros": "Facturación (€)"  # nombre del eje Y
#         }
#     )

#     # Quitar grid y limpiar fondo
#     fig_bar.update_layout(
#         xaxis=dict(showgrid=False),
#         yaxis=dict(showgrid=False),
#         # fondo de todo el canvas
#     )

#     # Mostrar en Streamlit
#     st.plotly_chart(fig_bar, width='stretch')

# --- FASE 2: RENTABILIDAD ---
st.divider()
# --- SECCIÓN 2: OPTIMIZACIÓN DEL MARGEN Y RENTABILIDAD ---
st.header("II. Matriz de Decisión")
st.markdown("""
    ###### Solo se han analizado productos con un **Margen > 30%**.
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
st.plotly_chart(fig_bubble, use_container_width=True)

# --- 3. ANÁLISIS PARA TU JEFE ---
# --- TU GRÁFICO (con los emojis ya configurados) ---
# ... (código del scatter que ya tienes) ...

# --- ESTILO DE FAJAS DIRECTAS ---
# --- PLAN DE ACCIÓN DISCRETO EN EXPANDER ---

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

with st.expander("Listado Detallado"):
    col_ganadores, col_revision = st.columns(2)
    with col_ganadores:
        st.success(" Generadores de Valor")
        df_gan = df_final.groupby(['Código artículo', 'Nombre Artículo']).agg(
            {'Beneficio_Real_Euros': 'sum', 'Margen Bruto Unitario': 'mean'}).nlargest(10, 'Beneficio_Real_Euros').reset_index()
        st.dataframe(df_gan, width='stretch', hide_index=True)
    with col_revision:
        st.error(" En Revisión")
        df_rev = df_final.groupby(['Código artículo', 'Nombre Artículo']).agg(
            {'Margen Bruto Unitario': 'mean', 'Beneficio_Real_Euros': 'sum'}).nsmallest(10, 'Margen Bruto Unitario').reset_index()
        st.dataframe(df_rev, width='stretch', hide_index=True)
st.divider()


st.header("III. Smart Bakery App: Motor de Inteligencia Predictiva")

st.markdown("""
    Para garantizar que el modelo de Machine Learning aporte el máximo valor operativo, he aplicado un **filtro de viabilidad**:
    * **Selección Estratégica:** Solo se han analizado productos con un **Margen > 30%**.

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

            # JOAN

            final_model = RandomForestRegressor(
                **best_params, random_state=42).fit(X_p, y_p)
            y_pred_h = final_model.predict(X_p)
            r2_p = r2_score(y_p, y_pred_h)

            st.subheader(f"Métricas de Confianza: {productos_dict[id_sel]}")
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
            # Evitamos división por cero con un pequeño epsilon

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

            st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"Ver desglose de previsiones"):
                df_futuro_solo = df_proj.iloc[1:].copy()
                df_futuro_solo['Mes'] = df_futuro_solo['Fecha'].dt.strftime(
                    '%B %Y')
                df_futuro_solo['Previsto (Media)'] = df_futuro_solo['Cantidad_Unidades'].round(
                    0).astype(int)

                st.dataframe(
                    df_futuro_solo[['Mes', 'Previsto (Media)']],
                    hide_index=True,
                    use_container_width=True
                )

except Exception as e:
    st.error(f"Error en el motor predictivo: {e}")

# --- SECCIÓN FINAL: CONCLUSIONES ESTRATÉGICAS ---
st.divider()


st.subheader(
    f"Validación Real: ¿Qué pasó en Enero con el producto {productos_dict[id_sel]}?")

# ===============================
# 1️⃣ Diccionario
# ===============================

ventas_reales_enero = {
    "101": 993, "131": 1470, "120": 633, "6": 400,
    "105": 301, "30": 533, "3": 310, "1121": 1176,
    "113": 332, "9": 486, "300": 334, "1123": 251,
    "103": 318, "303": 191, "330": 524, "1144": 229,
    "1141": 436, "124": 237, "1010": 222, "7": 124,
    "1122": 221, "4": 54, "36": 81, "31": 77,
    "1011": 171, "0000": 99, "402": 176, "410": 118,
    "1604": 215, "35": 55, "403": 182, "17": 18,
    "309": 85, "1310": 96, "140": 65, "404": 171,
    "138": 124, "161": 100, "331": 89, "1125": 49,
    "1001": 186, "405": 85, "1802": 152, "1013": 92,
    "2027": 44, "106": 94, "8": 134, "1134": 561,
    "107": 70, "401": 47, "25": 41, "142": 83,
    "1605": 94, "110": 64, "19": 35, "22": 53,
    "1309": 56, "1021": 63, "1016": 49, "321": 67,
    "1132": 128, "1114": 52, "818": 25, "604": 1,
    "1014": 37, "118": 25, "37": 28, "515": 20,
    "156": 41, "817": 5, "38": 25, "23": 11,
    "15": 34, "1308": 50, "1018": 32, "200": 53,
    "137": 59, "815": 9, "1112": 42, "100": 14,
    "308": 32, "332": 16, "210": 31, "307": 48,
    "700": 17, "181": 17, "114": 16, "1501": 39,
    "521": 11, "1314": 20, "314": 17, "108": 31,
    "517": 3, "202": 26, "333": 8, "104": 13,
    "516": 4, "201": 12, "208": 12, "518": 17,
    "313": 15, "1315": 15, "1316": 7, "816": 4,
    "522": 4, "502": 2, "1509": 22, "1508": 28,
    "820": 1, "1505": 6, "1500": 11, "1504": 4,
    "150": 9, "1313": 3, "1494": 4, "1507": 5,
    "315": 2, "211": 9, "350": 119, "351": 37,
    "334": 6, "326": 16, "324": 93, "323": 14,
    "327": 20, "0325": 91, "322": 69, "514": 2,
    "170": 1, "1606": -1, "IN400": 1,
    "26": 38, "28": 157
}

# ===============================
# 2️⃣ Limpiamos ID seleccionado (SIEMPRE STRING)
# ===============================

try:
    id_limpio = str(id_sel).strip()
except:
    st.error("Error leyendo el código seleccionado.")
    st.stop()

# ===============================
# 3️⃣ Validación real
# ===============================
if id_limpio in ventas_reales_enero:

    if 'df_proj' in locals() and len(df_proj) > 1:

        pred_ia = float(df_proj["Cantidad_Unidades"].iloc[1])
        real_alberto = ventas_reales_enero[id_limpio]
        diferencia = pred_ia - real_alberto

        # Cálculo del porcentaje de brecha (evitando división por cero)
        porcentaje_brecha = (diferencia / real_alberto) * \
            100 if real_alberto != 0 else 0

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Predicción Smart Bakery", f"{pred_ia:.0f} unidades")

        with c2:
            st.metric("Venta Real Enero 2026", f"{real_alberto:.0f} unidades")

        with c3:
            # Mostramos el porcentaje en el delta
            st.metric(
                "Brecha de Demanda",
                f"{diferencia:.0f} unidades",
                delta=f"{porcentaje_brecha:.1f}% vs Real",
                delta_color="inverse" if diferencia > 0 else "normal"
            )

        # Tabla resumen con la columna de porcentaje
        st.table(pd.DataFrame([{
            "Código": id_limpio,
            "Demanda IA": f"{pred_ia:.0f}",
            "Venta Real": f"{real_alberto}",
            "Diferencia": f"{diferencia:.0f}",
            "Impacto (%)": f"{porcentaje_brecha:.1f}%"
        }]))

    else:
        st.warning("No hay proyección disponible (df_proj vacío).")
else:
    st.warning(f"El código {id_limpio} no tiene registro en Enero 2026.")

# --- SECCIÓN: ACCESO A DATOS (RAW DATA) ---
st.markdown("---")
st.header("IV. Centro de Datos (Raw Data)")

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
                     use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Datos Procesados para Random Forest")
        st.write(
            "Esta tabla muestra los **Lags** (ventas de meses anteriores) que la IA usa para aprender:")

        if 'df_train_all' in locals() or 'df_train_all' in globals():
            df_vis_train = df_train_all.copy()
            # Formateamos la fecha para que se vea el mes y el año claramente
            df_vis_train['Fecha'] = pd.to_datetime(
                df_vis_train['Fecha']).dt.strftime('%B %Y')

            st.dataframe(df_vis_train, use_container_width=True,
                         hide_index=True)
        else:
            st.warning(
                "Selecciona un producto arriba para generar los datos de entrenamiento.")

    with tab3:
        st.subheader("Consolidado de Métricas")
        if 'df_resumen' in locals() or 'df_resumen' in globals():
            st.dataframe(df_resumen, use_container_width=True, hide_index=True)
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

st.caption("© 2026 Smart Bakery Solutions | Strategic Data Analysis")

st.divider()
st.header("V. Centro de Control")

# El botón de subida
uploaded_file = st.file_uploader(
    "", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Si Alberto sube un archivo, la App lo lee
    if uploaded_file.name.endswith('.csv'):
        df_final = pd.read_csv(uploaded_file)
    else:
        df_final = pd.read_excel(uploaded_file)

    st.success("✅ Datos cargados correctamente. ¡Smart Bakery está listo!")

    # AQUÍ IRÍA EL RESTO DE TU CÓDIGO (EL MODELO, LOS GRÁFICOS, ETC.)
    # ...
else:
    st.info("Por favor, sube un archivo para comenzar el análisis.")


# El Gran Final
if st.button("Finalizar Presentación Estratégica"):
    st.balloons()
    st.snow()  # Un toque extra para que parezca confeti cayendo

# --- ESTO DEBE IR AL FINAL DE TODO, FUERA DE LOS BUCLES ---
st.write("")  # Un espacio en blanco
st.write("")
st.markdown("---")  # Línea divisoria

# Creamos columnas para que quede alineado
col_c1, col_c2 = st.columns([3, 1])

with col_c1:
    st.caption(
        "© 2026 **Smart Bakery Solutions** | Industrial Digital Transformation")
    st.caption("Developed by **José**  -  Data Specialist Student")

with col_c2:
    st.caption("v1.0.4-stable 🚀")


# # Creamos un diccionario con todas las tablas clave de tu proyecto
# tablas_proyecto = {
#     # El Excel de Alberto tal cual
#     "1. Raw Data (Original)": df_final,
#     # Tras quitar nulos y corregir IDs
#     "2. Data Cleaned (Limpio)": df_margen_raw,
#     # La tabla que mostraste con Lag_1, Lag_2...
#     "3. Features (Lags)": df_areas,
#     # Variables de entrada para la IA
#     "5. Predictions (Output)": df_proj            # El resultado final de la IA
# }

# st.header("📋 Auditoría de Estructura de Datos")

# for nombre, df in tablas_proyecto.items():
#     with st.expander(f"Ver estructura de: {nombre}"):
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Filas", df.shape[0])
#         col2.metric("Columnas", df.shape[1])
#         col3.write(f"**Columnas clave:** {', '.join(df.columns[:5])}...")

#         # Mostramos las primeras 5 filas para que se vea el contenido
#         st.dataframe(df.head(5), use_container_width=True)
