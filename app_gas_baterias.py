import io
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Caídas de producción por batería y pozo",
    page_icon="📉",
    layout="wide"
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

PRODUCTOS = {
    "Gas": {
        "columnas_preferidas": [
            "PROD_GAS_REAL",
            "ProdGas Real Cierre",
            "PROD_GAS",
            "PROD_GAS_POT_REAL",
            "PROD_GAS_POT"
        ],
        "columna_perdida": "GasPerdido",
        "unidad": "MPCD"
    },
    "Petróleo": {
        "columnas_preferidas": [
            "PROD_OIL_REAL",
            "ProdReal Cierre",
            "PROD_OIL"
        ],
        "columna_perdida": "PetroleoPerdido",
        "unidad": "BOPD"
    }
}


# ============================================================
# UTILIDADES
# ============================================================

def normalizar_texto(valor):
    if pd.isna(valor):
        return ""
    texto = str(valor).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def normalizar_columna(columna):
    texto = str(columna).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def clave_columna(columna):
    texto = normalizar_columna(columna).lower()
    texto = texto.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = texto.strip("_")
    return texto


def buscar_columna(df, posibles):
    mapa = {clave_columna(c): c for c in df.columns}
    for posible in posibles:
        key = clave_columna(posible)
        if key in mapa:
            return mapa[key]
    return None


def detectar_fila_header(file_bytes, sheet_name=0):
    preview = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=20)

    obligatorias = {"fecha", "pozo", "bateria"}
    for idx in range(len(preview)):
        valores = {clave_columna(x) for x in preview.iloc[idx].tolist() if pd.notna(x)}
        if obligatorias.issubset(valores):
            return idx

    return 0


@st.cache_data(show_spinner=False)
def cargar_excel(file_bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    hoja = xls.sheet_names[0]
    header_row = detectar_fila_header(file_bytes, sheet_name=hoja)

    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=hoja, header=header_row)
    df.columns = [normalizar_columna(c) for c in df.columns]
    df = df.dropna(how="all").copy()

    col_fecha = buscar_columna(df, ["Fecha", "FECHA", "date"])
    col_pozo = buscar_columna(df, ["Pozo", "POZO", "well"])
    col_bateria = buscar_columna(df, ["Bateria", "Batería", "BAT", "battery"])

    if col_fecha is None or col_pozo is None or col_bateria is None:
        raise ValueError("No encontré las columnas obligatorias Fecha, Pozo y Bateria.")

    df = df.rename(columns={
        col_fecha: "Fecha",
        col_pozo: "Pozo",
        col_bateria: "Bateria"
    })

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df[df["Fecha"].notna()].copy()
    df["Fecha"] = df["Fecha"].dt.normalize()

    df["Pozo"] = df["Pozo"].apply(normalizar_texto)
    df["Bateria"] = df["Bateria"].apply(normalizar_texto)
    df = df[(df["Pozo"] != "") & (df["Bateria"] != "")].copy()

    for col in df.columns:
        if col not in ["Fecha", "Pozo", "Bateria", "SubSistema", "SubSist", "TEST_PURP_CD"]:
            convertido = pd.to_numeric(df[col], errors="coerce")
            if convertido.notna().sum() > 0:
                df[col] = convertido

    return df, hoja, header_row + 1


def columnas_disponibles_producto(df, producto):
    preferidas = PRODUCTOS[producto]["columnas_preferidas"]
    disponibles = []

    for col in preferidas:
        real = buscar_columna(df, [col])
        if real is not None and real not in disponibles:
            disponibles.append(real)

    # Respaldo: si el nombre tiene GAS u OIL y es numérico, lo agrega.
    palabra = "gas" if producto == "Gas" else "oil"
    for col in df.columns:
        if palabra in clave_columna(col) and pd.api.types.is_numeric_dtype(df[col]):
            if col not in disponibles:
                disponibles.append(col)

    return disponibles


def obtener_columna_perdida(df, producto):
    perdida = PRODUCTOS[producto]["columna_perdida"]
    return buscar_columna(df, [perdida])


def aplicar_filtros(df, fecha_ini, fecha_fin, baterias_sel, texto_pozo):
    salida = df[(df["Fecha"] >= pd.to_datetime(fecha_ini)) & (df["Fecha"] <= pd.to_datetime(fecha_fin))].copy()

    if baterias_sel:
        salida = salida[salida["Bateria"].isin(baterias_sel)].copy()

    texto_pozo = normalizar_texto(texto_pozo)
    if texto_pozo:
        salida = salida[salida["Pozo"].str.contains(texto_pozo, case=False, na=False, regex=False)].copy()

    return salida


def lista_fechas(fecha_ini, fecha_fin):
    return pd.date_range(pd.to_datetime(fecha_ini), pd.to_datetime(fecha_fin), freq="D")


def diario_lote(df, col_valor, fechas):
    diario = df.groupby("Fecha", as_index=False)[col_valor].sum()
    diario = diario.set_index("Fecha").reindex(fechas, fill_value=0).rename_axis("Fecha").reset_index()
    diario = diario.rename(columns={col_valor: "PRODUCCION"})
    diario["PROM_MOVIL_7D"] = diario["PRODUCCION"].rolling(7, min_periods=1).mean()
    return diario


def promedio_periodo_diario(df, col_valor, fechas):
    if len(fechas) == 0:
        return 0.0
    diario = diario_lote(df, col_valor, fechas)
    return float(diario["PRODUCCION"].mean())


def calcular_ranking_caida(df, grupo_cols: List[str], col_valor: str, col_perdida: str, fecha_ini, fecha_fin, n_dias: int, min_prom_inicial: float):
    fecha_ini = pd.to_datetime(fecha_ini)
    fecha_fin = pd.to_datetime(fecha_fin)

    fechas_rango = lista_fechas(fecha_ini, fecha_fin)
    if len(fechas_rango) == 0:
        return pd.DataFrame(), None

    n_dias = int(max(1, min(n_dias, len(fechas_rango))))
    fechas_inicio = fechas_rango[:n_dias]
    fechas_final = fechas_rango[-n_dias:]

    grupos = df[grupo_cols].drop_duplicates().copy()
    if grupos.empty:
        return pd.DataFrame(), {
            "inicio_ini": fechas_inicio.min(), "inicio_fin": fechas_inicio.max(),
            "final_ini": fechas_final.min(), "final_fin": fechas_final.max(),
            "n_dias": n_dias
        }

    def promedio_por_grupo(fechas):
        tmp = df[df["Fecha"].isin(fechas)].copy()
        diario = tmp.groupby(grupo_cols + ["Fecha"], as_index=False)[col_valor].sum()

        base = grupos.assign(_key=1).merge(
            pd.DataFrame({"Fecha": fechas, "_key": 1}),
            on="_key"
        ).drop(columns="_key")

        diario = base.merge(diario, on=grupo_cols + ["Fecha"], how="left")
        diario[col_valor] = diario[col_valor].fillna(0)

        prom = diario.groupby(grupo_cols, as_index=False).agg(
            PROMEDIO=(col_valor, "mean"),
            TOTAL=(col_valor, "sum"),
            DIAS_EVALUADOS=("Fecha", "count")
        )
        return prom

    inicio = promedio_por_grupo(fechas_inicio).rename(columns={
        "PROMEDIO": "PROM_INICIAL",
        "TOTAL": "TOTAL_INICIAL",
        "DIAS_EVALUADOS": "DIAS_INICIO"
    })

    final = promedio_por_grupo(fechas_final).rename(columns={
        "PROMEDIO": "PROM_FINAL",
        "TOTAL": "TOTAL_FINAL",
        "DIAS_EVALUADOS": "DIAS_FINAL"
    })

    ranking = inicio.merge(final, on=grupo_cols, how="outer").fillna(0)

    ranking["PERDIDA_PROM_DIARIA"] = ranking["PROM_INICIAL"] - ranking["PROM_FINAL"]
    ranking["CAIDA_PCT"] = np.where(
        ranking["PROM_INICIAL"] > 0,
        ranking["PERDIDA_PROM_DIARIA"] / ranking["PROM_INICIAL"] * 100,
        np.nan
    )

    ranking["PERDIDA_TOTAL_ESTIMADA_RANGO"] = ranking["PERDIDA_PROM_DIARIA"] * len(fechas_rango)
    ranking["ESTADO_CAIDA"] = np.select(
        [
            (ranking["PROM_INICIAL"] > 0) & (ranking["PROM_FINAL"] <= 0),
            (ranking["CAIDA_PCT"] >= 50),
            (ranking["CAIDA_PCT"] >= 25),
            (ranking["CAIDA_PCT"] > 0),
        ],
        ["DEJO DE PRODUCIR", "CAIDA CRITICA", "CAIDA ALTA", "CAIDA"],
        default="SIN CAIDA"
    )

    if col_perdida and col_perdida in df.columns:
        perdidas = df.groupby(grupo_cols, as_index=False)[col_perdida].sum().rename(columns={col_perdida: "PERDIDA_REGISTRADA"})
        ranking = ranking.merge(perdidas, on=grupo_cols, how="left")
        ranking["PERDIDA_REGISTRADA"] = ranking["PERDIDA_REGISTRADA"].fillna(0)
    else:
        ranking["PERDIDA_REGISTRADA"] = 0.0

    ranking = ranking[ranking["PROM_INICIAL"] >= float(min_prom_inicial)].copy()
    ranking = ranking[ranking["PERDIDA_PROM_DIARIA"] > 0].copy()
    ranking = ranking.sort_values(
        ["PERDIDA_PROM_DIARIA", "CAIDA_PCT", "PERDIDA_REGISTRADA"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    periodos = {
        "inicio_ini": fechas_inicio.min(),
        "inicio_fin": fechas_inicio.max(),
        "final_ini": fechas_final.min(),
        "final_fin": fechas_final.max(),
        "n_dias": n_dias,
        "dias_rango": len(fechas_rango)
    }

    return ranking, periodos


def calcular_periodo_previo(df_total, col_valor, col_perdida, grupo_cols, fecha_ini, fecha_fin):
    fecha_ini = pd.to_datetime(fecha_ini)
    fecha_fin = pd.to_datetime(fecha_fin)
    dias = (fecha_fin - fecha_ini).days + 1

    previo_fin = fecha_ini - pd.Timedelta(days=1)
    previo_ini = previo_fin - pd.Timedelta(days=dias - 1)

    actual = df_total[(df_total["Fecha"] >= fecha_ini) & (df_total["Fecha"] <= fecha_fin)].copy()
    previo = df_total[(df_total["Fecha"] >= previo_ini) & (df_total["Fecha"] <= previo_fin)].copy()

    if previo.empty:
        return pd.DataFrame(), {"previo_ini": previo_ini, "previo_fin": previo_fin, "hay_previo": False}

    def prom(df, nombre):
        diario = df.groupby(grupo_cols + ["Fecha"], as_index=False)[col_valor].sum()
        return diario.groupby(grupo_cols, as_index=False)[col_valor].mean().rename(columns={col_valor: nombre})

    a = prom(actual, "PROM_ACTUAL")
    p = prom(previo, "PROM_PREVIO")
    res = a.merge(p, on=grupo_cols, how="outer").fillna(0)
    res["PERDIDA_PROM_DIARIA"] = res["PROM_PREVIO"] - res["PROM_ACTUAL"]
    res["CAIDA_PCT"] = np.where(res["PROM_PREVIO"] > 0, res["PERDIDA_PROM_DIARIA"] / res["PROM_PREVIO"] * 100, np.nan)

    if col_perdida and col_perdida in actual.columns:
        perdidas = actual.groupby(grupo_cols, as_index=False)[col_perdida].sum().rename(columns={col_perdida: "PERDIDA_REGISTRADA"})
        res = res.merge(perdidas, on=grupo_cols, how="left")
    else:
        res["PERDIDA_REGISTRADA"] = 0

    res = res[res["PERDIDA_PROM_DIARIA"] > 0].sort_values("PERDIDA_PROM_DIARIA", ascending=False).reset_index(drop=True)
    return res, {"previo_ini": previo_ini, "previo_fin": previo_fin, "hay_previo": True}


def tabla_perdida_registrada(df, grupo_cols, col_perdida):
    if not col_perdida or col_perdida not in df.columns:
        return pd.DataFrame()

    salida = (
        df.groupby(grupo_cols, as_index=False)
        .agg(
            PERDIDA_REGISTRADA=(col_perdida, "sum"),
            DIAS_CON_PERDIDA=(col_perdida, lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0) > 0).sum()))
        )
    )
    salida = salida[salida["PERDIDA_REGISTRADA"] > 0].sort_values("PERDIDA_REGISTRADA", ascending=False)
    return salida.reset_index(drop=True)


def formatear_tabla(df):
    salida = df.copy()
    for col in salida.columns:
        if pd.api.types.is_datetime64_any_dtype(salida[col]):
            salida[col] = salida[col].dt.strftime("%Y-%m-%d")
    for col in salida.select_dtypes(include=["number"]).columns:
        salida[col] = salida[col].round(2)
    return salida


def descargar_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def grafico_barras(df, y_col, x_col, titulo, unidad, top_n=20):
    if df.empty:
        return None

    plot_df = df.head(top_n).copy()
    plot_df = plot_df.sort_values(x_col, ascending=True)

    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        orientation="h",
        text=x_col,
        title=titulo,
        hover_data=[c for c in ["PROM_INICIAL", "PROM_FINAL", "CAIDA_PCT", "PERDIDA_REGISTRADA"] if c in plot_df.columns]
    )
    fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title=f"Pérdida promedio diaria ({unidad})",
        yaxis_title="",
        height=max(430, 22 * len(plot_df) + 180),
        margin=dict(l=20, r=30, t=70, b=40)
    )
    return fig


# ============================================================
# INTERFAZ
# ============================================================

st.title("📉 Análisis de caídas de producción por batería y pozo")
st.caption("El ranking principal compara el promedio de los primeros N días contra el promedio de los últimos N días dentro del rango seleccionado. Así no depende de tener datos del periodo anterior.")

archivo = st.file_uploader("Sube el Excel cierreprod", type=["xlsx", "xls"])

if archivo is None:
    st.info("Sube el Excel para iniciar el análisis.")
    st.stop()

try:
    df, hoja_usada, fila_header = cargar_excel(archivo.getvalue())
except Exception as e:
    st.error(f"No se pudo cargar el Excel: {e}")
    st.stop()

fecha_min = df["Fecha"].min().date()
fecha_max = df["Fecha"].max().date()
baterias = sorted(df["Bateria"].dropna().unique().tolist())

with st.form("formulario_analisis"):
    st.subheader("1. Selección principal")
    c1, c2, c3 = st.columns([1, 1.2, 1])

    with c1:
        producto = st.radio("Producto a analizar", ["Gas", "Petróleo"], horizontal=True)

    columnas_producto = columnas_disponibles_producto(df, producto)
    if not columnas_producto:
        st.error(f"No encontré columnas numéricas para {producto}.")
        st.stop()

    default_col = 0
    for i, col in enumerate(columnas_producto):
        if producto == "Gas" and clave_columna(col) == "prod_gas_real":
            default_col = i
        if producto == "Petróleo" and clave_columna(col) == "prod_oil_real":
            default_col = i

    with c2:
        col_valor = st.selectbox("Columna de producción", columnas_producto, index=default_col)

    with c3:
        top_n = st.number_input("Top a mostrar", min_value=5, max_value=100, value=20, step=5)

    st.subheader("2. Rango de fechas")
    c4, c5, c6 = st.columns([1.3, 1, 1])

    with c4:
        rango = st.date_input(
            "Rango de fechas del análisis",
            value=(fecha_min, fecha_max),
            min_value=fecha_min,
            max_value=fecha_max
        )

    with c5:
        n_dias = st.slider(
            "Comparar primeros N días contra últimos N días",
            min_value=1,
            max_value=31,
            value=7,
            help="Ejemplo: con 7 compara el promedio de los primeros 7 días contra el promedio de los últimos 7 días del rango seleccionado."
        )

    with c6:
        ordenar_info = st.info("El ranking se ordena por mayor pérdida promedio diaria.")

    with st.expander("Filtros opcionales"):
        f1, f2, f3 = st.columns([1.5, 1, 1])
        with f1:
            baterias_sel = st.multiselect(
                "Filtrar baterías opcional. Si lo dejas vacío, analiza todas.",
                baterias,
                default=[]
            )
        with f2:
            texto_pozo = st.text_input("Filtrar pozo por texto opcional", value="")
        with f3:
            min_prom_inicial = st.number_input(
                "Ignorar grupos con promedio inicial menor a",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Sirve para ocultar pozos o baterías con producción inicial muy pequeña. Déjalo en 0 para no filtrar."
            )

    ejecutar = st.form_submit_button("Ejecutar análisis", use_container_width=True)

if not ejecutar:
    st.info("Configura producto, fechas y filtros. Luego presiona Ejecutar análisis.")
    st.stop()

if isinstance(rango, tuple) and len(rango) == 2:
    fecha_ini, fecha_fin = rango
else:
    st.error("Selecciona fecha inicial y fecha final.")
    st.stop()

if pd.to_datetime(fecha_ini) > pd.to_datetime(fecha_fin):
    st.error("La fecha inicial no puede ser mayor que la fecha final.")
    st.stop()

col_perdida = obtener_columna_perdida(df, producto)
unidad = PRODUCTOS[producto]["unidad"]

df_filtrado = aplicar_filtros(df, fecha_ini, fecha_fin, baterias_sel, texto_pozo)
df_filtrado[col_valor] = pd.to_numeric(df_filtrado[col_valor], errors="coerce").fillna(0)

if df_filtrado.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

fechas_rango = lista_fechas(fecha_ini, fecha_fin)
diario_total = diario_lote(df_filtrado, col_valor, fechas_rango)

ranking_bateria, periodos = calcular_ranking_caida(
    df=df_filtrado,
    grupo_cols=["Bateria"],
    col_valor=col_valor,
    col_perdida=col_perdida,
    fecha_ini=fecha_ini,
    fecha_fin=fecha_fin,
    n_dias=n_dias,
    min_prom_inicial=min_prom_inicial
)

ranking_pozo, _ = calcular_ranking_caida(
    df=df_filtrado,
    grupo_cols=["Bateria", "Pozo"],
    col_valor=col_valor,
    col_perdida=col_perdida,
    fecha_ini=fecha_ini,
    fecha_fin=fecha_fin,
    n_dias=n_dias,
    min_prom_inicial=min_prom_inicial
)

perdida_bateria = tabla_perdida_registrada(df_filtrado, ["Bateria"], col_perdida)
perdida_pozo = tabla_perdida_registrada(df_filtrado, ["Bateria", "Pozo"], col_perdida)

# Comparación contra periodo previo solo como referencia secundaria.
ranking_bat_previo, info_previo = calcular_periodo_previo(
    df_total=df,
    col_valor=col_valor,
    col_perdida=col_perdida,
    grupo_cols=["Bateria"],
    fecha_ini=fecha_ini,
    fecha_fin=fecha_fin
)

prom_inicial_lote = promedio_periodo_diario(df_filtrado, col_valor, lista_fechas(periodos["inicio_ini"], periodos["inicio_fin"]))
prom_final_lote = promedio_periodo_diario(df_filtrado, col_valor, lista_fechas(periodos["final_ini"], periodos["final_fin"]))
perdida_lote = prom_inicial_lote - prom_final_lote
caida_pct_lote = (perdida_lote / prom_inicial_lote * 100) if prom_inicial_lote > 0 else np.nan

st.subheader("Resumen ejecutivo")
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Producto", producto)
with k2:
    st.metric(f"Promedio inicial lote ({unidad})", f"{prom_inicial_lote:,.2f}")
with k3:
    st.metric(f"Promedio final lote ({unidad})", f"{prom_final_lote:,.2f}")
with k4:
    st.metric(f"Caída prom. diaria ({unidad})", f"{perdida_lote:,.2f}", f"{caida_pct_lote:,.2f}%" if pd.notna(caida_pct_lote) else None)
with k5:
    st.metric("Días del rango", f"{len(fechas_rango)}")

st.info(
    f"Se compara el promedio inicial del {periodos['inicio_ini'].date()} al {periodos['inicio_fin'].date()} "
    f"contra el promedio final del {periodos['final_ini'].date()} al {periodos['final_fin'].date()}. "
    f"Columna usada: {col_valor}."
)

if ranking_bateria.empty:
    st.warning("No se detectaron caídas por batería dentro del rango seleccionado. Revisa la columna seleccionada, el rango de fechas o el filtro de producción inicial mínima.")
else:
    peor_bat = ranking_bateria.iloc[0]
    peor_pozo = ranking_pozo.iloc[0] if not ranking_pozo.empty else None
    mensaje = (
        f"Mayor caída por batería: {peor_bat['Bateria']} con pérdida promedio diaria de "
        f"{peor_bat['PERDIDA_PROM_DIARIA']:,.2f} {unidad}, pasando de "
        f"{peor_bat['PROM_INICIAL']:,.2f} a {peor_bat['PROM_FINAL']:,.2f} {unidad}."
    )
    if peor_pozo is not None:
        mensaje += (
            f" Mayor caída por pozo: {peor_pozo['Pozo']} en {peor_pozo['Bateria']} con pérdida promedio diaria de "
            f"{peor_pozo['PERDIDA_PROM_DIARIA']:,.2f} {unidad}."
        )
    st.warning(mensaje)

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Caída por batería",
    "Caída por pozo",
    "Tendencia diaria",
    "Pérdida registrada",
    "Detalle batería",
    "Validación"
])

with tab1:
    st.subheader("Ranking de caída por batería dentro del rango")
    st.caption("Ordenado de mayor a menor pérdida promedio diaria. Esta es la vista principal para saber dónde cayó más.")

    cols_bat = [
        "Bateria", "ESTADO_CAIDA", "PROM_INICIAL", "PROM_FINAL",
        "PERDIDA_PROM_DIARIA", "CAIDA_PCT", "PERDIDA_TOTAL_ESTIMADA_RANGO",
        "PERDIDA_REGISTRADA", "DIAS_INICIO", "DIAS_FINAL"
    ]

    st.dataframe(formatear_tabla(ranking_bateria[cols_bat]), use_container_width=True, hide_index=True)

    fig = grafico_barras(
        ranking_bateria,
        y_col="Bateria",
        x_col="PERDIDA_PROM_DIARIA",
        titulo=f"Top {top_n} baterías con mayor caída de {producto.lower()}",
        unidad=unidad,
        top_n=int(top_n)
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Descargar ranking por batería",
        data=descargar_csv(formatear_tabla(ranking_bateria[cols_bat])),
        file_name=f"ranking_caida_{producto.lower()}_por_bateria.csv",
        mime="text/csv"
    )

    with st.expander("Referencia secundaria: comparación contra periodo previo"):
        if info_previo["hay_previo"] and not ranking_bat_previo.empty:
            st.caption(
                f"Periodo previo usado: {info_previo['previo_ini'].date()} al {info_previo['previo_fin'].date()}. "
                "Esta vista solo sirve si el Excel tiene datos antes de la fecha inicial seleccionada."
            )
            st.dataframe(formatear_tabla(ranking_bat_previo.head(50)), use_container_width=True, hide_index=True)
        else:
            st.info(
                f"No hay datos suficientes del periodo previo {info_previo['previo_ini'].date()} al {info_previo['previo_fin'].date()}. "
                "Por eso antes te salía 0 contra periodo previo. Usa el ranking principal dentro del rango."
            )

with tab2:
    st.subheader("Ranking de caída por pozo dentro del rango")
    cols_pozo = [
        "Bateria", "Pozo", "ESTADO_CAIDA", "PROM_INICIAL", "PROM_FINAL",
        "PERDIDA_PROM_DIARIA", "CAIDA_PCT", "PERDIDA_TOTAL_ESTIMADA_RANGO",
        "PERDIDA_REGISTRADA", "DIAS_INICIO", "DIAS_FINAL"
    ]
    st.dataframe(formatear_tabla(ranking_pozo[cols_pozo]), use_container_width=True, hide_index=True)

    graf_pozo = ranking_pozo.copy()
    if not graf_pozo.empty:
        graf_pozo["Pozo_Bateria"] = graf_pozo["Pozo"].astype(str) + " | " + graf_pozo["Bateria"].astype(str)
        fig = grafico_barras(
            graf_pozo,
            y_col="Pozo_Bateria",
            x_col="PERDIDA_PROM_DIARIA",
            titulo=f"Top {top_n} pozos con mayor caída de {producto.lower()}",
            unidad=unidad,
            top_n=int(top_n)
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Descargar ranking por pozo",
        data=descargar_csv(formatear_tabla(ranking_pozo[cols_pozo])),
        file_name=f"ranking_caida_{producto.lower()}_por_pozo.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Tendencia diaria del lote filtrado")

    fig_line = px.line(
        diario_total,
        x="Fecha",
        y=["PRODUCCION", "PROM_MOVIL_7D"],
        title=f"Producción diaria y promedio móvil de 7 días de {producto.lower()}",
        labels={"value": unidad, "variable": "Serie"}
    )
    fig_line.update_layout(height=480)
    st.plotly_chart(fig_line, use_container_width=True)

    st.dataframe(formatear_tabla(diario_total), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Pérdida registrada en el Excel")
    if col_perdida:
        st.caption(f"Columna usada: {col_perdida}")
    else:
        st.warning("No se encontró columna de pérdida registrada para este producto.")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Por batería")
        st.dataframe(formatear_tabla(perdida_bateria.head(50)), use_container_width=True, hide_index=True)
    with c2:
        st.write("Por pozo")
        st.dataframe(formatear_tabla(perdida_pozo.head(50)), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Detalle por batería")
    if ranking_bateria.empty:
        st.info("No hay baterías con caída para detallar.")
    else:
        bateria_detalle = st.selectbox("Selecciona batería para ver sus pozos", ranking_bateria["Bateria"].tolist())
        detalle = ranking_pozo[ranking_pozo["Bateria"] == bateria_detalle].copy()
        st.dataframe(formatear_tabla(detalle[cols_pozo]), use_container_width=True, hide_index=True)

        graf_detalle = detalle.copy()
        if not graf_detalle.empty:
            graf_detalle["Pozo_Bateria"] = graf_detalle["Pozo"].astype(str)
            fig = grafico_barras(
                graf_detalle,
                y_col="Pozo_Bateria",
                x_col="PERDIDA_PROM_DIARIA",
                titulo=f"Pozos que explican la caída en {bateria_detalle}",
                unidad=unidad,
                top_n=int(top_n)
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Validación del archivo")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.metric("Hoja usada", hoja_usada)
    with v2:
        st.metric("Fila de encabezado", fila_header)
    with v3:
        st.metric("Filas leídas", f"{len(df):,}")
    with v4:
        st.metric("Baterías", f"{df['Bateria'].nunique():,}")

    st.write(f"Rango de fechas del archivo: {fecha_min} al {fecha_max}")
    st.write("Columnas detectadas")
    st.dataframe(pd.DataFrame({"Columnas": df.columns.tolist()}), use_container_width=True, hide_index=True)

    st.write("Muestra del archivo cargado")
    muestra_cols = [c for c in ["Fecha", "Pozo", "Bateria", col_valor, col_perdida] if c and c in df.columns]
    st.dataframe(df[muestra_cols].head(30), use_container_width=True, hide_index=True)
