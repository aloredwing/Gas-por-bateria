import io
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Análisis de caídas por batería y pozo",
    page_icon="⛽",
    layout="wide"
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

COLUMNAS_BASE = [
    "Fecha", "Pozo", "Bateria", "Batería", "SubSistema", "SubSist", "TEST_PURP_CD", "T_HRS",
    "PROD_OIL", "PROD_GAS", "PROD_WAT",
    "PROD_OIL_REAL", "PROD_GAS_REAL", "PROD_WAT_REAL",
    "PetroleoPerdido", "GasPerdido", "AguaPerdida",
    "ProdReal Cierre", "ProdGas Real Cierre", "ProdAgua Real Cierre",
    "PROD_GAS_POT", "PROD_GAS_POT_REAL"
]


# ============================================================
# FUNCIONES DE LIMPIEZA Y DETECCIÓN
# ============================================================

def normalizar_columna(columna):
    texto = str(columna).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def clave_columna(columna):
    texto = normalizar_columna(columna).lower()
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"
    }
    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)
    texto = re.sub(r"[^a-z0-9]", "", texto)
    return texto


def detectar_fila_header(excel_bytes):
    vista = pd.read_excel(
        io.BytesIO(excel_bytes),
        header=None,
        nrows=12,
        engine="openpyxl"
    )

    for idx, fila in vista.iterrows():
        valores = [clave_columna(x) for x in fila.tolist()]
        tiene_fecha = "fecha" in valores
        tiene_pozo = "pozo" in valores
        tiene_bateria = "bateria" in valores

        if tiene_fecha and tiene_pozo and tiene_bateria:
            return idx

    return 0


def resolver_columna(df, candidatos):
    mapa = {clave_columna(c): c for c in df.columns}
    for candidato in candidatos:
        clave = clave_columna(candidato)
        if clave in mapa:
            return mapa[clave]
    return None


def convertir_fecha(serie):
    fecha = pd.to_datetime(serie, errors="coerce")

    if fecha.notna().mean() < 0.60:
        numeros = pd.to_numeric(serie, errors="coerce")
        fecha_excel = pd.to_datetime(
            numeros,
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )
        if fecha_excel.notna().sum() > fecha.notna().sum():
            fecha = fecha_excel

    return fecha


@st.cache_data(show_spinner=False)
def cargar_excel(excel_bytes):
    fila_header = detectar_fila_header(excel_bytes)

    columnas = pd.read_excel(
        io.BytesIO(excel_bytes),
        header=fila_header,
        nrows=0,
        engine="openpyxl"
    ).columns.tolist()

    claves_necesarias = {clave_columna(c) for c in COLUMNAS_BASE}
    usecols = []

    for col in columnas:
        if clave_columna(col) in claves_necesarias:
            usecols.append(col)

    if not usecols:
        raise ValueError("No se detectaron columnas útiles. Revisa que el archivo tenga Fecha, Pozo y Bateria.")

    df = pd.read_excel(
        io.BytesIO(excel_bytes),
        header=fila_header,
        usecols=usecols,
        engine="openpyxl"
    )

    df.columns = [normalizar_columna(c) for c in df.columns]

    col_fecha = resolver_columna(df, ["Fecha"])
    col_pozo = resolver_columna(df, ["Pozo"])
    col_bateria = resolver_columna(df, ["Bateria", "Batería"])

    if col_fecha is None or col_pozo is None or col_bateria is None:
        raise ValueError("Faltan columnas base. El Excel debe tener Fecha, Pozo y Bateria.")

    df = df.rename(columns={
        col_fecha: "Fecha",
        col_pozo: "Pozo",
        col_bateria: "Bateria"
    })

    df["Fecha"] = convertir_fecha(df["Fecha"])
    df["Pozo"] = df["Pozo"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["Bateria"] = df["Bateria"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    df = df[df["Fecha"].notna()].copy()
    df = df[(df["Pozo"] != "") & (df["Bateria"] != "")].copy()
    df = df[~df["Pozo"].str.lower().isin(["nan", "none"])]
    df = df[~df["Bateria"].str.lower().isin(["nan", "none"])]

    for col in df.columns:
        if col not in ["Fecha", "Pozo", "Bateria", "SubSistema", "SubSist", "TEST_PURP_CD"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Fecha").reset_index(drop=True)

    return df, fila_header + 1


def columnas_por_producto(df, producto):
    if producto == "Gas":
        produccion = [
            "PROD_GAS_REAL",
            "ProdGas Real Cierre",
            "PROD_GAS",
            "PROD_GAS_POT_REAL",
            "PROD_GAS_POT"
        ]
        perdida = ["GasPerdido"]
        unidad = "MSCFD o unidad de gas del archivo"
    else:
        produccion = [
            "PROD_OIL_REAL",
            "ProdReal Cierre",
            "PROD_OIL"
        ]
        perdida = ["PetroleoPerdido"]
        unidad = "BOPD o unidad de petróleo del archivo"

    cols_prod = []
    for candidato in produccion:
        col = resolver_columna(df, [candidato])
        if col is not None and col not in cols_prod:
            cols_prod.append(col)

    col_perdida = resolver_columna(df, perdida)

    return cols_prod, col_perdida, unidad


# ============================================================
# FUNCIONES DE CÁLCULO
# ============================================================

def validar_rango(rango):
    if isinstance(rango, tuple) and len(rango) == 2:
        return pd.to_datetime(rango[0]), pd.to_datetime(rango[1])
    return None, None


def resumen_lote(df, valor_col, fecha_ini, fecha_fin):
    dias = max((fecha_fin - fecha_ini).days + 1, 1)
    previo_fin = fecha_ini - timedelta(days=1)
    previo_ini = previo_fin - timedelta(days=dias - 1)

    actual = df[(df["Fecha"] >= fecha_ini) & (df["Fecha"] <= fecha_fin)]
    previo = df[(df["Fecha"] >= previo_ini) & (df["Fecha"] <= previo_fin)]

    suma_actual = actual[valor_col].sum(skipna=True)
    suma_previa = previo[valor_col].sum(skipna=True)
    prom_actual = suma_actual / dias
    prom_previo = suma_previa / dias
    caida_abs = prom_actual - prom_previo
    caida_pct = caida_abs / prom_previo * 100 if prom_previo > 0 else np.nan

    return {
        "dias": dias,
        "previo_ini": previo_ini,
        "previo_fin": previo_fin,
        "suma_actual": suma_actual,
        "suma_previa": suma_previa,
        "prom_actual": prom_actual,
        "prom_previo": prom_previo,
        "caida_abs": caida_abs,
        "caida_pct": caida_pct
    }


def comparar_periodo_previo(df, group_cols, valor_col, fecha_ini, fecha_fin, min_prom_previo=0.0):
    dias = max((fecha_fin - fecha_ini).days + 1, 1)
    previo_fin = fecha_ini - timedelta(days=1)
    previo_ini = previo_fin - timedelta(days=dias - 1)

    actual = df[(df["Fecha"] >= fecha_ini) & (df["Fecha"] <= fecha_fin)].copy()
    previo = df[(df["Fecha"] >= previo_ini) & (df["Fecha"] <= previo_fin)].copy()

    if actual.empty and previo.empty:
        return pd.DataFrame(), previo_ini, previo_fin

    agg_actual = (
        actual
        .groupby(group_cols, dropna=False)
        .agg(
            SUM_ACTUAL=(valor_col, "sum"),
            DIAS_CON_DATO_ACTUAL=("Fecha", "nunique"),
            MAX_DIA_ACTUAL=(valor_col, "max"),
            MIN_DIA_ACTUAL=(valor_col, "min")
        )
        .reset_index()
    )

    agg_previo = (
        previo
        .groupby(group_cols, dropna=False)
        .agg(
            SUM_PREVIO=(valor_col, "sum"),
            DIAS_CON_DATO_PREVIO=("Fecha", "nunique")
        )
        .reset_index()
    )

    resumen = agg_actual.merge(agg_previo, on=group_cols, how="outer")

    for col in ["SUM_ACTUAL", "SUM_PREVIO", "DIAS_CON_DATO_ACTUAL", "DIAS_CON_DATO_PREVIO", "MAX_DIA_ACTUAL", "MIN_DIA_ACTUAL"]:
        if col in resumen.columns:
            resumen[col] = resumen[col].fillna(0)

    resumen["PROM_ACTUAL"] = resumen["SUM_ACTUAL"] / dias
    resumen["PROM_PREVIO"] = resumen["SUM_PREVIO"] / dias
    resumen["CAIDA_ABS"] = resumen["PROM_ACTUAL"] - resumen["PROM_PREVIO"]
    resumen["CAIDA_PCT"] = np.where(
        resumen["PROM_PREVIO"] > 0,
        resumen["CAIDA_ABS"] / resumen["PROM_PREVIO"] * 100,
        np.nan
    )
    resumen["PERDIDA_PROM_DIARIA"] = np.where(resumen["CAIDA_ABS"] < 0, -resumen["CAIDA_ABS"], 0)
    resumen["PERDIDA_PCT"] = np.where(resumen["CAIDA_PCT"] < 0, -resumen["CAIDA_PCT"], 0)

    condiciones = [
        (resumen["CAIDA_PCT"] <= -25) | (resumen["PERDIDA_PROM_DIARIA"] >= resumen["PERDIDA_PROM_DIARIA"].quantile(0.90)),
        (resumen["CAIDA_PCT"] <= -15),
        (resumen["CAIDA_PCT"] <= -5)
    ]
    valores = ["CRITICO", "ALTO", "MEDIO"]
    resumen["NIVEL"] = np.select(condiciones, valores, default="NORMAL")

    resumen = resumen[resumen["PROM_PREVIO"] >= min_prom_previo].copy()
    resumen = resumen.sort_values(
        ["PERDIDA_PROM_DIARIA", "PERDIDA_PCT"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return resumen, previo_ini, previo_fin


def comparar_inicio_fin_rango(df, group_cols, valor_col, fecha_ini, fecha_fin, ventana_dias):
    dias_rango = max((fecha_fin - fecha_ini).days + 1, 1)
    ventana = int(min(max(1, ventana_dias), max(1, dias_rango // 2)))

    ini_inicio = fecha_ini
    fin_inicio = fecha_ini + timedelta(days=ventana - 1)
    ini_final = fecha_fin - timedelta(days=ventana - 1)
    fin_final = fecha_fin

    inicio = df[(df["Fecha"] >= ini_inicio) & (df["Fecha"] <= fin_inicio)].copy()
    final = df[(df["Fecha"] >= ini_final) & (df["Fecha"] <= fin_final)].copy()

    if inicio.empty and final.empty:
        return pd.DataFrame(), ini_inicio, fin_inicio, ini_final, fin_final

    agg_inicio = (
        inicio
        .groupby(group_cols, dropna=False)
        .agg(SUM_INICIO=(valor_col, "sum"), DIAS_INICIO=("Fecha", "nunique"))
        .reset_index()
    )

    agg_final = (
        final
        .groupby(group_cols, dropna=False)
        .agg(SUM_FINAL=(valor_col, "sum"), DIAS_FINAL=("Fecha", "nunique"))
        .reset_index()
    )

    resumen = agg_inicio.merge(agg_final, on=group_cols, how="outer")
    for col in ["SUM_INICIO", "SUM_FINAL", "DIAS_INICIO", "DIAS_FINAL"]:
        resumen[col] = resumen[col].fillna(0)

    resumen["PROM_INICIO"] = resumen["SUM_INICIO"] / ventana
    resumen["PROM_FINAL"] = resumen["SUM_FINAL"] / ventana
    resumen["CAIDA_ABS_RANGO"] = resumen["PROM_FINAL"] - resumen["PROM_INICIO"]
    resumen["CAIDA_PCT_RANGO"] = np.where(
        resumen["PROM_INICIO"] > 0,
        resumen["CAIDA_ABS_RANGO"] / resumen["PROM_INICIO"] * 100,
        np.nan
    )
    resumen["PERDIDA_PROM_DIARIA_RANGO"] = np.where(resumen["CAIDA_ABS_RANGO"] < 0, -resumen["CAIDA_ABS_RANGO"], 0)
    resumen["PERDIDA_PCT_RANGO"] = np.where(resumen["CAIDA_PCT_RANGO"] < 0, -resumen["CAIDA_PCT_RANGO"], 0)

    resumen = resumen.sort_values(
        ["PERDIDA_PROM_DIARIA_RANGO", "PERDIDA_PCT_RANGO"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return resumen, ini_inicio, fin_inicio, ini_final, fin_final


def ranking_perdidas_registradas(df, group_cols, perdida_col, valor_col, fecha_ini, fecha_fin):
    if perdida_col is None or perdida_col not in df.columns:
        return pd.DataFrame()

    data = df[(df["Fecha"] >= fecha_ini) & (df["Fecha"] <= fecha_fin)].copy()
    if data.empty:
        return pd.DataFrame()

    resumen = (
        data
        .groupby(group_cols, dropna=False)
        .agg(
            PRODUCCION_TOTAL=(valor_col, "sum"),
            PERDIDA_REGISTRADA=(perdida_col, "sum"),
            DIAS_CON_PERDIDA=(perdida_col, lambda x: (x.fillna(0) > 0).sum()),
            DIAS_CON_DATO=("Fecha", "nunique")
        )
        .reset_index()
    )

    resumen["PERDIDA_SOBRE_PRODUCCION_PCT"] = np.where(
        resumen["PRODUCCION_TOTAL"] > 0,
        resumen["PERDIDA_REGISTRADA"] / resumen["PRODUCCION_TOTAL"] * 100,
        np.nan
    )

    resumen = resumen.sort_values("PERDIDA_REGISTRADA", ascending=False).reset_index(drop=True)
    return resumen


def serie_diaria(df, group_col, valor_col, fecha_ini, fecha_fin, entidad=None):
    data = df[(df["Fecha"] >= fecha_ini) & (df["Fecha"] <= fecha_fin)].copy()

    if entidad is not None and group_col in data.columns:
        data = data[data[group_col] == entidad]

    if data.empty:
        return pd.DataFrame()

    serie = (
        data
        .groupby("Fecha", as_index=False)
        .agg(PRODUCCION=(valor_col, "sum"))
        .sort_values("Fecha")
    )
    serie["PROM_MOVIL_7D"] = serie["PRODUCCION"].rolling(7, min_periods=1).mean()
    return serie


def formatear_tabla(df):
    salida = df.copy()
    for col in salida.columns:
        if pd.api.types.is_datetime64_any_dtype(salida[col]):
            salida[col] = salida[col].dt.strftime("%Y-%m-%d")
    for col in salida.select_dtypes(include=["number"]).columns:
        salida[col] = salida[col].round(2)
    return salida


def csv_download(df):
    return formatear_tabla(df).to_csv(index=False).encode("utf-8-sig")


# ============================================================
# FUNCIONES DE GRÁFICO
# ============================================================

def etiqueta_entidad(df, group_cols):
    if "Pozo" in group_cols and "Bateria" in df.columns:
        return df["Pozo"].astype(str) + " | " + df["Bateria"].astype(str)
    return df[group_cols[0]].astype(str)


def grafico_top_barras(df, group_cols, valor_col, titulo, etiqueta_x):
    if df.empty:
        return None

    data = df.head(20).copy()
    data["ENTIDAD_LABEL"] = etiqueta_entidad(data, group_cols)
    data = data.sort_values(valor_col, ascending=True)

    fig = px.bar(
        data,
        x=valor_col,
        y="ENTIDAD_LABEL",
        orientation="h",
        text=valor_col,
        title=titulo,
        labels={valor_col: etiqueta_x, "ENTIDAD_LABEL": ""},
        template="plotly_white"
    )
    fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
    fig.update_layout(height=520, margin=dict(l=20, r=30, t=70, b=30))
    return fig


def grafico_tendencia(serie, titulo, etiqueta_y):
    if serie.empty:
        return None

    fig = px.line(
        serie,
        x="Fecha",
        y=["PRODUCCION", "PROM_MOVIL_7D"],
        title=titulo,
        labels={"value": etiqueta_y, "variable": "Serie"},
        template="plotly_white"
    )
    fig.update_layout(height=430, margin=dict(l=20, r=30, t=70, b=30))
    return fig


# ============================================================
# INTERFAZ
# ============================================================

st.title("⛽ Análisis de caídas de producción por batería y pozo")
st.caption("Diseñado para Excel tipo cierre de producción diario con columnas Fecha, Pozo, Bateria, PROD_GAS y PROD_OIL.")

producto = st.radio(
    "Primero selecciona qué quieres analizar",
    ["Gas", "Petróleo"],
    horizontal=True
)

archivo = st.file_uploader(
    "Sube el Excel de cierre de producción",
    type=["xlsx"]
)

if archivo is None:
    st.info("Sube el Excel. Luego podrás elegir rango de fechas, columna de producción y presionar Ejecutar análisis.")
    st.stop()

excel_bytes = archivo.getvalue()

with st.spinner("Cargando Excel. Solo se leen las columnas necesarias para que no se ponga pesado."):
    try:
        df, fila_header_detectada = cargar_excel(excel_bytes)
    except Exception as e:
        st.error(f"No pude cargar el Excel: {e}")
        st.stop()

cols_prod, col_perdida, unidad = columnas_por_producto(df, producto)

if not cols_prod:
    st.error(f"No encontré columnas de producción para {producto}.")
    st.write("Columnas detectadas:")
    st.write(list(df.columns))
    st.stop()

fecha_min = df["Fecha"].min().date()
fecha_max = df["Fecha"].max().date()
baterias = sorted(df["Bateria"].dropna().unique().tolist())

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.1, 1.2, 1.2, 1])

    with c1:
        col_produccion = st.selectbox(
            "Columna de producción a usar",
            cols_prod,
            index=0
        )

    with c2:
        rango_fechas = st.date_input(
            "Rango de fechas del periodo actual",
            value=(fecha_min, fecha_max),
            min_value=fecha_min,
            max_value=fecha_max
        )

    with c3:
        baterias_sel = st.multiselect(
            "Filtrar baterías opcional",
            baterias,
            default=[]
        )

    with c4:
        min_prom_previo = st.number_input(
            "Mín. promedio previo",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    fecha_ini, fecha_fin = validar_rango(rango_fechas)

    if fecha_ini is None or fecha_fin is None:
        st.warning("Selecciona fecha inicial y fecha final.")
        st.stop()

    dias_rango = max((fecha_fin - fecha_ini).days + 1, 1)
    ventana_default = int(min(7, max(1, dias_rango // 2)))
    ventana_max = int(max(1, min(30, dias_rango // 2)))

    ventana_inicio_fin = st.slider(
        "Para caída dentro del rango, comparar primeros N días contra últimos N días",
        min_value=1,
        max_value=ventana_max,
        value=ventana_default
    )

    ejecutar = st.button("Ejecutar análisis", type="primary", use_container_width=True)

if not ejecutar:
    st.info("Ajusta los filtros y presiona Ejecutar análisis.")
    st.stop()

if baterias_sel:
    df_analisis = df[df["Bateria"].isin(baterias_sel)].copy()
else:
    df_analisis = df.copy()

if df_analisis.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# ============================================================
# EJECUCIÓN DE CÁLCULOS
# ============================================================

lote = resumen_lote(df_analisis, col_produccion, fecha_ini, fecha_fin)

ranking_bateria, previo_ini, previo_fin = comparar_periodo_previo(
    df_analisis,
    ["Bateria"],
    col_produccion,
    fecha_ini,
    fecha_fin,
    min_prom_previo=min_prom_previo
)

ranking_pozo, _, _ = comparar_periodo_previo(
    df_analisis,
    ["Pozo", "Bateria"],
    col_produccion,
    fecha_ini,
    fecha_fin,
    min_prom_previo=min_prom_previo
)

rango_bateria, ini_inicio, fin_inicio, ini_final, fin_final = comparar_inicio_fin_rango(
    df_analisis,
    ["Bateria"],
    col_produccion,
    fecha_ini,
    fecha_fin,
    ventana_inicio_fin
)

rango_pozo, _, _, _, _ = comparar_inicio_fin_rango(
    df_analisis,
    ["Pozo", "Bateria"],
    col_produccion,
    fecha_ini,
    fecha_fin,
    ventana_inicio_fin
)

perdidas_bateria = ranking_perdidas_registradas(
    df_analisis,
    ["Bateria"],
    col_perdida,
    col_produccion,
    fecha_ini,
    fecha_fin
)

perdidas_pozo = ranking_perdidas_registradas(
    df_analisis,
    ["Pozo", "Bateria"],
    col_perdida,
    col_produccion,
    fecha_ini,
    fecha_fin
)

peor_bateria = ranking_bateria.iloc[0] if not ranking_bateria.empty else None
peor_pozo = ranking_pozo.iloc[0] if not ranking_pozo.empty else None

# ============================================================
# RESUMEN EJECUTIVO
# ============================================================

st.subheader("Resumen ejecutivo")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Producto", producto)

with m2:
    st.metric("Promedio actual lote", f"{lote['prom_actual']:,.2f}")

with m3:
    st.metric("Promedio previo lote", f"{lote['prom_previo']:,.2f}")

with m4:
    delta_pct = "" if pd.isna(lote["caida_pct"]) else f"{lote['caida_pct']:,.2f}%"
    st.metric("Variación lote", f"{lote['caida_abs']:,.2f}", delta_pct)

with m5:
    st.metric("Días analizados", f"{lote['dias']}")

st.info(
    f"Periodo actual: {fecha_ini.date()} al {fecha_fin.date()}. "
    f"Periodo previo comparado: {previo_ini.date()} al {previo_fin.date()}. "
    f"Columna usada: {col_produccion}."
)

if peor_bateria is not None and peor_pozo is not None:
    st.warning(
        f"Mayor caída por batería: {peor_bateria['Bateria']} con pérdida promedio diaria de "
        f"{peor_bateria['PERDIDA_PROM_DIARIA']:,.2f}. "
        f"Mayor caída por pozo: {peor_pozo['Pozo']} en {peor_pozo['Bateria']} con pérdida promedio diaria de "
        f"{peor_pozo['PERDIDA_PROM_DIARIA']:,.2f}."
    )

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Caída por batería",
    "Caída por pozo",
    "Dentro del rango",
    "Pérdida registrada",
    "Detalle batería",
    "Validación"
])


# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.subheader("Ranking de caída por batería contra periodo previo")

    if ranking_bateria.empty:
        st.info("No hay ranking de batería con los filtros seleccionados.")
    else:
        fig = grafico_top_barras(
            ranking_bateria,
            ["Bateria"],
            "PERDIDA_PROM_DIARIA",
            f"Top baterías con mayor caída de {producto.lower()}",
            f"Pérdida promedio diaria de {producto.lower()}"
        )
        st.plotly_chart(fig, use_container_width=True)

        cols = [
            "NIVEL", "Bateria", "PROM_ACTUAL", "PROM_PREVIO", "CAIDA_ABS", "CAIDA_PCT",
            "PERDIDA_PROM_DIARIA", "PERDIDA_PCT", "SUM_ACTUAL", "SUM_PREVIO",
            "DIAS_CON_DATO_ACTUAL", "DIAS_CON_DATO_PREVIO"
        ]
        st.dataframe(formatear_tabla(ranking_bateria[cols]), use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar ranking de baterías",
            data=csv_download(ranking_bateria[cols]),
            file_name=f"ranking_caida_baterias_{producto.lower()}.csv",
            mime="text/csv"
        )


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.subheader("Ranking de caída por pozo contra periodo previo")

    if ranking_pozo.empty:
        st.info("No hay ranking de pozo con los filtros seleccionados.")
    else:
        fig = grafico_top_barras(
            ranking_pozo,
            ["Pozo", "Bateria"],
            "PERDIDA_PROM_DIARIA",
            f"Top pozos con mayor caída de {producto.lower()}",
            f"Pérdida promedio diaria de {producto.lower()}"
        )
        st.plotly_chart(fig, use_container_width=True)

        cols = [
            "NIVEL", "Pozo", "Bateria", "PROM_ACTUAL", "PROM_PREVIO", "CAIDA_ABS", "CAIDA_PCT",
            "PERDIDA_PROM_DIARIA", "PERDIDA_PCT", "SUM_ACTUAL", "SUM_PREVIO",
            "DIAS_CON_DATO_ACTUAL", "DIAS_CON_DATO_PREVIO"
        ]
        st.dataframe(formatear_tabla(ranking_pozo[cols]), use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar ranking de pozos",
            data=csv_download(ranking_pozo[cols]),
            file_name=f"ranking_caida_pozos_{producto.lower()}.csv",
            mime="text/csv"
        )


# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.subheader("Caída dentro del rango seleccionado")
    st.caption(
        f"Compara primeros {ventana_inicio_fin} días: {ini_inicio.date()} al {fin_inicio.date()} "
        f"contra últimos {ventana_inicio_fin} días: {ini_final.date()} al {fin_final.date()}."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("Baterías")
        if rango_bateria.empty:
            st.info("Sin datos.")
        else:
            fig = grafico_top_barras(
                rango_bateria,
                ["Bateria"],
                "PERDIDA_PROM_DIARIA_RANGO",
                f"Caída interna por batería en el rango",
                "Pérdida promedio diaria"
            )
            st.plotly_chart(fig, use_container_width=True)
            cols = ["Bateria", "PROM_INICIO", "PROM_FINAL", "CAIDA_ABS_RANGO", "CAIDA_PCT_RANGO", "PERDIDA_PROM_DIARIA_RANGO"]
            st.dataframe(formatear_tabla(rango_bateria[cols].head(20)), use_container_width=True, hide_index=True)

    with col_b:
        st.write("Pozos")
        if rango_pozo.empty:
            st.info("Sin datos.")
        else:
            fig = grafico_top_barras(
                rango_pozo,
                ["Pozo", "Bateria"],
                "PERDIDA_PROM_DIARIA_RANGO",
                f"Caída interna por pozo en el rango",
                "Pérdida promedio diaria"
            )
            st.plotly_chart(fig, use_container_width=True)
            cols = ["Pozo", "Bateria", "PROM_INICIO", "PROM_FINAL", "CAIDA_ABS_RANGO", "CAIDA_PCT_RANGO", "PERDIDA_PROM_DIARIA_RANGO"]
            st.dataframe(formatear_tabla(rango_pozo[cols].head(20)), use_container_width=True, hide_index=True)


# ============================================================
# TAB 4
# ============================================================

with tab4:
    st.subheader("Pérdida registrada en el Excel")

    if col_perdida is None:
        st.info("No se encontró columna de pérdida registrada para este producto.")
    else:
        st.caption(f"Columna usada para pérdida registrada: {col_perdida}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("Por batería")
            if perdidas_bateria.empty:
                st.info("No hay pérdida registrada.")
            else:
                fig = grafico_top_barras(
                    perdidas_bateria,
                    ["Bateria"],
                    "PERDIDA_REGISTRADA",
                    f"Top pérdida registrada por batería",
                    "Pérdida registrada"
                )
                st.plotly_chart(fig, use_container_width=True)
                cols = ["Bateria", "PERDIDA_REGISTRADA", "PRODUCCION_TOTAL", "PERDIDA_SOBRE_PRODUCCION_PCT", "DIAS_CON_PERDIDA"]
                st.dataframe(formatear_tabla(perdidas_bateria[cols].head(30)), use_container_width=True, hide_index=True)

        with col_b:
            st.write("Por pozo")
            if perdidas_pozo.empty:
                st.info("No hay pérdida registrada.")
            else:
                fig = grafico_top_barras(
                    perdidas_pozo,
                    ["Pozo", "Bateria"],
                    "PERDIDA_REGISTRADA",
                    f"Top pérdida registrada por pozo",
                    "Pérdida registrada"
                )
                st.plotly_chart(fig, use_container_width=True)
                cols = ["Pozo", "Bateria", "PERDIDA_REGISTRADA", "PRODUCCION_TOTAL", "PERDIDA_SOBRE_PRODUCCION_PCT", "DIAS_CON_PERDIDA"]
                st.dataframe(formatear_tabla(perdidas_pozo[cols].head(30)), use_container_width=True, hide_index=True)


# ============================================================
# TAB 5
# ============================================================

with tab5:
    st.subheader("Detalle por batería y pozos que explican la caída")

    if ranking_bateria.empty:
        st.info("No hay baterías para detallar.")
    else:
        lista_bat = ranking_bateria["Bateria"].astype(str).tolist()
        bateria_detalle = st.selectbox("Selecciona batería", lista_bat)

        df_bat = df_analisis[df_analisis["Bateria"] == bateria_detalle].copy()
        rank_pozos_bat, _, _ = comparar_periodo_previo(
            df_bat,
            ["Pozo"],
            col_produccion,
            fecha_ini,
            fecha_fin,
            min_prom_previo=min_prom_previo
        )

        serie_bat = serie_diaria(df_analisis, "Bateria", col_produccion, fecha_ini, fecha_fin, bateria_detalle)
        fig = grafico_tendencia(
            serie_bat,
            f"Tendencia diaria de {producto.lower()} en batería {bateria_detalle}",
            unidad
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("Pozos que más explican la caída de esta batería")
        if rank_pozos_bat.empty:
            st.info("Sin pozos con caída calculable para esta batería.")
        else:
            fig = grafico_top_barras(
                rank_pozos_bat,
                ["Pozo"],
                "PERDIDA_PROM_DIARIA",
                f"Pozos con mayor caída en {bateria_detalle}",
                "Pérdida promedio diaria"
            )
            st.plotly_chart(fig, use_container_width=True)
            cols = [
                "NIVEL", "Pozo", "PROM_ACTUAL", "PROM_PREVIO", "CAIDA_ABS", "CAIDA_PCT",
                "PERDIDA_PROM_DIARIA", "SUM_ACTUAL", "SUM_PREVIO", "DIAS_CON_DATO_ACTUAL", "DIAS_CON_DATO_PREVIO"
            ]
            st.dataframe(formatear_tabla(rank_pozos_bat[cols]), use_container_width=True, hide_index=True)


# ============================================================
# TAB 6
# ============================================================

with tab6:
    st.subheader("Validación de carga")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Filas cargadas", f"{len(df):,}")
    with c2:
        st.metric("Baterías", f"{df['Bateria'].nunique():,}")
    with c3:
        st.metric("Pozos", f"{df['Pozo'].nunique():,}")
    with c4:
        st.metric("Fila de encabezado", fila_header_detectada)

    st.write(f"Rango de fechas detectado: {fecha_min} al {fecha_max}")
    st.write(f"Columna de producción seleccionada: {col_produccion}")
    st.write(f"Columna de pérdida registrada: {col_perdida if col_perdida else 'No encontrada'}")

    st.write("Columnas leídas")
    st.dataframe(pd.DataFrame({"Columna": list(df.columns)}), use_container_width=True, hide_index=True)

    st.write("Muestra de datos")
    st.dataframe(formatear_tabla(df.head(30)), use_container_width=True, hide_index=True)
