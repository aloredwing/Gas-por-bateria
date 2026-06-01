import io
import re
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


st.set_page_config(
    page_title="Gas por Baterías Lote X",
    page_icon="⛽",
    layout="wide"
)


# ============================================================
# CONFIGURACIÓN VISUAL
# ============================================================

MESES_ES = {
    1: "Ene",
    2: "Feb",
    3: "Mar",
    4: "Abr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dic"
}


# ============================================================
# FUNCIONES DE CARGA Y LIMPIEZA
# ============================================================

def normalizar_columna(columna):
    texto = str(columna).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def detectar_hoja(xls):
    """
    Prioriza una hoja con nombre típico de tabla de gas.
    Si no existe, usa la primera hoja.
    """
    posibles = [
        "tbl_produccion_gas_baterias",
        "produccion_gas_baterias",
        "gas por baterias",
        "gas por baterías",
        "gas"
    ]

    for hoja in xls.sheet_names:
        hoja_norm = hoja.strip().lower()
        if hoja_norm in posibles:
            return hoja

    return xls.sheet_names[0]


def detectar_columna_fecha(df):
    """
    Detecta la columna de fecha. Primero busca nombres típicos.
    Si no los encuentra, prueba cuál columna puede convertirse mejor a fecha.
    """
    nombres_fecha = ["Fecha", "FECHA", "fecha", "Date", "DATE", "date"]

    for col in df.columns:
        if str(col).strip() in nombres_fecha:
            return col

    mejor_col = None
    mejor_ratio = 0

    for col in df.columns:
        serie = pd.to_datetime(df[col], errors="coerce")
        ratio = serie.notna().mean()
        if ratio > mejor_ratio:
            mejor_ratio = ratio
            mejor_col = col

    if mejor_col is not None and mejor_ratio >= 0.60:
        return mejor_col

    raise ValueError("No pude detectar la columna de fecha. La hoja debe tener una columna llamada Fecha.")


def cargar_excel_gas(uploaded_file):
    """
    Carga el Excel en formato ancho:
    Fecha | batería 1 | batería 2 | batería 3 | ...
    """
    xls = pd.ExcelFile(uploaded_file)
    hoja = detectar_hoja(xls)

    df = pd.read_excel(xls, sheet_name=hoja)
    df.columns = [normalizar_columna(c) for c in df.columns]

    col_fecha = detectar_columna_fecha(df)
    df = df.rename(columns={col_fecha: "Fecha"})
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    columnas_excluir_clave = [
        "fecha",
        "comentario",
        "comentarios",
        "observacion",
        "observación",
        "observaciones",
        "id",
        "total",
        "lote",
        "campo"
    ]

    baterias = []

    for col in df.columns:
        col_norm = str(col).strip().lower()

        if any(palabra in col_norm for palabra in columnas_excluir_clave):
            continue

        serie_num = pd.to_numeric(df[col], errors="coerce")

        if serie_num.notna().sum() > 0:
            df[col] = serie_num
            baterias.append(col)

    if not baterias:
        raise ValueError("No encontré columnas numéricas de baterías. Revisa que las baterías estén como columnas con valores de gas.")

    df = df[df["Fecha"].notna()].copy()
    df = df[df[baterias].notna().any(axis=1)].copy()
    df = df.sort_values("Fecha").reset_index(drop=True)

    return df, baterias, hoja


def crear_total_lote(df, baterias):
    salida = df.copy()
    salida["TOTAL_GAS_LOTE"] = salida[baterias].sum(axis=1, skipna=True)
    return salida


def convertir_a_formato_largo(df, baterias):
    largo = df.melt(
        id_vars=["Fecha"],
        value_vars=baterias,
        var_name="BATERIA",
        value_name="GAS"
    )

    largo["GAS"] = pd.to_numeric(largo["GAS"], errors="coerce")
    largo = largo[largo["GAS"].notna()].copy()
    largo["AÑO"] = largo["Fecha"].dt.year
    largo["MES"] = largo["Fecha"].dt.month
    largo["MES_TXT"] = largo["MES"].map(MESES_ES)
    largo["MES_LABEL"] = largo["MES"].astype(str).str.zfill(2) + " " + largo["MES_TXT"]
    largo["PERIODO_MES"] = largo["Fecha"].dt.to_period("M").dt.to_timestamp()

    return largo


# ============================================================
# FUNCIONES DE CÁLCULO
# ============================================================

def promedio_periodo(df, baterias, fecha_ini, fecha_fin):
    mask = (df["Fecha"] >= pd.to_datetime(fecha_ini)) & (df["Fecha"] <= pd.to_datetime(fecha_fin))
    return df.loc[mask, baterias].mean(numeric_only=True)


def suma_periodo(df, baterias, fecha_ini, fecha_fin):
    mask = (df["Fecha"] >= pd.to_datetime(fecha_ini)) & (df["Fecha"] <= pd.to_datetime(fecha_fin))
    return df.loc[mask, baterias].sum(numeric_only=True)


def dias_validos_periodo(df, baterias, fecha_ini, fecha_fin):
    mask = (df["Fecha"] >= pd.to_datetime(fecha_ini)) & (df["Fecha"] <= pd.to_datetime(fecha_fin))
    return df.loc[mask, baterias].notna().sum()


def crear_resumen_criticidad(df, baterias, fecha_corte, ventana_dias):
    """
    Criterio de criticidad:
    45 por ciento caída de periodo actual contra periodo previo
    25 por ciento caída últimos 7 días contra 7 días previos
    20 por ciento caída contra mismo periodo del año anterior
    10 por ciento último día contra promedio del periodo actual
    """
    fecha_corte = pd.to_datetime(fecha_corte)

    actual_ini = fecha_corte - pd.Timedelta(days=ventana_dias - 1)
    actual_fin = fecha_corte

    previo_fin = actual_ini - pd.Timedelta(days=1)
    previo_ini = previo_fin - pd.Timedelta(days=ventana_dias - 1)

    yoy_ini = actual_ini - pd.DateOffset(years=1)
    yoy_fin = actual_fin - pd.DateOffset(years=1)

    ult_7_ini = fecha_corte - pd.Timedelta(days=6)
    prev_7_fin = ult_7_ini - pd.Timedelta(days=1)
    prev_7_ini = prev_7_fin - pd.Timedelta(days=6)

    actual_prom = promedio_periodo(df, baterias, actual_ini, actual_fin)
    previo_prom = promedio_periodo(df, baterias, previo_ini, previo_fin)
    yoy_prom = promedio_periodo(df, baterias, yoy_ini, yoy_fin)

    actual_sum = suma_periodo(df, baterias, actual_ini, actual_fin)
    previo_sum = suma_periodo(df, baterias, previo_ini, previo_fin)

    dias_actual = dias_validos_periodo(df, baterias, actual_ini, actual_fin)
    dias_previo = dias_validos_periodo(df, baterias, previo_ini, previo_fin)

    ult_7_prom = promedio_periodo(df, baterias, ult_7_ini, actual_fin)
    prev_7_prom = promedio_periodo(df, baterias, prev_7_ini, prev_7_fin)

    ultimo_dia_df = df[df["Fecha"] == fecha_corte][baterias]

    if ultimo_dia_df.empty:
        ultimo = pd.Series(index=baterias, dtype=float)
    else:
        ultimo = ultimo_dia_df.iloc[-1]

    resumen = pd.DataFrame({
        "BATERIA": baterias,
        "PROM_ACTUAL": actual_prom.reindex(baterias).values,
        "PROM_PREVIO": previo_prom.reindex(baterias).values,
        "PROM_MISMO_PERIODO_AÑO_ANT": yoy_prom.reindex(baterias).values,
        "SUM_ACTUAL": actual_sum.reindex(baterias).values,
        "SUM_PREVIO": previo_sum.reindex(baterias).values,
        "DIAS_CON_DATO_ACTUAL": dias_actual.reindex(baterias).values,
        "DIAS_CON_DATO_PREVIO": dias_previo.reindex(baterias).values,
        "PROM_ULT_7D": ult_7_prom.reindex(baterias).values,
        "PROM_7D_PREVIO": prev_7_prom.reindex(baterias).values,
        "ULTIMO_DIA": ultimo.reindex(baterias).values
    })

    resumen["CAIDA_ABS"] = resumen["PROM_ACTUAL"] - resumen["PROM_PREVIO"]
    resumen["CAIDA_PCT"] = np.where(
        resumen["PROM_PREVIO"] > 0,
        resumen["CAIDA_ABS"] / resumen["PROM_PREVIO"] * 100,
        np.nan
    )

    resumen["CAIDA_ABS_7D"] = resumen["PROM_ULT_7D"] - resumen["PROM_7D_PREVIO"]
    resumen["CAIDA_PCT_7D"] = np.where(
        resumen["PROM_7D_PREVIO"] > 0,
        resumen["CAIDA_ABS_7D"] / resumen["PROM_7D_PREVIO"] * 100,
        np.nan
    )

    resumen["CAIDA_ABS_YOY"] = resumen["PROM_ACTUAL"] - resumen["PROM_MISMO_PERIODO_AÑO_ANT"]
    resumen["CAIDA_PCT_YOY"] = np.where(
        resumen["PROM_MISMO_PERIODO_AÑO_ANT"] > 0,
        resumen["CAIDA_ABS_YOY"] / resumen["PROM_MISMO_PERIODO_AÑO_ANT"] * 100,
        np.nan
    )

    resumen["VAR_ULTIMO_VS_PROM"] = resumen["ULTIMO_DIA"] - resumen["PROM_ACTUAL"]
    resumen["VAR_PCT_ULTIMO_VS_PROM"] = np.where(
        resumen["PROM_ACTUAL"] > 0,
        resumen["VAR_ULTIMO_VS_PROM"] / resumen["PROM_ACTUAL"] * 100,
        np.nan
    )

    resumen["SCORE_CRITICIDAD"] = (
        (-resumen["CAIDA_PCT"].clip(upper=0).fillna(0)) * 0.45 +
        (-resumen["CAIDA_PCT_7D"].clip(upper=0).fillna(0)) * 0.25 +
        (-resumen["CAIDA_PCT_YOY"].clip(upper=0).fillna(0)) * 0.20 +
        (-resumen["VAR_PCT_ULTIMO_VS_PROM"].clip(upper=0).fillna(0)) * 0.10
    )

    def nivel(row):
        if row["CAIDA_PCT"] <= -20 and row["CAIDA_PCT_7D"] <= -10:
            return "CRITICO"
        if row["CAIDA_PCT"] <= -12:
            return "ALTO"
        if row["CAIDA_PCT_7D"] <= -20:
            return "ALTO"
        if row["CAIDA_PCT"] <= -5:
            return "MEDIO"
        if row["CAIDA_PCT_7D"] <= -10:
            return "MEDIO"
        return "NORMAL"

    resumen["NIVEL"] = resumen.apply(nivel, axis=1)

    orden_nivel = {
        "CRITICO": 1,
        "ALTO": 2,
        "MEDIO": 3,
        "NORMAL": 4
    }

    resumen["ORDEN_NIVEL"] = resumen["NIVEL"].map(orden_nivel)
    resumen = resumen.sort_values(
        ["ORDEN_NIVEL", "SCORE_CRITICIDAD", "CAIDA_ABS"],
        ascending=[True, False, True]
    ).drop(columns=["ORDEN_NIVEL"])

    periodos = {
        "actual_ini": actual_ini,
        "actual_fin": actual_fin,
        "previo_ini": previo_ini,
        "previo_fin": previo_fin,
        "yoy_ini": yoy_ini,
        "yoy_fin": yoy_fin,
        "ult_7_ini": ult_7_ini,
        "prev_7_ini": prev_7_ini,
        "prev_7_fin": prev_7_fin
    }

    return resumen, periodos


def calcular_mensual_bateria(largo, bateria, anio, metrica):
    data = largo[(largo["BATERIA"] == bateria) & (largo["AÑO"] == anio)].copy()

    if data.empty:
        return pd.DataFrame()

    mensual = (
        data
        .groupby(["AÑO", "MES", "MES_LABEL"], as_index=False)
        .agg(
            PROM_MENSUAL=("GAS", "mean"),
            TOTAL_MENSUAL=("GAS", "sum"),
            DIAS_CON_DATO=("GAS", "count")
        )
    )

    mensual = mensual.sort_values("MES").reset_index(drop=True)
    mensual["VALOR_GRAFICA"] = mensual["PROM_MENSUAL"] if metrica == "Promedio diario mensual" else mensual["TOTAL_MENSUAL"]
    mensual["MES_PREVIO_VALOR"] = mensual["VALOR_GRAFICA"].shift(1)
    mensual["CAIDA_MENSUAL_ABS"] = mensual["VALOR_GRAFICA"] - mensual["MES_PREVIO_VALOR"]
    mensual["CAIDA_MENSUAL_PCT"] = np.where(
        mensual["MES_PREVIO_VALOR"] > 0,
        mensual["CAIDA_MENSUAL_ABS"] / mensual["MES_PREVIO_VALOR"] * 100,
        np.nan
    )

    mensual["ES_CAIDA"] = mensual["CAIDA_MENSUAL_ABS"] < 0

    caidas = mensual[mensual["ES_CAIDA"]].copy()

    if caidas.empty:
        mensual["ALERTA_CAIDA"] = ""
    else:
        umbral_abs = caidas["CAIDA_MENSUAL_ABS"].quantile(0.25)
        mensual["ALERTA_CAIDA"] = np.where(
            (mensual["CAIDA_MENSUAL_PCT"] <= -10) | (mensual["CAIDA_MENSUAL_ABS"] <= umbral_abs),
            "CAIDA FUERTE",
            ""
        )

    mensual["CAIDA_LABEL"] = mensual["CAIDA_MENSUAL_PCT"].apply(
        lambda x: "" if pd.isna(x) else f"{x:.1f}%"
    )

    return mensual


def calcular_mensual_lote(df_total, anio):
    data = df_total[df_total["Fecha"].dt.year == anio].copy()

    if data.empty:
        return pd.DataFrame()

    data["MES"] = data["Fecha"].dt.month
    data["MES_TXT"] = data["MES"].map(MESES_ES)
    data["MES_LABEL"] = data["MES"].astype(str).str.zfill(2) + " " + data["MES_TXT"]

    mensual = (
        data
        .groupby(["MES", "MES_LABEL"], as_index=False)
        .agg(
            PROMEDIO_DIARIO_LOTE=("TOTAL_GAS_LOTE", "mean"),
            TOTAL_MENSUAL_LOTE=("TOTAL_GAS_LOTE", "sum"),
            DIAS_CON_DATO=("TOTAL_GAS_LOTE", "count")
        )
        .sort_values("MES")
    )

    mensual["PROMEDIO_PREVIO"] = mensual["PROMEDIO_DIARIO_LOTE"].shift(1)
    mensual["VAR_MENSUAL_ABS"] = mensual["PROMEDIO_DIARIO_LOTE"] - mensual["PROMEDIO_PREVIO"]
    mensual["VAR_MENSUAL_PCT"] = np.where(
        mensual["PROMEDIO_PREVIO"] > 0,
        mensual["VAR_MENSUAL_ABS"] / mensual["PROMEDIO_PREVIO"] * 100,
        np.nan
    )

    return mensual


def calcular_caidas_mensuales_global(largo, anio, metrica):
    data = largo[largo["AÑO"] == anio].copy()

    if data.empty:
        return pd.DataFrame()

    mensual = (
        data
        .groupby(["BATERIA", "AÑO", "MES", "MES_LABEL"], as_index=False)
        .agg(
            PROM_MENSUAL=("GAS", "mean"),
            TOTAL_MENSUAL=("GAS", "sum"),
            DIAS_CON_DATO=("GAS", "count")
        )
    )

    mensual = mensual.sort_values(["BATERIA", "AÑO", "MES"])
    mensual["VALOR"] = mensual["PROM_MENSUAL"] if metrica == "Promedio diario mensual" else mensual["TOTAL_MENSUAL"]
    mensual["VALOR_PREVIO"] = mensual.groupby("BATERIA")["VALOR"].shift(1)
    mensual["CAIDA_ABS_MES"] = mensual["VALOR"] - mensual["VALOR_PREVIO"]
    mensual["CAIDA_PCT_MES"] = np.where(
        mensual["VALOR_PREVIO"] > 0,
        mensual["CAIDA_ABS_MES"] / mensual["VALOR_PREVIO"] * 100,
        np.nan
    )

    caidas = mensual[mensual["CAIDA_ABS_MES"] < 0].copy()
    caidas = caidas.sort_values(["CAIDA_PCT_MES", "CAIDA_ABS_MES"], ascending=[True, True])

    return caidas


def formatear_tabla(df):
    salida = df.copy()

    for col in salida.columns:
        if pd.api.types.is_datetime64_any_dtype(salida[col]):
            salida[col] = salida[col].dt.strftime("%Y-%m-%d")

    columnas_num = salida.select_dtypes(include=["number"]).columns

    for col in columnas_num:
        salida[col] = salida[col].round(2)

    return salida


def descargar_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# FUNCIONES DE GRÁFICOS
# ============================================================

def grafica_mensual_bateria(mensual, metrica):
    titulo_y = "Promedio diario mensual" if metrica == "Promedio diario mensual" else "Total mensual"

    base = alt.Chart(mensual).encode(
        x=alt.X("MES_LABEL:N", sort=list(mensual["MES_LABEL"]), title="Mes"),
        y=alt.Y("VALOR_GRAFICA:Q", title=titulo_y),
        tooltip=[
            alt.Tooltip("MES_LABEL:N", title="Mes"),
            alt.Tooltip("VALOR_GRAFICA:Q", title=titulo_y, format=",.2f"),
            alt.Tooltip("CAIDA_MENSUAL_ABS:Q", title="Variación vs mes previo", format=",.2f"),
            alt.Tooltip("CAIDA_MENSUAL_PCT:Q", title="Variación %", format=",.2f"),
            alt.Tooltip("DIAS_CON_DATO:Q", title="Días con dato")
        ]
    )

    barras = base.mark_bar()

    linea = base.mark_line(point=True).encode(
        y=alt.Y("VALOR_GRAFICA:Q", title=titulo_y)
    )

    caidas = mensual[mensual["ALERTA_CAIDA"] == "CAIDA FUERTE"].copy()

    puntos = alt.Chart(caidas).mark_point(
        size=140,
        filled=True
    ).encode(
        x=alt.X("MES_LABEL:N", sort=list(mensual["MES_LABEL"])),
        y="VALOR_GRAFICA:Q",
        tooltip=[
            alt.Tooltip("MES_LABEL:N", title="Mes"),
            alt.Tooltip("CAIDA_MENSUAL_ABS:Q", title="Caída", format=",.2f"),
            alt.Tooltip("CAIDA_MENSUAL_PCT:Q", title="Caída %", format=",.2f")
        ]
    )

    texto = alt.Chart(caidas).mark_text(
        dy=-12,
        fontSize=12
    ).encode(
        x=alt.X("MES_LABEL:N", sort=list(mensual["MES_LABEL"])),
        y="VALOR_GRAFICA:Q",
        text="CAIDA_LABEL:N"
    )

    return (barras + linea + puntos + texto).properties(height=420)


def grafica_lote_mensual(mensual_lote):
    base = alt.Chart(mensual_lote).encode(
        x=alt.X("MES_LABEL:N", sort=list(mensual_lote["MES_LABEL"]), title="Mes"),
        y=alt.Y("PROMEDIO_DIARIO_LOTE:Q", title="Promedio diario del lote"),
        tooltip=[
            alt.Tooltip("MES_LABEL:N", title="Mes"),
            alt.Tooltip("PROMEDIO_DIARIO_LOTE:Q", title="Promedio diario", format=",.2f"),
            alt.Tooltip("VAR_MENSUAL_ABS:Q", title="Variación vs mes previo", format=",.2f"),
            alt.Tooltip("VAR_MENSUAL_PCT:Q", title="Variación %", format=",.2f")
        ]
    )

    barras = base.mark_bar()
    linea = base.mark_line(point=True)

    caidas = mensual_lote[mensual_lote["VAR_MENSUAL_ABS"] < 0].copy()

    if not caidas.empty:
        umbral = caidas["VAR_MENSUAL_ABS"].quantile(0.25)
        caidas_fuertes = caidas[
            (caidas["VAR_MENSUAL_PCT"] <= -5) |
            (caidas["VAR_MENSUAL_ABS"] <= umbral)
        ].copy()
    else:
        caidas_fuertes = pd.DataFrame(columns=mensual_lote.columns)

    puntos = alt.Chart(caidas_fuertes).mark_point(
        size=140,
        filled=True
    ).encode(
        x=alt.X("MES_LABEL:N", sort=list(mensual_lote["MES_LABEL"])),
        y="PROMEDIO_DIARIO_LOTE:Q",
        tooltip=[
            alt.Tooltip("MES_LABEL:N", title="Mes"),
            alt.Tooltip("VAR_MENSUAL_ABS:Q", title="Caída", format=",.2f"),
            alt.Tooltip("VAR_MENSUAL_PCT:Q", title="Caída %", format=",.2f")
        ]
    )

    caidas_fuertes["LABEL"] = caidas_fuertes["VAR_MENSUAL_PCT"].apply(
        lambda x: "" if pd.isna(x) else f"{x:.1f}%"
    )

    texto = alt.Chart(caidas_fuertes).mark_text(
        dy=-12,
        fontSize=12
    ).encode(
        x=alt.X("MES_LABEL:N", sort=list(mensual_lote["MES_LABEL"])),
        y="PROMEDIO_DIARIO_LOTE:Q",
        text="LABEL:N"
    )

    return (barras + linea + puntos + texto).properties(height=420)


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("⛽ Análisis de gas por baterías Lote X")
st.caption("Sube el Excel y el sistema identificará las baterías más críticas, las caídas mensuales y el detalle por batería.")

archivo = st.file_uploader("Sube el Excel de gas por baterías", type=["xlsx"])

if archivo is None:
    st.info("Primero sube el Excel. Después se habilitarán los filtros de batería, año y periodo de análisis.")
    st.stop()

try:
    df, baterias, hoja_usada = cargar_excel_gas(archivo)
except Exception as e:
    st.error(f"No se pudo cargar el Excel: {e}")
    st.stop()

df_total = crear_total_lote(df, baterias)
largo = convertir_a_formato_largo(df, baterias)

fecha_min = df["Fecha"].min().date()
fecha_max = df["Fecha"].max().date()
anios = sorted(largo["AÑO"].dropna().unique().astype(int).tolist())

resumen_base, periodos_base = crear_resumen_criticidad(
    df=df,
    baterias=baterias,
    fecha_corte=fecha_max,
    ventana_dias=30
)

bateria_mas_critica = resumen_base.iloc[0]["BATERIA"] if not resumen_base.empty else baterias[0]

with st.sidebar:
    st.header("Filtros")

    fecha_corte = st.date_input(
        "Fecha de corte",
        value=fecha_max,
        min_value=fecha_min,
        max_value=fecha_max
    )

    ventana_dias = st.selectbox(
        "Ventana para criticidad",
        [7, 15, 30, 60, 90, 180, 365],
        index=2,
        format_func=lambda x: f"Últimos {x} días"
    )

    anio_sel = st.selectbox(
        "Año para gráficas mensuales",
        anios,
        index=len(anios) - 1
    )

    metrica_mensual = st.selectbox(
        "Métrica mensual",
        ["Promedio diario mensual", "Total mensual"],
        index=0
    )

    resumen_sidebar, _ = crear_resumen_criticidad(
        df=df,
        baterias=baterias,
        fecha_corte=fecha_corte,
        ventana_dias=ventana_dias
    )

    lista_baterias_ordenadas = resumen_sidebar["BATERIA"].tolist()
    indice_default = lista_baterias_ordenadas.index(bateria_mas_critica) if bateria_mas_critica in lista_baterias_ordenadas else 0

    bateria_sel = st.selectbox(
        "Batería para detalle",
        lista_baterias_ordenadas,
        index=indice_default
    )

resumen, periodos = crear_resumen_criticidad(
    df=df,
    baterias=baterias,
    fecha_corte=fecha_corte,
    ventana_dias=ventana_dias
)

mensual_bateria = calcular_mensual_bateria(
    largo=largo,
    bateria=bateria_sel,
    anio=anio_sel,
    metrica=metrica_mensual
)

mensual_lote = calcular_mensual_lote(df_total, anio_sel)

caidas_globales = calcular_caidas_mensuales_global(
    largo=largo,
    anio=anio_sel,
    metrica=metrica_mensual
)

# ============================================================
# PRIMERA VISTA
# ============================================================

st.subheader("Primera lectura de criticidad")

peor = resumen.iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Batería más crítica", str(peor["BATERIA"]))

with col2:
    st.metric("Nivel", str(peor["NIVEL"]))

with col3:
    st.metric("Promedio actual", f"{peor['PROM_ACTUAL']:,.2f}")

with col4:
    st.metric("Caída vs previo", f"{peor['CAIDA_ABS']:,.2f}", f"{peor['CAIDA_PCT']:,.2f}%")

with col5:
    st.metric("Score criticidad", f"{peor['SCORE_CRITICIDAD']:,.2f}")

st.warning(
    f"Según el criterio de criticidad, la batería más comprometida es {peor['BATERIA']}. "
    f"Su promedio actual es {peor['PROM_ACTUAL']:.2f}, contra {peor['PROM_PREVIO']:.2f} del periodo previo. "
    f"La variación es {peor['CAIDA_ABS']:.2f}, equivalente a {peor['CAIDA_PCT']:.2f}%."
)

st.caption(
    f"Periodo actual evaluado: {periodos['actual_ini'].date()} al {periodos['actual_fin'].date()}. "
    f"Periodo previo comparado: {periodos['previo_ini'].date()} al {periodos['previo_fin'].date()}."
)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ranking crítico",
    "Detalle por batería",
    "Promedio mensual del lote",
    "Caídas fuertes",
    "Validación"
])


# ============================================================
# TAB 1
# ============================================================

with tab1:
    st.subheader("Ranking de baterías críticas")

    columnas_ranking = [
        "NIVEL",
        "BATERIA",
        "PROM_ACTUAL",
        "PROM_PREVIO",
        "CAIDA_ABS",
        "CAIDA_PCT",
        "PROM_ULT_7D",
        "PROM_7D_PREVIO",
        "CAIDA_ABS_7D",
        "CAIDA_PCT_7D",
        "PROM_MISMO_PERIODO_AÑO_ANT",
        "CAIDA_ABS_YOY",
        "CAIDA_PCT_YOY",
        "ULTIMO_DIA",
        "VAR_PCT_ULTIMO_VS_PROM",
        "SCORE_CRITICIDAD"
    ]

    st.dataframe(
        formatear_tabla(resumen[columnas_ranking]),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Descargar ranking crítico",
        data=descargar_csv(formatear_tabla(resumen[columnas_ranking])),
        file_name="ranking_criticidad_gas_baterias.csv",
        mime="text/csv"
    )

    st.subheader("Top 10 por caída porcentual del periodo")
    top_pct = resumen[resumen["CAIDA_PCT"] < 0].sort_values("CAIDA_PCT", ascending=True).head(10)
    st.bar_chart(top_pct.set_index("BATERIA")["CAIDA_PCT"])

    st.subheader("Top 10 por caída absoluta del periodo")
    top_abs = resumen[resumen["CAIDA_ABS"] < 0].sort_values("CAIDA_ABS", ascending=True).head(10)
    st.bar_chart(top_abs.set_index("BATERIA")["CAIDA_ABS"])


# ============================================================
# TAB 2
# ============================================================

with tab2:
    st.subheader(f"Detalle mensual de la batería {bateria_sel}")

    if mensual_bateria.empty:
        st.info("No hay datos mensuales para la batería y año seleccionados.")
    else:
        st.altair_chart(
            grafica_mensual_bateria(mensual_bateria, metrica_mensual),
            use_container_width=True
        )

        caidas_bat = mensual_bateria[mensual_bateria["CAIDA_MENSUAL_ABS"] < 0].copy()
        caidas_bat = caidas_bat.sort_values(["CAIDA_MENSUAL_PCT", "CAIDA_MENSUAL_ABS"], ascending=[True, True])

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("Tabla mensual")
            cols_mes = [
                "MES_LABEL",
                "PROM_MENSUAL",
                "TOTAL_MENSUAL",
                "DIAS_CON_DATO",
                "CAIDA_MENSUAL_ABS",
                "CAIDA_MENSUAL_PCT",
                "ALERTA_CAIDA"
            ]
            st.dataframe(
                formatear_tabla(mensual_bateria[cols_mes]),
                use_container_width=True,
                hide_index=True
            )

        with col_b:
            st.write("Caídas más fuertes de esta batería")
            cols_caidas = [
                "MES_LABEL",
                "PROM_MENSUAL",
                "TOTAL_MENSUAL",
                "CAIDA_MENSUAL_ABS",
                "CAIDA_MENSUAL_PCT",
                "ALERTA_CAIDA"
            ]
            st.dataframe(
                formatear_tabla(caidas_bat[cols_caidas].head(10)),
                use_container_width=True,
                hide_index=True
            )

        ultimos_dias = df[["Fecha", bateria_sel]].dropna().copy()
        ultimos_dias = ultimos_dias[ultimos_dias["Fecha"].dt.year == anio_sel].copy()
        ultimos_dias = ultimos_dias.rename(columns={bateria_sel: "GAS"})
        ultimos_dias = ultimos_dias.set_index("Fecha")

        st.subheader(f"Tendencia diaria de {bateria_sel}")
        st.line_chart(ultimos_dias["GAS"])

        st.subheader(f"Promedio móvil de 7 días de {bateria_sel}")
        st.line_chart(ultimos_dias["GAS"].rolling(7).mean())


# ============================================================
# TAB 3
# ============================================================

with tab3:
    st.subheader(f"Promedio mensual del lote en {anio_sel}")

    if mensual_lote.empty:
        st.info("No hay datos del lote para el año seleccionado.")
    else:
        st.altair_chart(
            grafica_lote_mensual(mensual_lote),
            use_container_width=True
        )

        st.write("Tabla mensual del lote")
        cols_lote = [
            "MES_LABEL",
            "PROMEDIO_DIARIO_LOTE",
            "TOTAL_MENSUAL_LOTE",
            "DIAS_CON_DATO",
            "VAR_MENSUAL_ABS",
            "VAR_MENSUAL_PCT"
        ]
        st.dataframe(
            formatear_tabla(mensual_lote[cols_lote]),
            use_container_width=True,
            hide_index=True
        )

        ultimo_mes = mensual_lote.iloc[-1]
        st.info(
            f"Al último mes disponible en {anio_sel}, el promedio diario del lote es "
            f"{ultimo_mes['PROMEDIO_DIARIO_LOTE']:.2f}. "
            f"La variación contra el mes anterior es {ultimo_mes['VAR_MENSUAL_ABS']:.2f}, "
            f"equivalente a {ultimo_mes['VAR_MENSUAL_PCT']:.2f}%."
        )


# ============================================================
# TAB 4
# ============================================================

with tab4:
    st.subheader(f"Caídas fuertes mensuales en {anio_sel}")

    if caidas_globales.empty:
        st.info("No se encontraron caídas mensuales para el año seleccionado.")
    else:
        top_caidas = caidas_globales.head(20).copy()

        cols_global = [
            "BATERIA",
            "MES_LABEL",
            "PROM_MENSUAL",
            "TOTAL_MENSUAL",
            "VALOR_PREVIO",
            "VALOR",
            "CAIDA_ABS_MES",
            "CAIDA_PCT_MES",
            "DIAS_CON_DATO"
        ]

        st.dataframe(
            formatear_tabla(top_caidas[cols_global]),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Descargar caídas fuertes mensuales",
            data=descargar_csv(formatear_tabla(caidas_globales[cols_global])),
            file_name="caidas_fuertes_mensuales_baterias.csv",
            mime="text/csv"
        )

        graf_caidas = top_caidas.copy()
        graf_caidas["BATERIA_MES"] = graf_caidas["BATERIA"].astype(str) + " " + graf_caidas["MES_LABEL"].astype(str)

        st.subheader("Top 20 caídas porcentuales")
        st.bar_chart(graf_caidas.set_index("BATERIA_MES")["CAIDA_PCT_MES"])

        st.subheader("Top 20 caídas absolutas")
        top_abs_mes = caidas_globales.sort_values("CAIDA_ABS_MES", ascending=True).head(20).copy()
        top_abs_mes["BATERIA_MES"] = top_abs_mes["BATERIA"].astype(str) + " " + top_abs_mes["MES_LABEL"].astype(str)
        st.bar_chart(top_abs_mes.set_index("BATERIA_MES")["CAIDA_ABS_MES"])


# ============================================================
# TAB 5
# ============================================================

with tab5:
    st.subheader("Validación del archivo cargado")

    st.write(f"Hoja usada: {hoja_usada}")
    st.write(f"Rango de fechas detectado: {fecha_min} al {fecha_max}")
    st.write(f"Número de baterías detectadas: {len(baterias)}")

    st.write("Baterías detectadas")
    st.dataframe(
        pd.DataFrame({"BATERIA": baterias}),
        use_container_width=True,
        hide_index=True
    )

    st.write("Primeras filas leídas")
    columnas_muestra = ["Fecha"] + baterias[:12]
    st.dataframe(
        df[columnas_muestra].head(20),
        use_container_width=True,
        hide_index=True
    )
