import io
import re
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Análisis de Gas por Baterías",
    page_icon="⛽",
    layout="wide"
)


# ============================================================
# FUNCIONES
# ============================================================

def limpiar_nombre_columna(col):
    return str(col).strip()


def cargar_excel_gas(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    hoja = xls.sheet_names[0]

    # Si existe esta hoja, usarla directamente.
    for h in xls.sheet_names:
        if h.strip().lower() == "tbl_produccion_gas_baterias":
            hoja = h
            break

    df = pd.read_excel(xls, sheet_name=hoja)
    df.columns = [limpiar_nombre_columna(c) for c in df.columns]

    if "Fecha" not in df.columns:
        raise ValueError("No encontré la columna 'Fecha'. Revisa que el Excel tenga una columna llamada Fecha.")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    columnas_excluir = ["Fecha", "Comentarios", "ID"]
    baterias = [c for c in df.columns if c not in columnas_excluir]

    for c in baterias:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Limpieza de filas inválidas o vacías.
    df = df[df["Fecha"].notna()].copy()
    df = df[df["Fecha"].dt.year >= 2020].copy()
    df = df[df[baterias].notna().any(axis=1)].copy()
    df = df.sort_values("Fecha").reset_index(drop=True)

    return df, baterias, hoja


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

    ultimo_dia = df[df["Fecha"] == fecha_corte][baterias]
    if ultimo_dia.empty:
        ultimo = pd.Series(index=baterias, dtype=float)
    else:
        ultimo = ultimo_dia.iloc[-1]

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
    resumen["CAIDA_%"] = np.where(
        resumen["PROM_PREVIO"] > 0,
        resumen["CAIDA_ABS"] / resumen["PROM_PREVIO"] * 100,
        np.nan
    )

    resumen["CAIDA_ABS_7D"] = resumen["PROM_ULT_7D"] - resumen["PROM_7D_PREVIO"]
    resumen["CAIDA_%_7D"] = np.where(
        resumen["PROM_7D_PREVIO"] > 0,
        resumen["CAIDA_ABS_7D"] / resumen["PROM_7D_PREVIO"] * 100,
        np.nan
    )

    resumen["CAIDA_ABS_YOY"] = resumen["PROM_ACTUAL"] - resumen["PROM_MISMO_PERIODO_AÑO_ANT"]
    resumen["CAIDA_%_YOY"] = np.where(
        resumen["PROM_MISMO_PERIODO_AÑO_ANT"] > 0,
        resumen["CAIDA_ABS_YOY"] / resumen["PROM_MISMO_PERIODO_AÑO_ANT"] * 100,
        np.nan
    )

    resumen["VAR_ULTIMO_VS_PROM"] = resumen["ULTIMO_DIA"] - resumen["PROM_ACTUAL"]
    resumen["VAR_%_ULTIMO_VS_PROM"] = np.where(
        resumen["PROM_ACTUAL"] > 0,
        resumen["VAR_ULTIMO_VS_PROM"] / resumen["PROM_ACTUAL"] * 100,
        np.nan
    )

    # Score de criticidad:
    # Más peso a caída del periodo actual vs periodo previo.
    # También considera caída semanal, caída interanual y último día debajo del promedio.
    resumen["SCORE_CRITICIDAD"] = (
        (-resumen["CAIDA_%"].clip(upper=0).fillna(0)) * 0.45 +
        (-resumen["CAIDA_%_7D"].clip(upper=0).fillna(0)) * 0.25 +
        (-resumen["CAIDA_%_YOY"].clip(upper=0).fillna(0)) * 0.20 +
        (-resumen["VAR_%_ULTIMO_VS_PROM"].clip(upper=0).fillna(0)) * 0.10
    )

    def nivel(row):
        if row["CAIDA_%"] <= -20 and row["CAIDA_%_7D"] <= -10:
            return "CRÍTICO"
        if row["CAIDA_%"] <= -12:
            return "ALTO"
        if row["CAIDA_%_7D"] <= -20:
            return "ALTO"
        if row["CAIDA_%"] <= -5:
            return "MEDIO"
        if row["CAIDA_%_7D"] <= -10:
            return "MEDIO"
        return "NORMAL"

    resumen["NIVEL"] = resumen.apply(nivel, axis=1)

    orden_nivel = {
        "CRÍTICO": 1,
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


def formatear_tabla(df):
    salida = df.copy()

    columnas_redondear = [
        "PROM_ACTUAL", "PROM_PREVIO", "PROM_MISMO_PERIODO_AÑO_ANT",
        "SUM_ACTUAL", "SUM_PREVIO",
        "PROM_ULT_7D", "PROM_7D_PREVIO", "ULTIMO_DIA",
        "CAIDA_ABS", "CAIDA_%", "CAIDA_ABS_7D", "CAIDA_%_7D",
        "CAIDA_ABS_YOY", "CAIDA_%_YOY",
        "VAR_ULTIMO_VS_PROM", "VAR_%_ULTIMO_VS_PROM",
        "SCORE_CRITICIDAD"
    ]

    for c in columnas_redondear:
        if c in salida.columns:
            salida[c] = pd.to_numeric(salida[c], errors="coerce").round(2)

    return salida


def descargar_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# INTERFAZ
# ============================================================

st.title("⛽ Análisis de caída de gas por baterías")
st.caption("Carga el Excel de gas por baterías y el sistema identifica las baterías con mayor caída y mayor criticidad.")

archivo = st.file_uploader("Sube el Excel: Gas por baterías Lote X.xlsx", type=["xlsx"])

if archivo is None:
    st.info("Sube el Excel para comenzar el análisis.")
    st.stop()

try:
    df, baterias, hoja = cargar_excel_gas(archivo)
except Exception as e:
    st.error(f"No se pudo cargar el Excel: {e}")
    st.stop()

fecha_min = df["Fecha"].min().date()
fecha_max = df["Fecha"].max().date()

with st.sidebar:
    st.header("Filtros de análisis")

    fecha_corte = st.date_input(
        "Fecha de corte",
        value=fecha_max,
        min_value=fecha_min,
        max_value=fecha_max
    )

    ventana_dias = st.selectbox(
        "Periodo actual a evaluar",
        [7, 15, 30, 60, 90, 180, 365],
        index=2,
        format_func=lambda x: f"Últimos {x} días"
    )

    modo_baterias = st.radio(
        "Baterías a mostrar",
        ["Todas", "Solo críticas, altas y medias", "Solo críticas y altas"]
    )

resumen, periodos = crear_resumen_criticidad(df, baterias, fecha_corte, ventana_dias)

tabla = resumen.copy()

if modo_baterias == "Solo críticas, altas y medias":
    tabla = tabla[tabla["NIVEL"].isin(["CRÍTICO", "ALTO", "MEDIO"])]
elif modo_baterias == "Solo críticas y altas":
    tabla = tabla[tabla["NIVEL"].isin(["CRÍTICO", "ALTO"])]

# Total del lote
df["TOTAL_GAS"] = df[baterias].sum(axis=1, skipna=True)

mask_actual = (df["Fecha"] >= periodos["actual_ini"]) & (df["Fecha"] <= periodos["actual_fin"])
mask_previo = (df["Fecha"] >= periodos["previo_ini"]) & (df["Fecha"] <= periodos["previo_fin"])

prom_total_actual = df.loc[mask_actual, "TOTAL_GAS"].mean()
prom_total_previo = df.loc[mask_previo, "TOTAL_GAS"].mean()
caida_total_abs = prom_total_actual - prom_total_previo
caida_total_pct = caida_total_abs / prom_total_previo * 100 if prom_total_previo > 0 else 0

criticas = int((resumen["NIVEL"] == "CRÍTICO").sum())
altas = int((resumen["NIVEL"] == "ALTO").sum())
medias = int((resumen["NIVEL"] == "MEDIO").sum())

st.subheader("Resumen del lote")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Promedio actual total", f"{prom_total_actual:,.2f}")

with c2:
    st.metric("Promedio previo total", f"{prom_total_previo:,.2f}")

with c3:
    st.metric("Caída total", f"{caida_total_abs:,.2f}", f"{caida_total_pct:,.2f}%")

with c4:
    st.metric("Baterías críticas/altas", f"{criticas + altas}")

with c5:
    st.metric("Baterías medias", f"{medias}")

st.caption(
    f"Periodo actual: {periodos['actual_ini'].date()} al {periodos['actual_fin'].date()} | "
    f"Periodo previo: {periodos['previo_ini'].date()} al {periodos['previo_fin'].date()}"
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Criticidad por batería",
    "Tendencia diaria",
    "Top caídas",
    "Validación de datos"
])

with tab1:
    st.subheader("Ranking de criticidad")

    columnas_mostrar = [
        "NIVEL",
        "BATERIA",
        "PROM_ACTUAL",
        "PROM_PREVIO",
        "CAIDA_ABS",
        "CAIDA_%",
        "PROM_ULT_7D",
        "PROM_7D_PREVIO",
        "CAIDA_ABS_7D",
        "CAIDA_%_7D",
        "PROM_MISMO_PERIODO_AÑO_ANT",
        "CAIDA_ABS_YOY",
        "CAIDA_%_YOY",
        "ULTIMO_DIA",
        "VAR_%_ULTIMO_VS_PROM",
        "SCORE_CRITICIDAD"
    ]

    st.dataframe(
        formatear_tabla(tabla[columnas_mostrar]),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Descargar ranking de criticidad",
        data=descargar_csv(formatear_tabla(tabla[columnas_mostrar])),
        file_name="ranking_criticidad_gas_baterias.csv",
        mime="text/csv"
    )

    peor = resumen.iloc[0]
    st.warning(
        f"La batería más crítica según el score es {peor['BATERIA']}. "
        f"Promedio actual: {peor['PROM_ACTUAL']:.2f}; "
        f"promedio previo: {peor['PROM_PREVIO']:.2f}; "
        f"variación: {peor['CAIDA_ABS']:.2f} ({peor['CAIDA_%']:.2f}%)."
    )

with tab2:
    st.subheader("Tendencia diaria de gas")

    baterias_seleccionadas = st.multiselect(
        "Selecciona baterías",
        options=baterias,
        default=list(resumen.head(5)["BATERIA"])
    )

    fecha_ini_graf = st.date_input(
        "Fecha inicial del gráfico",
        value=max(fecha_min, (pd.to_datetime(fecha_corte) - pd.Timedelta(days=180)).date()),
        min_value=fecha_min,
        max_value=fecha_max
    )

    graf = df[(df["Fecha"] >= pd.to_datetime(fecha_ini_graf)) & (df["Fecha"] <= pd.to_datetime(fecha_corte))].copy()

    if baterias_seleccionadas:
        graf_linea = graf.set_index("Fecha")[baterias_seleccionadas]
        st.line_chart(graf_linea)

        st.write("Promedio móvil de 7 días")
        st.line_chart(graf_linea.rolling(7).mean())

with tab3:
    st.subheader("Top caídas")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Mayor caída absoluta")
        top_abs = resumen.sort_values("CAIDA_ABS", ascending=True).head(10)
        st.dataframe(
            formatear_tabla(top_abs[["BATERIA", "PROM_ACTUAL", "PROM_PREVIO", "CAIDA_ABS", "CAIDA_%", "NIVEL"]]),
            use_container_width=True,
            hide_index=True
        )
        st.bar_chart(top_abs.set_index("BATERIA")["CAIDA_ABS"])

    with col2:
        st.write("Mayor caída porcentual")
        top_pct = resumen.sort_values("CAIDA_%", ascending=True).head(10)
        st.dataframe(
            formatear_tabla(top_pct[["BATERIA", "PROM_ACTUAL", "PROM_PREVIO", "CAIDA_ABS", "CAIDA_%", "NIVEL"]]),
            use_container_width=True,
            hide_index=True
        )
        st.bar_chart(top_pct.set_index("BATERIA")["CAIDA_%"])

    st.write("Menor producción promedio actual")
    menor_prod = resumen.sort_values("PROM_ACTUAL", ascending=True).head(10)
    st.dataframe(
        formatear_tabla(menor_prod[["BATERIA", "PROM_ACTUAL", "PROM_PREVIO", "CAIDA_ABS", "CAIDA_%", "NIVEL"]]),
        use_container_width=True,
        hide_index=True
    )

with tab4:
    st.subheader("Validación de datos cargados")

    st.write(f"Hoja usada: {hoja}")
    st.write(f"Rango de fechas detectado: {fecha_min} al {fecha_max}")
    st.write(f"Número de baterías detectadas: {len(baterias)}")

    st.write("Baterías detectadas:")
    st.dataframe(pd.DataFrame({"BATERIA": baterias}), use_container_width=True, hide_index=True)

    st.write("Primeras filas del archivo limpio:")
    cols_muestra = ["Fecha"] + baterias[:10]
    st.dataframe(df[cols_muestra].head(20), use_container_width=True, hide_index=True)
