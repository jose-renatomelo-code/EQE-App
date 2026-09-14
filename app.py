import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import io

script_dir = Path(__file__).resolve().parent

st.set_page_config(
    page_title="EQE Analyser",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_solar_spectrum(am15g_path):
    df = pd.read_excel(am15g_path, header=1)
    wl = pd.to_numeric(df.iloc[:, 0].astype(str).str.replace(",", ".").str.strip(), errors="coerce")
    irr = pd.to_numeric(df.iloc[:, 2].astype(str).str.replace(",", ".").str.strip(),
                        errors="coerce")
    valid = wl.notnull() & irr.notnull()
    return wl[valid].values.astype(float), irr[valid].values.astype(float)


def parse_data(uploaded_file):
    filename = uploaded_file.name
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    start_idx = None
    for i, l in enumerate(lines):
        tokens = l.strip().upper().split()
        if len(tokens) >= 2 and tokens[0] == "START" and tokens[1].startswith("DATA"):
            start_idx = i + 1
            break
        if len(tokens) >= 2 and tokens[0] == "END" and tokens[1].startswith("HEADER"):
            start_idx = i + 1
            break

    if start_idx is None:
        for i, l in enumerate(lines):
            if any(kw in l.lower() for kw in ("wavelength", "eqe", "sr(a")):
                start_idx = i
                break

    data_lines = lines[start_idx:] if start_idx is not None else lines
    clean_lines = [l for l in data_lines if l.strip() and not l.strip().startswith("#")]
    data_block = "\n".join(clean_lines)

    df = None
    strategies = [
        dict(sep=None, engine="python"),
        dict(sep=r"\s+", engine="python"),
        dict(sep="\t", engine="c"),
        dict(sep=",", engine="c"),
        dict(sep=";", engine="c"),
    ]
    for kwargs in strategies:
        try:
            df = pd.read_table(
                io.StringIO(data_block),
                on_bad_lines="skip",
                **kwargs,
            )
            if df.shape[1] >= 3:
                break
        except Exception:
            df = None

    if df is None or df.shape[1] < 3:
        st.error(f"Could not parse file: {filename}. Check the file format.")
        st.stop()


    def find_col(candidates, cols):
        cols_lower = {c.lower(): c for c in cols}
        for pattern in candidates:
            matches = [orig for low, orig in cols_lower.items() if pattern in low]
            if matches:
                return matches[0]
        return None

    col_wl  = find_col(["wavelength", "wl(", "wl "], df.columns)
    col_eqe = find_col(["eqe"], df.columns)
    col_sr  = find_col(["sr(", "sr "], df.columns)

    missing = [name for name, col in [("Wavelength(nm)", col_wl), ("EQE(%)", col_eqe), ("SR(A/W)", col_sr)] if col is None]
    if missing:
        st.error(f"**{filename}**: could not find columns {missing}.\n\nColumns found: `{list(df.columns)}`")
        st.stop()

    wavelength_nm = pd.to_numeric(df[col_wl], errors="coerce").values.astype(float)
    eqe = pd.to_numeric(df[col_eqe], errors="coerce").values.astype(float)
    sr = pd.to_numeric(df[col_sr], errors="coerce").values.astype(float)

    valid = np.isfinite(eqe) & np.isfinite(sr) & np.isfinite(wavelength_nm)
    eqe = eqe[valid]
    sr = sr[valid]
    wavelength_nm = wavelength_nm[valid]

    am15g_path = script_dir / "AM0AM1_5.xls"
    wl_am, irr_am = load_solar_spectrum(am15g_path)

    irr = np.interp(wavelength_nm, wl_am, irr_am)
    j_lambda = sr * irr * 0.1

    cum_jsc = np.zeros_like(wavelength_nm)
    for i in range(1, len(wavelength_nm)):
        cum_jsc[i] = cum_jsc[i - 1] + 0.5 * (j_lambda[i - 1] + j_lambda[i]) * (wavelength_nm[i] - wavelength_nm[i - 1])

    jsc_integrated = cum_jsc[-1]
    idx_max = int(np.argmax(eqe))
    eqe_max = eqe[idx_max]
    sr_max = sr[idx_max]
    wl_max = wavelength_nm[idx_max]

    df_results = pd.DataFrame(
        {
            "filename": filename,
            "Wavelength(nm)": wavelength_nm,
            "EQE(%)": eqe,
            "SR(A/W)": sr,
            "AM1.5G(W/m2/nm)": irr,
            "j_lambda(mA/cm2/nm)": j_lambda,
            "Cumulative_Jsc(mA/cm²)": cum_jsc,
        }
    )

    df_metrics = pd.DataFrame({
        "filename": [filename],
        "Jsc(mA/cm2)": [jsc_integrated],
        "EQE_MAX": [eqe_max],
        "SR_MAX": [sr_max],
        "WL_MAX": [wl_max],
    })
    return df_results, df_metrics


with st.sidebar:
    st.title("EQE Analyser")
    st.markdown("---")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    def clean_all_uploads():
        st.session_state["uploaded_files"] = []
        st.session_state["uploader_key"] += 1

    new_files = st.file_uploader(
        "Upload .qsdat files",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if new_files:
        st.session_state["uploaded_files"] = new_files

    uploaded_files = st.session_state["uploaded_files"]

    st.button("Clean Uploads", on_click=clean_all_uploads)


if not uploaded_files:
    st.info("Welcome! Please upload your EQE .qsdat files in the sidebar to get started.")
    st.stop()

all_data = {"results": [], "metrics": []}
for f in uploaded_files:
    results, metrics = parse_data(f)
    all_data["results"].append(results)
    all_data["metrics"].append(metrics)

all_results_df = pd.concat(all_data["results"], ignore_index=True)
all_metrics_df = pd.concat(all_data["metrics"], ignore_index=True)

tab1, tab2 = st.tabs(["Dashboard", "Statistics"])
with tab1:
    st.subheader("EQE Curves")
    fig_eqe = make_subplots(specs=[[{"secondary_y": True}]])
    fig_eqe.update_yaxes(title_text="EQE(%)", secondary_y=False)
    fig_eqe.update_yaxes(title_text="Jsc(mA/cm²)", secondary_y=True)
    colors = px.colors.qualitative.Plotly

    for i, d in enumerate(all_data["results"]):
        color = colors[i % len(colors)]
        name = d["filename"].iloc[0]

        fig_eqe.add_trace(
            go.Scatter(x=d["Wavelength(nm)"], y=d["EQE(%)"], mode='lines+markers', name=name, line=dict(color=color)),
            secondary_y=False
        )

        fig_eqe.add_trace(
            go.Scatter(x=d["Wavelength(nm)"], y=d["Cumulative_Jsc(mA/cm²)"], mode='lines', name=f"{name} Jsc", line=dict(color=color, dash="dot")),
            secondary_y=True
        )

    fig_eqe.update_layout(
        title="EQE curves",
        xaxis_title="Wavelength(nm)",
        hovermode="x unified",
        template="seaborn",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_eqe, use_container_width=True)

    def safe_fmt(val, fmt):
        return fmt.format(val) if not (isinstance(val, float) and np.isnan(val)) else "N/A"

    st.markdown("### 🏆 Best Performing Cell")
    m1, m2, m3, m4 = st.columns(4)
    best_idx = all_metrics_df["Jsc(mA/cm2)"].idxmax()
    best_cell = all_metrics_df.loc[best_idx]
    m1.metric("EQE", safe_fmt(best_cell["EQE_MAX"], "{:.2f}%"))
    m2.metric("Jsc", safe_fmt(best_cell["Jsc(mA/cm2)"], "{:.3f} mA/cm²"))
    m3.metric("Max_wavelength", safe_fmt(best_cell["WL_MAX"], "{:.2f} nm"))
    m4.metric("Max_SR", safe_fmt(best_cell["SR_MAX"], "{:.1f}A/W"))

with tab2:
    st.subheader("Data")
    st.dataframe(all_metrics_df)

    st.markdown("### Statistics")
    numeric_cols = all_metrics_df.select_dtypes(include="number").columns
    stats = all_metrics_df[numeric_cols].describe().T
    st.dataframe(stats.style.format("{:.3f}"), use_container_width=True)

    st.markdown("### Distributions")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("**Boxplots**")
        param_to_plot = st.selectbox("Select Parameter", numeric_cols.tolist())
        fig_box = px.box(all_metrics_df, y=param_to_plot, points="all", title=f"{param_to_plot} Distribution")
        st.plotly_chart(fig_box, use_container_width=True)

    with d_col2:
        st.markdown("**Correlations**")
        fig_corr = px.scatter_matrix(all_metrics_df, dimensions=numeric_cols.tolist(), title="Parameter Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)

    csv = all_results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results",
        data=csv,
        file_name="eqe_results.csv",
        mime="text/csv",
    )