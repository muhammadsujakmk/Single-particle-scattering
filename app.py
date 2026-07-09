from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from mie_core import (
    MaterialData,
    calculate_mie_sphere,
    load_material_file,
    material_dataframe,
    mode_label,
    spectra_dataframe,
    wiscombe_order,
)

APP_DIR = Path(__file__).resolve().parent
MATERIAL_DIR = APP_DIR / "materials"

BUILTIN_MATERIALS = {
    "Si — Green": {"file": "Si_Green.txt", "unit": "nm", "format": "n_k"},
    "Si — Aspnes": {"file": "Si_Aspnes.txt", "unit": "um", "format": "n_k"},
    "Si — Palik": {"file": "Si_Palik.txt", "unit": "um", "format": "n_k"},
    "Si — Palik (nm table)": {"file": "Si_Palik2.txt", "unit": "nm", "format": "n_k"},
    "Au — Johnson & Christy": {"file": "Au_JC.txt", "unit": "um", "format": "n_k"},
    "Ag — Johnson & Christy": {"file": "Ag_JC.txt", "unit": "um", "format": "n_k"},
    "Au — SHARC table": {"file": "Au_sharck.txt", "unit": "nm", "format": "n_k"},
}

MODE_OPTIONS = [
    ("ED", "E", 1),
    ("MD", "M", 1),
    ("EQ", "E", 2),
    ("MQ", "M", 2),
    ("EO", "E", 3),
    ("MO", "M", 3),
]


def inject_css() -> None:
    """Apply the bright, three-panel arrangement requested in the blueprint."""
    st.markdown(
        """
        <style>
        .stApp { background: #F8FAFD; color: #172033; }
        .block-container { max-width: 1580px; padding-top: 1.1rem; padding-bottom: 1.3rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #ABC0DC !important;
            border-radius: 0.20rem !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div { background: #FFFFFF !important; }
        [data-testid="stSidebar"] { background: #F8FAFD !important; }
        .hero-title {
            color: #111827; font-size: 2.00rem; font-weight: 700; line-height: 1.16;
            text-align: center; margin: 0.15rem 0 0.20rem 0;
        }
        .hero-subtitle {
            color: #455468; font-size: 1.03rem; line-height: 1.40;
            text-align: center; margin: 0;
        }
        .panel-title {
            color: #172033; font-size: 1.08rem; font-weight: 700;
            margin: 0 0 0.15rem 0;
        }
        .panel-note { color: #5A687A; font-size: 0.79rem; margin-bottom: 0.35rem; }
        .control-heading { color: #172033; font-size: 1.36rem; font-weight: 700; margin: 0 0 0.65rem 0; }
        .section-heading { color: #24466D; font-size: 0.88rem; font-weight: 750; letter-spacing: 0.025em; margin: 0.78rem 0 0.25rem 0; }
        .small-note { color: #5B6678; font-size: 0.76rem; line-height: 1.42; }
        div[data-testid="stCheckbox"] { margin-bottom: 0.10rem; }
        div[data-testid="stCheckbox"] label { font-size: 0.92rem; }
        div[data-testid="stRadio"] label { font-size: 0.88rem; }
        div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button {
            border: 1px solid #2A73B9 !important;
        }
        div[data-testid="stDownloadButton"] button { color: #174A7A !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_builtin_material(material_name: str) -> MaterialData:
    spec = BUILTIN_MATERIALS[material_name]
    return load_material_file(
        MATERIAL_DIR / spec["file"],
        wavelength_unit=spec["unit"],
        optical_format=spec["format"],
        name=material_name,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_material(
    raw_bytes: bytes,
    filename: str,
    wavelength_unit: str,
    optical_format: str,
) -> MaterialData:
    return load_material_file(
        BytesIO(raw_bytes),
        wavelength_unit=wavelength_unit,  # type: ignore[arg-type]
        optical_format=optical_format,  # type: ignore[arg-type]
        name=filename,
    )


def constant_material(name: str, n_value: float, k_value: float, lo_nm: float, hi_nm: float) -> MaterialData:
    """Create a two-point, constant optical-constant material over the selected scan range."""
    lo_nm = max(0.1, float(lo_nm))
    hi_nm = max(lo_nm + 1e-6, float(hi_nm))
    return MaterialData(
        wavelength_nm=np.array([lo_nm, hi_nm], dtype=float),
        n=np.array([n_value, n_value], dtype=float),
        k=np.array([k_value, k_value], dtype=float),
        name=name,
    )


def base_figure(height: int, y_title: str, x_title: str = "Wavelength (nm)") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=66, r=26, t=34, b=50),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E4EAF2", zeroline=False, linecolor="#AFC1D8")
    fig.update_yaxes(showgrid=True, gridcolor="#E4EAF2", zeroline=False, linecolor="#AFC1D8")
    return fig


def _mode_values(result: dict, kind: str, order: int, display_mode: str) -> np.ndarray:
    idx = order - 1
    field = "sca_e_m2" if kind == "E" else "sca_m_m2"
    values = np.asarray(result[field])[idx]
    if display_mode == "Efficiency, Q":
        return values / float(result["area_m2"])
    return values * 1e18


def total_spectra_figure(
    result: dict,
    display_mode: str,
    selected_totals: list[str],
    selected_modes: list[tuple[str, str, int]],
) -> go.Figure:
    wavelength = np.asarray(result["wavelength_nm"])
    if display_mode == "Efficiency, Q":
        total_curves = {
            "Qsca": np.asarray(result["q_sca"]),
            "Qabs": np.asarray(result["q_abs"]),
            "Qext": np.asarray(result["q_ext"]),
        }
        y_label = "Efficiency, Q"
    else:
        total_curves = {
            "Qsca": np.asarray(result["c_sca_m2"]) * 1e18,
            "Qabs": np.asarray(result["c_abs_m2"]) * 1e18,
            "Qext": np.asarray(result["c_ext_m2"]) * 1e18,
        }
        y_label = "Cross section (nm²)"

    fig = base_figure(386, y_label)
    line_widths = {"Qsca": 2.7, "Qabs": 2.5, "Qext": 2.5}
    for label in selected_totals:
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=total_curves[label],
                mode="lines",
                name=label,
                line=dict(width=line_widths[label]),
            )
        )

    for mode_name, kind, order in selected_modes:
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=_mode_values(result, kind, order, display_mode),
                mode="lines",
                name=f"{mode_name} (sca)",
                line=dict(width=1.75, dash="dot"),
            )
        )

    if not selected_totals and not selected_modes:
        fig.add_annotation(
            text="Select at least one total spectrum or multipole mode in the input panel.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#5A687A"),
        )
    return fig


def coefficient_figure(result: dict, selected_modes: list[tuple[str, str, int]]) -> go.Figure:
    """Show |a_l| / |b_l| and principal phase on a shared wavelength axis with dual y axes."""
    wavelength = np.asarray(result["wavelength_nm"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        height=372,
        margin=dict(l=66, r=72, t=34, b=50),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )

    for mode_name, kind, order in selected_modes:
        idx = order - 1
        coefficient = np.asarray(result["a"])[idx] if kind == "E" else np.asarray(result["b"])[idx]
        amplitude = np.abs(coefficient)
        phase_deg = np.angle(coefficient, deg=True)
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=amplitude,
                mode="lines",
                name=f"{mode_name} amplitude",
                line=dict(width=2.05),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=phase_deg,
                mode="lines",
                name=f"{mode_name} phase",
                line=dict(width=1.35, dash="dash"),
            ),
            secondary_y=True,
        )

    fig.update_xaxes(title_text="Wavelength (nm)", showgrid=True, gridcolor="#E4EAF2", zeroline=False, linecolor="#AFC1D8")
    fig.update_yaxes(
        title_text="Mie-coefficient amplitude",
        range=[0, 1.05],
        showgrid=True,
        gridcolor="#E4EAF2",
        zeroline=False,
        linecolor="#AFC1D8",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Phase (degree)",
        range=[-180, 180],
        tickvals=[-180, -90, 0, 90, 180],
        showgrid=False,
        zeroline=False,
        linecolor="#AFC1D8",
        secondary_y=True,
    )

    if not selected_modes:
        fig.add_annotation(
            text="Select one or more modes (ED, MD, EQ, MQ, EO, or MO) to display amplitude and phase.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#5A687A"),
        )
    return fig


def initial_defaults() -> None:
    defaults = {
        "ri_source": "Built-in library",
        "material_name": "Si — Green",
        "wavelength_min": 400.0,
        "wavelength_max": 1000.0,
        "samples": 601,
        "sphere_diameter": 200.0,
        "n_medium": 1.0,
        "user_n": 3.5,
        "user_k": 0.0,
        "truncation_method": "Automatic (Wiscombe)",
        "manual_order": 4,
        "display_mode": "Efficiency, Q",
        "show_qsca": True,
        "show_qabs": True,
        "show_qext": True,
        "mode_ED": True,
        "mode_MD": True,
        "mode_EQ": True,
        "mode_MQ": True,
        "mode_EO": False,
        "mode_MO": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def selected_modes_from_state() -> list[tuple[str, str, int]]:
    return [(label, kind, order) for label, kind, order in MODE_OPTIONS if st.session_state.get(f"mode_{label}", False)]


def selected_totals_from_state() -> list[str]:
    selected: list[str] = []
    if st.session_state.get("show_qsca", False):
        selected.append("Qsca")
    if st.session_state.get("show_qabs", False):
        selected.append("Qabs")
    if st.session_state.get("show_qext", False):
        selected.append("Qext")
    return selected


def build_controls() -> dict:
    with st.container(border=True):
        st.markdown('<div class="control-heading">Parameter control and input</div>', unsafe_allow_html=True)
        st.caption("Exact Mie scattering by a homogeneous isotropic sphere.")

        st.markdown('<div class="section-heading">Refractive-index data</div>', unsafe_allow_html=True)
        source = st.radio(
            "Data source",
            ["Built-in library", "Import RI", "Define RI by user"],
            key="ri_source",
            label_visibility="collapsed",
        )

        uploaded_file = None
        upload_unit = "nm"
        upload_format = "n_k"
        material_name = "Custom material"

        if source == "Built-in library":
            material_name = st.selectbox("Material", list(BUILTIN_MATERIALS), key="material_name")
        elif source == "Import RI":
            uploaded_file = st.file_uploader("Import RI file (.txt, .csv, .dat)", type=["txt", "csv", "dat"])
            upload_unit = st.selectbox("Wavelength unit in imported file", ["nm", "um"], index=0)
            upload_format_choice = st.radio("Columns 2–3 represent", ["n and k", "ε₁ and ε₂"], horizontal=True)
            upload_format = "n_k" if upload_format_choice == "n and k" else "epsilon"
            st.caption("Expected columns: wavelength, n, k — or wavelength, ε₁, ε₂. Lines beginning with # are ignored.")
        else:
            custom_cols = st.columns(2)
            with custom_cols[0]:
                user_n = st.number_input("User-defined n", min_value=0.0, max_value=30.0, step=0.01, key="user_n")
            with custom_cols[1]:
                user_k = st.number_input("User-defined k", min_value=0.0, max_value=30.0, step=0.01, key="user_k")
            material_name = f"User RI: n = {float(user_n):.3g}, k = {float(user_k):.3g}"

        st.markdown('<div class="section-heading">Geometry and spectral range</div>', unsafe_allow_html=True)
        n_medium = st.number_input("Medium refractive index", min_value=1.0, max_value=5.0, step=0.01, key="n_medium")
        sphere_diameter_nm = st.number_input("Diameter of sphere (nm)", min_value=1.0, max_value=10000.0, step=1.0, key="sphere_diameter")
        wave_cols = st.columns(2)
        with wave_cols[0]:
            wavelength_min = st.number_input("Minimum wavelength (nm)", min_value=1.0, max_value=100000.0, step=1.0, key="wavelength_min")
        with wave_cols[1]:
            wavelength_max = st.number_input("Maximum wavelength (nm)", min_value=1.0, max_value=100000.0, step=1.0, key="wavelength_max")
        samples = st.slider("Number of wavelength points", min_value=101, max_value=2501, step=50, key="samples")

        st.markdown('<div class="section-heading">Total spectra selection</div>', unsafe_allow_html=True)
        total_cols = st.columns(3)
        with total_cols[0]:
            st.checkbox("Qsca", key="show_qsca")
        with total_cols[1]:
            st.checkbox("Qabs", key="show_qabs")
        with total_cols[2]:
            st.checkbox("Qext", key="show_qext")

        st.markdown('<div class="section-heading">Modes selection</div>', unsafe_allow_html=True)
        mode_columns = st.columns(2)
        for idx, (label, _, _) in enumerate(MODE_OPTIONS):
            with mode_columns[idx % 2]:
                st.checkbox(label, key=f"mode_{label}")

        with st.expander("Advanced calculation settings", expanded=False):
            truncation_method = st.radio(
                "Maximum multipole order",
                ["Automatic (Wiscombe)", "Manual"],
                key="truncation_method",
                horizontal=True,
            )
            manual_order = 4
            if truncation_method == "Manual":
                manual_order = st.slider("Manual maximum order, ℓ", 1, 60, key="manual_order")
            display_mode = st.radio(
                "Vertical axis for the total spectrum",
                ["Efficiency, Q", "Cross section (nm²)"],
                key="display_mode",
                horizontal=True,
            )

        submitted = st.button("Calculate spectra", use_container_width=True, type="primary")
        st.markdown(
            '<p class="small-note">The selected modes are plotted as individual scattering contributions in the central panel. '
            'The lower panel shows the amplitude and principal phase of their Mie coefficients.</p>',
            unsafe_allow_html=True,
        )

    return {
        "submitted": submitted,
        "source": source,
        "material_name": material_name,
        "uploaded_file": uploaded_file,
        "upload_unit": upload_unit,
        "upload_format": upload_format,
        "user_n": float(st.session_state["user_n"]),
        "user_k": float(st.session_state["user_k"]),
        "sphere_diameter_nm": float(sphere_diameter_nm),
        "radius_nm": float(sphere_diameter_nm) / 2.0,
        "n_medium": float(n_medium),
        "wavelength_min": float(wavelength_min),
        "wavelength_max": float(wavelength_max),
        "samples": int(samples),
        "truncation_method": st.session_state["truncation_method"],
        "manual_order": int(st.session_state["manual_order"]),
        "display_mode": st.session_state["display_mode"],
        "selected_totals": selected_totals_from_state(),
        "selected_modes": selected_modes_from_state(),
    }


def resolve_material(config: dict) -> MaterialData:
    if config["source"] == "Built-in library":
        return load_builtin_material(config["material_name"])
    if config["source"] == "Import RI":
        upload = config["uploaded_file"]
        if upload is None:
            raise ValueError("Import an RI file before calculating.")
        return load_uploaded_material(
            upload.getvalue(),
            upload.name,
            config["upload_unit"],
            config["upload_format"],
        )
    return constant_material(
        config["material_name"],
        config["user_n"],
        config["user_k"],
        config["wavelength_min"],
        config["wavelength_max"],
    )


def calculate_from_config(config: dict) -> dict:
    if config["wavelength_min"] >= config["wavelength_max"]:
        raise ValueError("Minimum wavelength must be smaller than maximum wavelength.")

    material = resolve_material(config)
    wavelength_nm = np.linspace(config["wavelength_min"], config["wavelength_max"], config["samples"])
    optical_index = material.interpolate(wavelength_nm)

    selected_order = max((order for _, _, order in config["selected_modes"]), default=1)
    x_max = 2.0 * np.pi * config["n_medium"] * config["radius_nm"] / config["wavelength_min"]
    recommended_order = wiscombe_order(x_max)
    requested_order = recommended_order if config["truncation_method"] == "Automatic (Wiscombe)" else config["manual_order"]
    max_order = min(max(int(requested_order), selected_order), 100)

    result = calculate_mie_sphere(
        wavelength_nm=wavelength_nm,
        radius_nm=config["radius_nm"],
        n_medium=config["n_medium"],
        particle_index=optical_index,
        max_order=max_order,
    )
    return {"material": material, "result": result, "config": config}


def show_header(payload: dict) -> None:
    config = payload["config"]
    material: MaterialData = payload["material"]
    with st.container(border=True):
        st.markdown('<div class="hero-title">Single Particle Scattering</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Introduction to Mie Theory</div>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Personal project by Muhammad Sujak MK</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="hero-subtitle" style="font-size:0.85rem; margin-top:0.35rem;">'
            f'{material.name} · sphere diameter = {config["sphere_diameter_nm"]:.1f} nm · '
            f'n<sub>medium</sub> = {config["n_medium"]:.3g}</p>',
            unsafe_allow_html=True,
        )


def show_dashboard(payload: dict) -> None:
    material: MaterialData = payload["material"]
    result: dict = payload["result"]
    config: dict = payload["config"]
    wavelength = np.asarray(result["wavelength_nm"])

    show_header(payload)

    with st.container(border=True):
        st.markdown('<div class="panel-title">Scattering, absorption, and extinction spectra</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Qsca, Qabs, and Qext are shown according to the selected check boxes. '
            'Selected modal curves are individual scattering contributions.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            total_spectra_figure(result, config["display_mode"], config["selected_totals"], config["selected_modes"]),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with st.container(border=True):
        st.markdown('<div class="panel-title">Amplitude and phase of selected Mie modes</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">Solid curves: coefficient amplitude |a<sub>ℓ</sub>| or |b<sub>ℓ</sub>| (left axis). '
            'Dashed curves: principal phase (right axis).</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            coefficient_figure(result, config["selected_modes"]),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with st.expander("Download calculated data", expanded=False):
        spectra = spectra_dataframe(result)
        nk_table = material_dataframe(material, wavelength)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download spectra, modes, amplitudes, and phases (.csv)",
                data=spectra.to_csv(index=False).encode("utf-8"),
                file_name="mie_sphere_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download interpolated n, k, ε (.csv)",
                data=nk_table.to_csv(index=False).encode("utf-8"),
                file_name="interpolated_optical_constants.csv",
                mime="text/csv",
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title="Introduction to Mie Theory", page_icon="◌", layout="wide")
    initial_defaults()
    inject_css()

    left, right = st.columns([0.95, 2.05], gap="large")
    with left:
        config = build_controls()

    with right:
        should_calculate = config["submitted"] or "mie_payload" not in st.session_state
        if should_calculate:
            try:
                with st.spinner("Calculating Mie coefficients..."):
                    st.session_state["mie_payload"] = calculate_from_config(config)
            except Exception as exc:
                st.session_state.pop("mie_payload", None)
                st.error(str(exc))

        payload = st.session_state.get("mie_payload")
        if payload is not None:
            show_dashboard(payload)
        else:
            with st.container(border=True):
                st.info("Set the parameters on the left and click **Calculate spectra**.")


if __name__ == "__main__":
    main()
