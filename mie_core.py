"""Numerically stable Mie-theory utilities for a homogeneous spherical particle."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import spherical_jn, spherical_yn

NM = 1e-9


@dataclass(frozen=True)
class MaterialData:
    """Tabulated optical constants using wavelength in nm and n + i k."""

    wavelength_nm: np.ndarray
    n: np.ndarray
    k: np.ndarray
    name: str = "Material"

    @property
    def wavelength_limits_nm(self) -> tuple[float, float]:
        return float(self.wavelength_nm.min()), float(self.wavelength_nm.max())

    def interpolate(self, target_nm: np.ndarray) -> np.ndarray:
        """Shape-preserving interpolation without extrapolation."""
        target_nm = np.asarray(target_nm, dtype=float)
        lo, hi = self.wavelength_limits_nm
        if np.any(target_nm < lo) or np.any(target_nm > hi):
            raise ValueError(
                f"Requested wavelength range ({target_nm.min():.1f}–{target_nm.max():.1f} nm) "
                f"is outside the material-data range ({lo:.1f}–{hi:.1f} nm)."
            )

        n_interp = PchipInterpolator(self.wavelength_nm, self.n, extrapolate=False)(target_nm)
        k_interp = PchipInterpolator(self.wavelength_nm, self.k, extrapolate=False)(target_nm)
        return np.asarray(n_interp + 1j * k_interp, dtype=complex)


def _clean_numeric_table(raw: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 3:
        raise ValueError(f"{name} must contain at least three numeric columns.")

    raw = raw[:, :3]
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    if len(raw) < 2:
        raise ValueError(f"{name} does not contain at least two valid numeric rows.")

    raw = raw[np.argsort(raw[:, 0])]
    # Average duplicated wavelengths so interpolation is well defined.
    frame = pd.DataFrame(raw, columns=["wavelength", "column_2", "column_3"])
    frame = frame.groupby("wavelength", as_index=False).mean()
    raw = frame.to_numpy(dtype=float)

    if np.any(np.diff(raw[:, 0]) <= 0):
        raise ValueError(f"{name} contains repeated or unsorted wavelength values.")
    return raw


def material_from_array(
    values: np.ndarray,
    *,
    wavelength_unit: Literal["nm", "um"] = "nm",
    optical_format: Literal["n_k", "epsilon"] = "n_k",
    name: str = "Material",
) -> MaterialData:
    """Create MaterialData from columns [wavelength, n, k] or [wavelength, eps1, eps2]."""
    values = _clean_numeric_table(values, name)
    wavelength_nm = values[:, 0].copy()
    if wavelength_unit == "um":
        wavelength_nm *= 1000.0
    elif wavelength_unit != "nm":
        raise ValueError("wavelength_unit must be 'nm' or 'um'.")

    c2 = values[:, 1]
    c3 = values[:, 2]
    if optical_format == "n_k":
        n = c2
        k = c3
    elif optical_format == "epsilon":
        # Passive-medium branch: epsilon = (n + i k)^2, k >= 0.
        abs_eps = np.hypot(c2, c3)
        n = np.sqrt(np.maximum((abs_eps + c2) / 2.0, 0.0))
        k = np.sqrt(np.maximum((abs_eps - c2) / 2.0, 0.0))
    else:
        raise ValueError("optical_format must be 'n_k' or 'epsilon'.")

    if np.any(wavelength_nm <= 0):
        raise ValueError("All wavelengths must be positive.")
    if np.any(n < 0) or np.any(k < 0):
        raise ValueError("For passive materials, n and k must be non-negative.")

    order = np.argsort(wavelength_nm)
    return MaterialData(
        wavelength_nm=wavelength_nm[order],
        n=np.asarray(n[order], dtype=float),
        k=np.asarray(k[order], dtype=float),
        name=name,
    )


def load_material_file(
    source: str | Path | BytesIO,
    *,
    wavelength_unit: Literal["nm", "um"],
    optical_format: Literal["n_k", "epsilon"] = "n_k",
    name: str = "Material",
) -> MaterialData:
    """Read whitespace-, tab-, comma-, or semicolon-separated three-column data."""
    try:
        frame = pd.read_csv(
            source,
            sep=r"[\s,;]+",
            engine="python",
            comment="#",
            header=None,
            usecols=[0, 1, 2],
        )
    except Exception as exc:
        raise ValueError(f"Could not read {name}: {exc}") from exc

    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    return material_from_array(
        numeric.to_numpy(),
        wavelength_unit=wavelength_unit,
        optical_format=optical_format,
        name=name,
    )


def wiscombe_order(size_parameter: float) -> int:
    """Recommended truncation order for a sphere with real external size parameter x."""
    x = float(size_parameter)
    if x <= 0:
        return 1
    return max(1, int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0)))


def _riccati_bessel(order: int, z: complex) -> tuple[complex, complex, complex, complex]:
    """Return psi, psi', xi, xi' for Riccati-Bessel functions at z."""
    jn = spherical_jn(order, z)
    jn_d = spherical_jn(order, z, derivative=True)
    yn = spherical_yn(order, z)
    yn_d = spherical_yn(order, z, derivative=True)

    h1 = jn + 1j * yn
    h1_d = jn_d + 1j * yn_d
    psi = z * jn
    psi_d = jn + z * jn_d
    xi = z * h1
    xi_d = h1 + z * h1_d
    return psi, psi_d, xi, xi_d


def mie_coefficients(x: float, m: complex, order: int) -> tuple[complex, complex]:
    """Exact Mie coefficients a_l (electric) and b_l (magnetic)."""
    psi_x, psi_x_d, xi_x, xi_x_d = _riccati_bessel(order, complex(x))
    psi_mx, psi_mx_d, _, _ = _riccati_bessel(order, complex(m * x))

    a_num = m * psi_mx * psi_x_d - psi_x * psi_mx_d
    a_den = m * psi_mx * xi_x_d - xi_x * psi_mx_d
    b_num = psi_mx * psi_x_d - m * psi_x * psi_mx_d
    b_den = psi_mx * xi_x_d - m * xi_x * psi_mx_d

    with np.errstate(divide="ignore", invalid="ignore"):
        a = a_num / a_den
        b = b_num / b_den
    return complex(a), complex(b)


def mode_label(kind: Literal["E", "M"], order: int) -> str:
    known = {
        ("E", 1): "ED",
        ("M", 1): "MD",
        ("E", 2): "EQ",
        ("M", 2): "MQ",
        ("E", 3): "EO",
        ("M", 3): "MO",
    }
    return known.get((kind, order), f"{kind}{order}")


def calculate_mie_sphere(
    wavelength_nm: np.ndarray,
    radius_nm: float,
    n_medium: float,
    particle_index: np.ndarray,
    max_order: int,
) -> dict[str, np.ndarray | int | float]:
    """Calculate total and modal cross sections for a sphere in a homogeneous, lossless host.

    Returned cross sections are in m^2. Efficiencies are normalized by pi r^2.
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    particle_index = np.asarray(particle_index, dtype=complex)
    if wavelength_nm.ndim != 1 or particle_index.shape != wavelength_nm.shape:
        raise ValueError("wavelength_nm and particle_index must be one-dimensional arrays of equal length.")
    if len(wavelength_nm) < 2:
        raise ValueError("At least two wavelength samples are required.")
    if radius_nm <= 0 or n_medium <= 0 or max_order < 1:
        raise ValueError("radius_nm, n_medium, and max_order must be positive.")

    radius_m = radius_nm * NM
    wavelength_m = wavelength_nm * NM
    area_m2 = np.pi * radius_m**2
    k_host = 2.0 * np.pi * n_medium / wavelength_m
    x_values = k_host * radius_m

    n_points = len(wavelength_nm)
    orders = np.arange(1, max_order + 1)
    a_values = np.zeros((max_order, n_points), dtype=complex)
    b_values = np.zeros((max_order, n_points), dtype=complex)

    for i, (x, index) in enumerate(zip(x_values, particle_index)):
        relative_index = index / n_medium
        for l_idx, order in enumerate(orders):
            a_values[l_idx, i], b_values[l_idx, i] = mie_coefficients(x, relative_index, int(order))

    weights = (2.0 * orders + 1.0)[:, None]
    prefactor = 2.0 * np.pi / k_host**2

    ext_e = prefactor * np.real(weights * a_values)
    ext_m = prefactor * np.real(weights * b_values)
    sca_e = prefactor * weights * np.abs(a_values) ** 2
    sca_m = prefactor * weights * np.abs(b_values) ** 2
    abs_e = ext_e - sca_e
    abs_m = ext_m - sca_m

    c_ext = np.sum(ext_e + ext_m, axis=0)
    c_sca = np.sum(sca_e + sca_m, axis=0)
    c_abs = c_ext - c_sca

    result: dict[str, np.ndarray | int | float] = {
        "wavelength_nm": wavelength_nm,
        "n": particle_index.real,
        "k": particle_index.imag,
        "max_order": int(max_order),
        "recommended_order": wiscombe_order(float(np.max(x_values))),
        "area_m2": area_m2,
        "c_ext_m2": c_ext,
        "c_sca_m2": c_sca,
        "c_abs_m2": c_abs,
        "q_ext": c_ext / area_m2,
        "q_sca": c_sca / area_m2,
        "q_abs": c_abs / area_m2,
        "a": a_values,
        "b": b_values,
        "ext_e_m2": ext_e,
        "ext_m_m2": ext_m,
        "sca_e_m2": sca_e,
        "sca_m_m2": sca_m,
        "abs_e_m2": abs_e,
        "abs_m_m2": abs_m,
    }
    return result


def spectra_dataframe(result: dict[str, np.ndarray | int | float]) -> pd.DataFrame:
    """Build a CSV-ready table with total and modal spectra."""
    data: dict[str, np.ndarray] = {
        "wavelength_nm": np.asarray(result["wavelength_nm"]),
        "n": np.asarray(result["n"]),
        "k": np.asarray(result["k"]),
        "Q_ext": np.asarray(result["q_ext"]),
        "Q_sca": np.asarray(result["q_sca"]),
        "Q_abs": np.asarray(result["q_abs"]),
        "C_ext_nm2": np.asarray(result["c_ext_m2"]) * 1e18,
        "C_sca_nm2": np.asarray(result["c_sca_m2"]) * 1e18,
        "C_abs_nm2": np.asarray(result["c_abs_m2"]) * 1e18,
    }
    max_order = int(result["max_order"])
    for l in range(1, max_order + 1):
        idx = l - 1
        for kind, field, coefficient_field in (("E", "sca_e_m2", "a"), ("M", "sca_m_m2", "b")):
            label = mode_label(kind, l)
            coefficient = np.asarray(result[coefficient_field])[idx]
            data[f"Q_sca_{label}"] = np.asarray(result[field])[idx] / float(result["area_m2"])
            data[f"amplitude_{label}"] = np.abs(coefficient)
            data[f"phase_{label}_deg"] = np.angle(coefficient, deg=True)
    return pd.DataFrame(data)


def material_dataframe(material: MaterialData, target_nm: np.ndarray) -> pd.DataFrame:
    index = material.interpolate(target_nm)
    return pd.DataFrame(
        {
            "wavelength_nm": np.asarray(target_nm),
            "n": index.real,
            "k": index.imag,
            "epsilon_1": index.real**2 - index.imag**2,
            "epsilon_2": 2.0 * index.real * index.imag,
        }
    )
