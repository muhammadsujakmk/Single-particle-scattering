# Mie Theory Streamlit App

A bright Streamlit interface for the supplied single-sphere Mie-theory code. The layout follows the requested blueprint:

- **Left panel:** parameter control, built-in/imported/user-defined refractive-index data, medium index, sphere diameter, total-spectrum check boxes, and modal check boxes.
- **Top-right panel:** introduction header and project attribution.
- **Middle-right panel:** selected `Qsca`, `Qabs`, and `Qext` curves together with selected modal scattering contributions.
- **Bottom-right panel:** amplitude and phase of the selected Mie coefficients using a double y-axis.

## Material input choices

1. **Built-in library:** supplied Si, Au, and Ag tables.
2. **Import RI:** a whitespace/tab/comma/semicolon separated text file containing either:
   - wavelength, `n`, `k`, or
   - wavelength, `ε1`, `ε2`.
3. **Define RI by user:** constant `n + ik` over the selected wavelength sweep.

For an imported file, choose `nm` or `µm` explicitly. The application does not silently guess the unit or extrapolate outside a material table.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder.
2. In Streamlit Community Cloud, choose **Create app**.
3. Select the repository and set the entry point to `app.py`.
4. Deploy.

## Numerical notes

- The sphere and host medium are assumed homogeneous and isotropic.
- The host medium is lossless with real refractive index.
- Optical constants use the passive convention `n + ik`, where `k ≥ 0`.
- Material interpolation uses PCHIP and does not extrapolate.
- Automatic calculation order follows the Wiscombe rule. Manual order is available under **Advanced calculation settings**.
- The phase panel displays the principal phase from −180° to +180°.
