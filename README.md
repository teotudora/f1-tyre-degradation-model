# 🏎️ F1 Tyre Degradation Modeling & Strategy Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-tyre-degradation-model-ph7nypxbllmgmzg2wcs3nn.streamlit.app/)

🔗 **Live Demo:** [f1-tyre-degradation-model.streamlit.app](https://f1-tyre-degradation-model-ph7nypxbllmgmzg2wcs3nn.streamlit.app/)

An end-to-end Formula 1 telemetry analysis and race strategy simulation tool built with Python, **FastF1**, **SciPy**, **Scikit-Learn**, and **Streamlit**. This application isolates true compound degradation rates by correcting for fuel mass loss, performs spatial telemetry overlays, and simulates head-to-head race scenarios—including pit stop undercuts—in real time.

---

## 📌 Key Features

* **Fuel Burn-off Correction:** Adjusts raw lap time progression across stints by isolating vehicle mass loss ($\sim 0.035\text{ s/lap}$ fuel burn factor) to evaluate pure tyre degradation.
* **Linear Degradation Modeling:** Fits linear regression models via `SciPy` / `Scikit-Learn` to extract precise compound degradation rates ($\text{s/lap}$) while automatically filtering Safety Car (SC) and Virtual Safety Car (VSC) anomalies.
* **Spatial Telemetry & Delta-Time Integration:** Resamples non-uniform temporal telemetry onto a 1-meter spatial track grid ($d$) to calculate continuous corner-by-corner velocity deltas ($\Delta t$).
* **Interactive Undercut Simulator:** Computes lap-by-lap cumulative gap trajectories between two drivers based on pit stop timing choices, pit lane loss deltas, and tyre degradation slopes.
* **Cloud-Optimized Architecture:** Deployed on Streamlit Cloud utilizing a pre-cached data pipeline (`./f1_cache`) to bypass cloud hosting IP throttling from F1 timing CDNs.

---

## 🛠️ Local Environment & Pre-caching Setup

To run the application locally or refresh the pre-cached session files:

### 1. Activate Environment & Install Dependencies

```bash
# Navigate to project directory
cd f1_tyre_model

# Activate your virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
