# 🏎️ F1 Tyre Degradation Modeling & Strategy Simulator

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Fast-F1](https://img.shields.io/badge/dependency-Fast--F1-red.svg)
![Streamlit](https://img.shields.io/badge/ui-Streamlit-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end Formula 1 telemetry analysis and race strategy simulation tool built with Python, `FastF1`, `Scikit-Learn`, and `Streamlit`. This application isolates true compound degradation rates by correcting for fuel mass loss and simulates head-to-head race scenarios—including pit stop undercuts—in real time.

---

## 📌 Key Features

- **Fuel Burn-off Correction:** Adjusts raw lap time progression across stints by isolating vehicle mass loss ($\sim 0.035\text{ s/lap}$ fuel burn factor) to evaluate pure tyre degradation.
- **Linear Degradation Modeling:** Fits linear regression models via `Scikit-Learn` to extract precise compound degradation rates ($\text{ms/lap}$) while automatically filtering safety car (SC) and virtual safety car (VSC) anomalies.
- **Interactive Undercut Simulator:** Computes lap-by-lap cumulative gap trajectories between two drivers based on pit stop timing choices, out-lap cold tyre penalties, and pit lane loss deltas.
- **Streamlit Web Dashboard:** Interactive web interface featuring live parameter sliders, compound selection dropdowns, and dynamic gap prediction charts.

---

## 🛠️ Quick Start

### 1. Clone & Install
```bash
git clone [https://github.com/teotudora/f1-tyre-degradation-model.git](https://github.com/teotudora/f1-tyre-degradation-model.git)
cd f1-tyre-degradation-model
pip install -r requirements.txt
