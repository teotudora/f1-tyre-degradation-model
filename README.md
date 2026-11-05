# 🏎️ F1 Tyre Degradation Modeling & Strategy Simulator

An end-to-end Formula 1 telemetry analysis tool built with Python, FastF1, and Streamlit. This repository models fuel-corrected tyre degradation rates and simulates head-to-head race strategies (including pit stop undercuts) in real time.

---

## ✨ Features

- **Fuel Burn-off Correction:** Adjusts raw lap times by accounting for vehicle mass loss (~0.035 s/lap) across stints.
- **Tyre Degradation Modeling:** Calculates linear regression degradation rates (ms/lap) per tyre compound using FastF1 data.
- **Interactive Undercut Simulator:** Simulates lap-by-lap cumulative gaps between two drivers based on pit stop lap choices and pit lane loss time.
- **Streamlit Web Dashboard:** Interactive UI with sliders and dropdowns to run custom race simulations.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/teotudora/f1-tyre-degradation-model.git](https://github.com/teotudora/f1-tyre-degradation-model.git)
   cd f1-tyre-degradation-model
