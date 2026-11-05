import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

# Set Streamlit page config
st.set_page_config(page_title="F1 Race Strategy Simulator", layout="wide")
st.title("🏎️ F1 Race Strategy & Undercut Simulator")

# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

@st.cache_data
def load_session(year, gp):
    session = fastf1.get_session(year, gp, 'R')
    session.load()
    return session

def get_tyre_deg_rate(session, driver):
    laps = session.laps.pick_driver(driver).pick_accurate()
    laps = laps[laps['TrackStatus'] == '1'].copy()
    
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    total_laps = session.total_laps
    laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] + ((total_laps - laps['LapNumber']) * 0.035)

    deg_rates = {}
    base_times = {}

    for compound in laps['Compound'].unique():
        if pd.isna(compound) or compound in ['UNKNOWN', 'TEST']:
            continue
        compound_laps = laps[laps['Compound'] == compound]
        if len(compound_laps) < 3:
            continue

        slope, intercept, _, _, _ = linregress(compound_laps['TyreLife'], compound_laps['FuelCorrectedLapTime'])
        deg_rates[compound] = slope
        base_times[compound] = intercept

    return deg_rates, base_times

# Sidebar parameters
st.sidebar.header("Race Settings")
year = st.sidebar.number_input("Year", min_value=2018, max_value=2024, value=2024)
gp = st.sidebar.text_input("Grand Prix", value="Bahrain")
pit_loss = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15.0, 30.0, 22.5)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Loading telemetry and running simulation..."):
        session = load_session(year, gp)
        total_laps = session.total_laps

        drivers = list(session.laps['Driver'].unique())
        
        st.subheader(f"Strategy Comparison: {gp} GP ({year})")
        
        col1, col2 = st.columns(2)
        with col1:
            d1 = st.selectbox("Driver 1 (Undercutter)", drivers, index=0)
            d1_pit = st.slider(f"{d1} Pit Lap", 1, total_laps, 14)
        with col2:
            d2 = st.selectbox("Driver 2 (Target)", drivers, index=min(1, len(drivers)-1))
            d2_pit = st.slider(f"{d2} Pit Lap", 1, total_laps, 18)

        d1_deg, d1_base = get_tyre_deg_rate(session, d1)
        d2_deg, d2_base = get_tyre_deg_rate(session, d2)

        laps_seq = np.arange(1, total_laps + 1)
        
        # Simulate Driver 1
        d1_times = []
        d1_age = 0
        for lap in laps_seq:
            penalty = pit_loss if lap == d1_pit else 0.0
            base = d1_base.get('MEDIUM', 90.0)
            deg = d1_deg.get('MEDIUM', 0.05)
            lap_time = base + (d1_age * deg) - ((total_laps - lap) * 0.035) + penalty
            d1_times.append(lap_time)
            d1_age = 0 if lap == d1_pit else d1_age + 1

        # Simulate Driver 2
        d2_times = []
        d2_age = 0
        for lap in laps_seq:
            penalty = pit_loss if lap == d2_pit else 0.0
            base = d2_base.get('MEDIUM', 90.0)
            deg = d2_deg.get('MEDIUM', 0.05)
            lap_time = base + (d2_age * deg) - ((total_laps - lap) * 0.035) + penalty
            d2_times.append(lap_time)
            d2_age = 0 if lap == d2_pit else d2_age + 1

        gap = np.cumsum(d1_times) - np.cumsum(d2_times)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=laps_seq, y=gap, mode='lines+markers', name=f'{d1} vs {d2} Gap', line=dict(color='#E10600', width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Equal Position")
        fig.update_layout(title=f"Gap Delta Over Race ({d1} vs {d2})", xaxis_title="Lap", yaxis_title=f"Gap (s) — Negative = {d1} Ahead", template="plotly_dark")
        
        st.plotly_chart(fig, use_container_width=True)
