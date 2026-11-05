import streamlit as st
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Set Streamlit page config
st.set_page_config(page_title="F1 Analytics & Strategy Suite", layout="wide")
st.title("🏎️ F1 Telemetry & Strategy Suite")

# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

@st.cache_data
def load_session(year, gp, session_type='R'):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session

def get_tyre_deg_rate(session, driver):
    laps = session.laps.pick_drivers(driver).pick_accurate()
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

# Create Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Race Strategy Simulator", "⚡ Telemetry Comparison", "🗺️ 2D Track Heatmap"])

# ---------------- TAB 1: STRATEGY SIMULATOR ----------------
with tab1:
    st.sidebar.header("Race Settings")
    year_strat = st.sidebar.number_input("Year (Strategy)", min_value=2018, max_value=2024, value=2024)
    gp_strat = st.sidebar.text_input("Grand Prix (Strategy)", value="Bahrain")
    pit_loss = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15.0, 30.0, 22.5)

    if st.sidebar.button("Run Strategy Simulation"):
        with st.spinner("Loading race session..."):
            session = load_session(year_strat, gp_strat, 'R')
            total_laps = session.total_laps
            drivers = list(session.laps['Driver'].unique())
            
            st.subheader(f"Strategy Comparison: {gp_strat} GP ({year_strat})")
            
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
            
            d1_times = []
            d1_age = 0
            for lap in laps_seq:
                penalty = pit_loss if lap == d1_pit else 0.0
                base = d1_base.get('MEDIUM', 90.0)
                deg = d1_deg.get('MEDIUM', 0.05)
                lap_time = base + (d1_age * deg) - ((total_laps - lap) * 0.035) + penalty
                d1_times.append(lap_time)
                d1_age = 0 if lap == d1_pit else d1_age + 1

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

# ---------------- TAB 2: TELEMETRY COMPARISON ----------------
with tab2:
    st.subheader("Fastest Lap Telemetry & Speed Delta")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        year_tel = st.number_input("Year", min_value=2018, max_value=2024, value=2024, key="tel_year")
    with col_t2:
        gp_tel = st.text_input("Grand Prix", value="Bahrain", key="tel_gp")
    with col_t3:
        session_type = st.selectbox("Session", ["Q", "R", "FP3"], index=0)

    if st.button("Compare Telemetry"):
        with st.spinner("Fetching lap telemetry data..."):
            session_tel = load_session(year_tel, gp_tel, session_type)
            drivers_tel = list(session_tel.laps['Driver'].unique())

            c1, c2 = st.columns(2)
            with c1:
                driver1 = st.selectbox("Driver 1", drivers_tel, index=0)
            with c2:
                driver2 = st.selectbox("Driver 2", drivers_tel, index=min(1, len(drivers_tel)-1))

            lap_d1 = session_tel.laps.pick_drivers(driver1).pick_fastest()
            lap_d2 = session_tel.laps.pick_drivers(driver2).pick_fastest()

            tel_d1 = lap_d1.get_telemetry().add_distance()
            tel_d2 = lap_d2.get_telemetry().add_distance()

            d2_time_interp = np.interp(
                tel_d1['Distance'], 
                tel_d2['Distance'], 
                tel_d2['Time'].dt.total_seconds()
            )
            d1_time = tel_d1['Time'].dt.total_seconds().to_numpy()
            delta_time = d1_time - d2_time_interp

            color_d1 = fastf1.plotting.get_driver_color(driver1, session=session_tel)
            color_d2 = fastf1.plotting.get_driver_color(driver2, session=session_tel)

            fastf1.plotting.setup_mpl(mpl_style='dark_background')
            fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})

            ax[0].plot(tel_d1['Distance'], tel_d1['Speed'], color=color_d1, label=f"{driver1} ({lap_d1['LapTime'].total_seconds():.3f}s)")
            ax[0].plot(tel_d2['Distance'], tel_d2['Speed'], color=color_d2, label=f"{driver2} ({lap_d2['LapTime'].total_seconds():.3f}s)")
            ax[0].set_ylabel("Speed (km/h)")
            ax[0].set_title(f"{driver1} vs {driver2} Telemetry — {year_tel} {gp_tel} ({session_type})", fontsize=14)
            ax[0].legend(loc='lower right')
            ax[0].grid(True, linestyle='--', alpha=0.3)

            ax[1].plot(tel_d1['Distance'], delta_time, color='cyan')
            ax[1].axhline(0, color='white', linestyle='--', alpha=0.5)
            ax[1].set_ylabel("Delta (s)")
            ax[1].grid(True, linestyle='--', alpha=0.3)

            ax[2].plot(tel_d1['Distance'], tel_d1['Throttle'], color=color_d1)
            ax[2].plot(tel_d2['Distance'], tel_d2['Throttle'], color=color_d2)
            ax[2].set_ylabel("Throttle (%)")
            ax[2].grid(True, linestyle='--', alpha=0.3)

            ax[3].plot(tel_d1['Distance'], tel_d1['Brake'], color=color_d1)
            ax[3].plot(tel_d2['Distance'], tel_d2['Brake'], color=color_d2)
            ax[3].set_ylabel("Brake")
            ax[3].set_xlabel("Track Distance (m)")
            ax[3].grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

# ---------------- TAB 3: 2D TRACK HEATMAP ----------------
with tab3:
    st.subheader("2D Circuit Telemetry Heatmap")
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        map_year = st.number_input("Year", min_value=2018, max_value=2024, value=2024, key="map_year")
    with c_m2:
        map_gp = st.text_input("Grand Prix", value="Bahrain", key="map_gp")
    with c_m3:
        map_driver = st.text_input("Driver Code", value="VER", key="map_driver")
    with c_m4:
        map_metric = st.selectbox("Color By", ["Speed", "Gear"], index=0)

    if st.button("Generate Heatmap"):
        with st.spinner("Processing GPS & Telemetry Data..."):
            session_map = load_session(map_year, map_gp, 'Q')
            lap_map = session_map.laps.pick_drivers(map_driver).pick_fastest()
            tel_map = lap_map.get_telemetry()

            x = tel_map['X'].to_numpy()
            y = tel_map['Y'].to_numpy()

            if map_metric == 'Gear':
                c_vals = tel_map['nGear'].to_numpy()
                cmap_name = 'tab10'
                label_title = "Gear"
            else:
                c_vals = tel_map['Speed'].to_numpy()
                cmap_name = 'plasma'
                label_title = "Speed (km/h)"

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            fastf1.plotting.setup_mpl(mpl_style='dark_background')
            fig_map, ax_map = plt.subplots(figsize=(10, 8))

            norm = plt.Normalize(c_vals.min(), c_vals.max())
            lc = LineCollection(segments, cmap=plt.get_cmap(cmap_name), norm=norm, linewidth=5)
            lc.set_array(c_vals)

            ax_map.add_collection(lc)
            ax_map.autoscale()
            ax_map.set_axis_off()

            cbar = fig_map.colorbar(lc, ax=ax_map, orientation='horizontal', pad=0.05, shrink=0.7)
            cbar.set_label(label_title, fontsize=12)

            ax_map.set_title(f"Track Heatmap ({map_metric}): {map_driver} — {map_year} {map_gp} (Q)\nFastest Lap: {lap_map['LapTime'].total_seconds():.3f}s", fontsize=14)
            
            st.pyplot(fig_map)
