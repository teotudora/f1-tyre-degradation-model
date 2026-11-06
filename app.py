import streamlit as st
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="F1 Analytics & Strategy Suite", layout="wide")
st.title("🏎️ F1 Telemetry & Strategy Suite")

# Safe Cache Initialization for Streamlit Cloud
cache_dir = '/tmp/f1_cache' if os.path.exists('/tmp') else './f1_cache_dir'
try:
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
except Exception:
    pass

@st.cache_data
def load_session(year, gp, session_type='R'):
    """Loads session data with robust fallback handling."""
    session = fastf1.get_session(int(year), str(gp).strip(), session_type)
    try:
        session.load()
    except Exception:
        try:
            session.load(telemetry=False, weather=False, messages=False)
        except Exception:
            pass
    return session

def is_laps_loaded(session):
    """Safely checks if laps data is loaded without triggering DataNotLoadedError on FastF1 property methods."""
    try:
        return session.laps is not None and not session.laps.empty
    except Exception:
        return False

def get_total_laps(session):
    """Safely derive total race laps without raising DataNotLoadedError."""
    try:
        if hasattr(session, 'total_laps') and session.total_laps is not None:
            return int(session.total_laps)
    except Exception:
        pass
    try:
        if is_laps_loaded(session):
            return int(session.laps['LapNumber'].max())
    except Exception:
        pass
    return 57  # Standard default fallback

def get_tyre_deg_rate(session, driver, total_laps):
    try:
        laps = session.laps.pick_drivers(driver).pick_accurate()
        laps = laps[laps['TrackStatus'] == '1'].copy()
        
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
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
    except Exception:
        return {'MEDIUM': 0.05}, {'MEDIUM': 90.0}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Race Strategy Simulator", 
    "⚡ Telemetry Comparison", 
    "🗺️ 2D Track Heatmap",
    "🏎️ Lando Norris Analysis",
    "🤖 ML Finishing Predictor"
])

# ---------------- TAB 1: STRATEGY SIMULATOR ----------------
with tab1:
    st.sidebar.header("Race Settings")
    year_strat = st.sidebar.number_input("Year (Strategy)", min_value=2018, max_value=2024, value=2024)
    gp_strat = st.sidebar.text_input("Grand Prix (Strategy)", value="Bahrain")
    pit_loss = st.sidebar.slider("Pit Stop Time Loss (seconds)", 15.0, 30.0, 22.5)

    if 'run_sim' not in st.session_state:
        st.session_state.run_sim = False

    if st.sidebar.button("Run Strategy Simulation"):
        st.session_state.run_sim = True

    if st.session_state.run_sim:
        with st.spinner("Loading race session..."):
            session = load_session(year_strat, gp_strat, 'R')
            
            if not is_laps_loaded(session):
                st.error(f"⚠️ Session data for '{gp_strat} {year_strat}' could not be loaded. F1 servers might be throttling cloud IPs. Try another GP (e.g., Monaco, Spa, Monza).")
            else:
                total_laps = get_total_laps(session)
                drivers = list(session.laps['Driver'].unique())
                
                st.subheader(f"Strategy Comparison: {gp_strat} GP ({year_strat})")
                
                col1, col2 = st.columns(2)
                with col1:
                    d1 = st.selectbox("Driver 1 (Undercutter)", drivers, index=0, key="sim_d1")
                    d1_pit = st.slider(f"{d1} Pit Lap", 1, total_laps, max(1, int(total_laps * 0.3)), key="sim_d1_pit")
                with col2:
                    d2 = st.selectbox("Driver 2 (Target)", drivers, index=min(1, len(drivers)-1), key="sim_d2")
                    d2_pit = st.slider(f"{d2} Pit Lap", 1, total_laps, max(1, int(total_laps * 0.4)), key="sim_d2_pit")

                d1_deg, d1_base = get_tyre_deg_rate(session, d1, total_laps)
                d2_deg, d2_base = get_tyre_deg_rate(session, d2, total_laps)

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

    if 'run_tel' not in st.session_state:
        st.session_state.run_tel = False

    if st.button("Compare Telemetry"):
        st.session_state.run_tel = True

    if st.session_state.run_tel:
        with st.spinner("Fetching lap telemetry data..."):
            session_tel = load_session(year_tel, gp_tel, session_type)
            if not is_laps_loaded(session_tel):
                st.error(f"Telemetry data currently unavailable for {gp_tel} ({year_tel}).")
            else:
                drivers_tel = list(session_tel.laps['Driver'].unique())

                c1, c2 = st.columns(2)
                with c1:
                    driver1 = st.selectbox("Driver 1", drivers_tel, index=0, key="tel_d1")
                with c2:
                    driver2 = st.selectbox("Driver 2", drivers_tel, index=min(1, len(drivers_tel)-1), key="tel_d2")

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
            if not is_laps_loaded(session_map):
                st.error("Heatmap telemetry data currently unavailable for this session.")
            else:
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

# ---------------- TAB 4: LANDO NORRIS ANALYSIS ----------------
with tab4:
    st.subheader("Lando Norris vs Rival Telemetry Profile")
    col_ln1, col_ln2, col_ln3 = st.columns(3)
    with col_ln1:
        ln_year = st.number_input("Year", min_value=2018, max_value=2024, value=2024, key="ln_year")
    with col_ln2:
        ln_gp = st.text_input("Grand Prix", value="Zandvoort", key="ln_gp")
    with col_ln3:
        ln_rival = st.text_input("Rival Driver Code", value="VER", key="ln_rival")

    if st.button("Run Lando Norris Analysis"):
        with st.spinner("Analyzing Lando Norris telemetry..."):
            session_ln = load_session(ln_year, ln_gp, 'Q')
            if not is_laps_loaded(session_ln):
                st.error("Session data unavailable for Norris comparison.")
            else:
                lap_nor = session_ln.laps.pick_drivers('NOR').pick_fastest()
                lap_riv = session_ln.laps.pick_drivers(ln_rival).pick_fastest()

                tel_nor = lap_nor.get_telemetry().add_distance()
                tel_riv = lap_riv.get_telemetry().add_distance()

                color_nor = fastf1.plotting.get_driver_color('NOR', session=session_ln)
                color_riv = fastf1.plotting.get_driver_color(ln_rival, session=session_ln)

                riv_time_interp = np.interp(
                    tel_nor['Distance'], 
                    tel_riv['Distance'], 
                    tel_riv['Time'].dt.total_seconds()
                )
                nor_time = tel_nor['Time'].dt.total_seconds().to_numpy()
                delta_time = nor_time - riv_time_interp

                fastf1.plotting.setup_mpl(mpl_style='dark_background')
                fig_ln = plt.figure(figsize=(16, 9))
                gs = fig_ln.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[1.2, 1])

                ax_speed = fig_ln.add_subplot(gs[0, 0])
                ax_speed.plot(tel_nor['Distance'], tel_nor['Speed'], color=color_nor, label=f"NOR ({lap_nor['LapTime'].total_seconds():.3f}s)", linewidth=2)
                ax_speed.plot(tel_riv['Distance'], tel_riv['Speed'], color=color_riv, label=f"{ln_rival} ({lap_riv['LapTime'].total_seconds():.3f}s)", linewidth=2, linestyle='--')
                ax_speed.set_ylabel("Speed (km/h)")
                ax_speed.set_title(f"Lando Norris vs {ln_rival} — {ln_year} {ln_gp} Qualifying", fontsize=14, color='#FF8000')
                ax_speed.legend(loc='lower right')
                ax_speed.grid(True, linestyle='--', alpha=0.3)

                ax_delta = fig_ln.add_subplot(gs[1, 0], sharex=ax_speed)
                ax_delta.plot(tel_nor['Distance'], delta_time, color='#FF8000', linewidth=2)
                ax_delta.axhline(0, color='white', linestyle='--', alpha=0.5)
                ax_delta.set_ylabel(f"Delta (s)\n< NOR faster | {ln_rival} faster >")
                ax_delta.grid(True, linestyle='--', alpha=0.3)

                ax_throttle = fig_ln.add_subplot(gs[2, 0], sharex=ax_speed)
                ax_throttle.plot(tel_nor['Distance'], tel_nor['Throttle'], color=color_nor, label="NOR Throttle")
                ax_throttle.plot(tel_riv['Distance'], tel_riv['Throttle'], color=color_riv, linestyle='--', label=f"{ln_rival} Throttle")
                ax_throttle.set_ylabel("Throttle (%)")
                ax_throttle.set_xlabel("Track Distance (m)")
                ax_throttle.grid(True, linestyle='--', alpha=0.3)

                ax_map = fig_ln.add_subplot(gs[:, 1])
                x = tel_nor['X'].to_numpy()
                y = tel_nor['Y'].to_numpy()
                speeds = tel_nor['Speed'].to_numpy()

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                norm = plt.Normalize(speeds.min(), speeds.max())
                lc = LineCollection(segments, cmap=plt.get_cmap('plasma'), norm=norm, linewidth=4)
                lc.set_array(speeds)

                ax_map.add_collection(lc)
                ax_map.autoscale()
                ax_map.set_axis_off()

                cbar = fig_ln.colorbar(lc, ax=ax_map, orientation='horizontal', pad=0.05, shrink=0.7)
                cbar.set_label("Lando Norris Telemetry Speed (km/h)", fontsize=11)
                ax_map.set_title(f"Track Layout: {ln_gp}", fontsize=13)

                plt.tight_layout()
                st.pyplot(fig_ln)

# ---------------- TAB 5: ML FINISHING PREDICTOR ----------------
with tab5:
    st.subheader("Machine Learning Race Finish Position Predictor")
    st.write("Predict a driver's final race position based on starting grid slot and qualifying pace gap.")

    c_ml1, c_ml2 = st.columns(2)
    with c_ml1:
        input_grid = st.slider("Starting Grid Position", 1, 20, 3)
    with c_ml2:
        input_q_delta = st.number_input("Qualifying Pace Gap to Pole (seconds)", min_value=0.0, max_value=5.0, value=0.25, step=0.05)

    if st.button("Predict Race Finish"):
        X_train = np.array([
            [1, 0.0], [2, 0.1], [3, 0.25], [4, 0.35], [5, 0.5],
            [6, 0.65], [7, 0.8], [8, 0.95], [10, 1.2], [15, 2.0], [20, 3.5]
        ])
        y_train = np.array([1.2, 2.1, 3.4, 4.2, 5.1, 6.0, 7.3, 8.5, 9.8, 14.1, 18.2])

        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)

        pred_pos = rf.predict([[input_grid, input_q_delta]])[0]

        st.success(f"Predicted Finishing Position: P{max(1, int(round(pred_pos)))} (Raw value: P{pred_pos:.2f})")
