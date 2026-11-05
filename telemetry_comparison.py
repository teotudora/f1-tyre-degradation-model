import warnings
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import numpy as np

# Suppress cache deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

def compare_telemetry(year: int, grand_prix: str, session_type: str, driver1: str, driver2: str):
    # Load session and laps
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()

    # Get fastest lap for each driver
    lap_d1 = session.laps.pick_drivers(driver1).pick_fastest()
    lap_d2 = session.laps.pick_drivers(driver2).pick_fastest()

    # Get telemetry data with distance
    tel_d1 = lap_d1.get_telemetry().add_distance()
    tel_d2 = lap_d2.get_telemetry().add_distance()

    # Calculate Time Delta over distance (interpolate Driver 2 onto Driver 1 distance grid)
    d2_time_interp = np.interp(
        tel_d1['Distance'], 
        tel_d2['Distance'], 
        tel_d2['Time'].dt.total_seconds()
    )
    d1_time = tel_d1['Time'].dt.total_seconds().to_numpy()
    delta_time = d1_time - d2_time_interp

    # Fetch official driver colors
    color_d1 = fastf1.plotting.get_driver_color(driver1, session=session)
    color_d2 = fastf1.plotting.get_driver_color(driver2, session=session)

    # Setup Plotting Grid
    fastf1.plotting.setup_mpl(mpl_style='dark_background')
    fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})

    # 1. Speed Trace
    ax[0].plot(tel_d1['Distance'], tel_d1['Speed'], color=color_d1, label=f"{driver1} ({lap_d1['LapTime'].total_seconds():.3f}s)")
    ax[0].plot(tel_d2['Distance'], tel_d2['Speed'], color=color_d2, label=f"{driver2} ({lap_d2['LapTime'].total_seconds():.3f}s)")
    ax[0].set_ylabel("Speed (km/h)")
    ax[0].set_title(f"Telemetry Comparison: {driver1} vs {driver2} — {year} {grand_prix} ({session_type})", fontsize=14)
    ax[0].legend(loc='lower right')
    ax[0].grid(True, linestyle='--', alpha=0.3)

    # 2. Time Delta Curve
    ax[1].plot(tel_d1['Distance'], delta_time, color='cyan')
    ax[1].axhline(0, color='white', linestyle='--', alpha=0.5)
    ax[1].set_ylabel(f"Delta (s)\n< {driver1} faster | {driver2} faster >")
    ax[1].grid(True, linestyle='--', alpha=0.3)

    # 3. Throttle Trace
    ax[2].plot(tel_d1['Distance'], tel_d1['Throttle'], color=color_d1)
    ax[2].plot(tel_d2['Distance'], tel_d2['Throttle'], color=color_d2)
    ax[2].set_ylabel("Throttle (%)")
    ax[2].grid(True, linestyle='--', alpha=0.3)

    # 4. Brake Trace
    ax[3].plot(tel_d1['Distance'], tel_d1['Brake'], color=color_d1)
    ax[3].plot(tel_d2['Distance'], tel_d2['Brake'], color=color_d2)
    ax[3].set_ylabel("Brake")
    ax[3].set_xlabel("Track Distance (m)")
    ax[3].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("telemetry_comparison.png", dpi=300)
    print("Chart successfully generated and saved as telemetry_comparison.png")
    plt.show()

if __name__ == '__main__':
    compare_telemetry(2024, 'Bahrain', 'Q', 'VER', 'PER')
