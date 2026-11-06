import warnings
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

def analyze_lando_lap(year: int = 2024, grand_prix: str = 'Zandvoort', rival: str = 'VER'):
    """Performs a comprehensive telemetry and track dominance analysis for Lando Norris."""
    print(f"Fetching session data for {year} {grand_prix} GP...")
    session = fastf1.get_session(year, grand_prix, 'Q')
    session.load()

    # Pick fastest qualifying laps
    lap_nor = session.laps.pick_drivers('NOR').pick_fastest()
    lap_riv = session.laps.pick_drivers(rival).pick_fastest()

    # Get telemetry data
    tel_nor = lap_nor.get_telemetry().add_distance()
    tel_riv = lap_riv.get_telemetry().add_distance()

    # Fetch official team colors
    color_nor = fastf1.plotting.get_driver_color('NOR', session=session)
    color_riv = fastf1.plotting.get_driver_color(rival, session=session)

    # 1. Calculate time delta over distance
    riv_time_interp = np.interp(
        tel_nor['Distance'], 
        tel_riv['Distance'], 
        tel_riv['Time'].dt.total_seconds()
    )
    nor_time = tel_nor['Time'].dt.total_seconds().to_numpy()
    delta_time = nor_time - riv_time_interp  # Negative values = NOR is ahead/faster

    fastf1.plotting.setup_mpl(mpl_style='dark_background')
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[1.2, 1])

    # Subplot A: Speed Comparison Trace
    ax_speed = fig.add_subplot(gs[0, 0])
    ax_speed.plot(tel_nor['Distance'], tel_nor['Speed'], color=color_nor, label=f"NOR ({lap_nor['LapTime'].total_seconds():.3f}s)", linewidth=2)
    ax_speed.plot(tel_riv['Distance'], tel_riv['Speed'], color=color_riv, label=f"{rival} ({lap_riv['LapTime'].total_seconds():.3f}s)", linewidth=2, linestyle='--')
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_title(f"Lando Norris vs {rival} — {year} {grand_prix} Qualifying", fontsize=14, color='papayawhip')
    ax_speed.legend(loc='lower right')
    ax_speed.grid(True, linestyle='--', alpha=0.3)

    # Subplot B: Time Delta Curve
    ax_delta = fig.add_subplot(gs[1, 0], sharex=ax_speed)
    ax_delta.plot(tel_nor['Distance'], delta_time, color='#FF8000', linewidth=2)  # McLaren Papaya Orange
    ax_delta.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax_delta.set_ylabel(f"Delta (s)\n< NOR faster | {rival} faster >")
    ax_delta.grid(True, linestyle='--', alpha=0.3)

    # Subplot C: Throttle Application
    ax_throttle = fig.add_subplot(gs[2, 0], sharex=ax_speed)
    ax_throttle.plot(tel_nor['Distance'], tel_nor['Throttle'], color=color_nor, label="NOR Throttle")
    ax_throttle.plot(tel_riv['Distance'], tel_riv['Throttle'], color=color_riv, linestyle='--', label=f"{rival} Throttle")
    ax_throttle.set_ylabel("Throttle (%)")
    ax_throttle.set_xlabel("Track Distance (m)")
    ax_throttle.grid(True, linestyle='--', alpha=0.3)

    # Subplot D: 2D Track Map (Norris Speed Profile)
    ax_map = fig.add_subplot(gs[:, 1])
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

    cbar = fig.colorbar(lc, ax=ax_map, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label("Lando Norris Telemetry Speed (km/h)", fontsize=11)
    ax_map.set_title(f"Track Layout: {grand_prix}", fontsize=13)

    plt.tight_layout()
    output_filename = f"lando_vs_{rival}_{grand_prix}_{year}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Lando Norris telemetry analysis saved as {output_filename}")
    plt.show()

if __name__ == '__main__':
    # Defaulting to Lando's dominant 2024 Zandvoort Pole position vs Verstappen
    analyze_lando_lap(2024, 'Zandvoort', 'VER')
