import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

def generate_track_map(year: int, grand_prix: str, session_type: str, driver: str, color_by: str = 'Speed'):
    """Generates a 2D track layout heatmap color-coded by telemetry metrics (Speed or nGear)."""
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()

    # Get fastest lap and position telemetry
    lap = session.laps.pick_drivers(driver).pick_fastest()
    tel = lap.get_telemetry()

    x = tel['X'].to_numpy()
    y = tel['Y'].to_numpy()

    # Color metric selection
    if color_by == 'Gear':
        c_vals = tel['nGear'].to_numpy()
        cmap_name = 'tab10'
        label_title = "Gear"
    else:
        c_vals = tel['Speed'].to_numpy()
        cmap_name = 'plasma'
        label_title = "Speed (km/h)"

    # Create line segments for multi-color track line
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fastf1.plotting.setup_mpl(mpl_style='dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))

    norm = plt.Normalize(c_vals.min(), c_vals.max())
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap_name), norm=norm, linewidth=5)
    lc.set_array(c_vals)

    ax.add_collection(lc)
    ax.autoscale()

    # Hide grid axes for clean map aesthetic
    ax.set_axis_off()

    cbar = fig.colorbar(lc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label(label_title, fontsize=12)

    lap_time_str = f"{lap['LapTime'].total_seconds():.3f}s"
    ax.set_title(f"Track Heatmap ({color_by}): {driver} — {year} {grand_prix} ({session_type})\nFastest Lap: {lap_time_str}", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"track_map_{driver}_{year}.png", dpi=300)
    print(f"Track map successfully saved as track_map_{driver}_{year}.png")
    plt.show()

if __name__ == '__main__':
    # Plot Speed Heatmap for Verstappen's 2024 Bahrain Qualifying lap
    generate_track_map(2024, 'Bahrain', 'Q', 'VER', color_by='Speed')
