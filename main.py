import os
import numpy as np
import pandas as pd
import fastf1
from sklearn.linear_model import HuberRegressor
import plotly.graph_objects as go

# Enable FastF1 Cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'f1_cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

def load_and_clean_race_data(year: int, grand_prix: str, session_type: str = 'R') -> pd.DataFrame:
    """Loads F1 race data and filters out invalid laps (SC/VSC, pit laps, outliers)."""
    print(f"Loading session data for {year} {grand_prix} ({session_type})...")
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load(laps=True, telemetry=False, weather=False)
    
    laps = session.laps.pick_accurate().copy()
    valid_laps = laps[laps['TrackStatus'] == '1'].copy()
    valid_laps.loc[:, 'LapTimeSeconds'] = valid_laps['LapTime'].dt.total_seconds()
    
    filtered_laps = []
    for (driver, stint), group in valid_laps.groupby(['Driver', 'Stint']):
        if len(group) < 5:
            continue
        q1 = group['LapTimeSeconds'].quantile(0.25)
        q3 = group['LapTimeSeconds'].quantile(0.75)
        iqr = q3 - q1
        clean_stint = group[(group['LapTimeSeconds'] >= q1 - 1.5 * iqr) & 
                            (group['LapTimeSeconds'] <= q3 + 1.5 * iqr)].copy()
        filtered_laps.append(clean_stint)
        
    return pd.concat(filtered_laps, ignore_index=True)

def calculate_fuel_corrected_laptime(df: pd.DataFrame, fuel_effect_per_lap: float = 0.050) -> pd.DataFrame:
    """Applies realistic fuel correction (~0.050s burnoff adjustment per lap)."""
    df = df.copy()
    df['FuelCorrectedLapTime'] = df['LapTimeSeconds'] + (df['LapNumber'] * fuel_effect_per_lap)
    return df

def fit_degradation_models(stint_df: pd.DataFrame):
    """Fits Huber Robust Linear Regressor across compounds."""
    results = {}
    
    for compound, group in stint_df.groupby('Compound'):
        if compound in ['UNKNOWN', None]:
            continue
            
        X = group[['StintLapNumber']].values
        y = group['FuelCorrectedLapTime'].values
        
        model = HuberRegressor()
        model.fit(X, y)
        deg_per_lap = model.coef_[0]
        
        results[compound] = {
            'model': model,
            'deg_rate_sec_per_lap': deg_per_lap,
            'base_pace': model.intercept_,
            'data': group
        }
        
    return results

def create_interactive_dashboard(model_results: dict, output_filename: str = "tyre_degradation.html"):
    """Generates an interactive Plotly HTML visualizer for portfolio presentation."""
    fig = go.Figure()
    
    colors = {
        'SOFT': '#FF3333',
        'MEDIUM': '#FFF200',
        'HARD': '#FFFFFF',
        'INTERMEDIATE': '#39B54A',
        'WET': '#00AEEF'
    }
    
    for compound, res in model_results.items():
        data = res['data']
        color = colors.get(compound, '#888888')
        
        # 1. Add raw lap data points
        fig.add_trace(go.Scatter(
            x=data['StintLapNumber'],
            y=data['FuelCorrectedLapTime'],
            mode='markers',
            name=f'{compound} Lap Data',
            marker=dict(color=color, opacity=0.35, size=7),
            text=[f"Driver: {d}<br>Stint: {s}<br>Lap: {l}" for d, s, l in zip(data['Driver'], data['Stint'], data['LapNumber'])],
            hoverinfo='text+y'
        ))
        
        # 2. Add linear trend line
        max_stint_lap = int(data['StintLapNumber'].max())
        X_line = np.linspace(1, max_stint_lap, 50).reshape(-1, 1)
        y_line = res['model'].predict(X_line)
        
        fig.add_trace(go.Scatter(
            x=X_line.flatten(),
            y=y_line,
            mode='lines',
            name=f'{compound} Fit (+{res["deg_rate_sec_per_lap"]:.3f}s/l)',
            line=dict(color=color, width=3.5)
        ))

    fig.update_layout(
        template='plotly_dark',
        title=dict(text="F1 Tyre Degradation Curves (Fuel-Corrected Pace)", font=dict(size=20)),
        xaxis=dict(title="Stint Lap Number", gridcolor='#333333'),
        yaxis=dict(title="Fuel-Corrected Lap Time (seconds)", gridcolor='#333333'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.write_html(output_filename)
    print(f"Interactive visual dashboard saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    clean_laps = load_and_clean_race_data(2023, 'Bahrain')
    clean_laps = calculate_fuel_corrected_laptime(clean_laps, fuel_effect_per_lap=0.050)
    
    clean_laps['StintLapNumber'] = clean_laps.groupby(['Driver', 'Stint']).cumcount() + 1
    
    model_results = fit_degradation_models(clean_laps)
    
    print("\n" + "="*55)
    print(f"{'COMPOUND':<12} | {'BASE PACE (s)':<15} | {'DEG / LAP (s)':<15}")
    print("="*55)
    for compound, res in model_results.items():
        print(f"{compound:<12} | {res['base_pace']:<15.3f} | +{res['deg_rate_sec_per_lap']:<14.3f}")
    print("="*55 + "\n")
    
    create_interactive_dashboard(model_results)
