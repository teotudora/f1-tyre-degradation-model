import warnings
import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
fastf1.Cache.enable_cache('f1_cache')

def fetch_season_features(year: int):
    """Fetches and builds predictive feature matrix from FastF1 with fast lightweight loading."""
    print(f"Extracting race & qualifying features for {year} season...")
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['EventFormat'] == 'conventional']

    feature_rows = []

    for _, event in races.iterrows():
        gp_name = event['EventName']
        try:
            session_q = fastf1.get_session(year, gp_name, 'Q')
            session_r = fastf1.get_session(year, gp_name, 'R')
            
            # Fast loading: Disable telemetry, weather, and race control messages
            session_q.load(telemetry=False, weather=False, messages=False)
            session_r.load(telemetry=False, weather=False, messages=False)

            pole_lap = session_q.laps.pick_fastest()
            pole_time = pole_lap['LapTime'].total_seconds()

            r_results = session_r.results

            for _, driver in r_results.iterrows():
                driver_code = driver['Abbreviation']
                grid_pos = driver['GridPosition']
                finish_pos = driver['Position']

                if pd.isna(grid_pos) or pd.isna(finish_pos) or grid_pos == 0:
                    continue

                driver_q_laps = session_q.laps.pick_drivers(driver_code)
                if len(driver_q_laps) == 0:
                    continue
                
                fastest_q = driver_q_laps.pick_fastest()
                if pd.isna(fastest_q['LapTime']):
                    continue
                
                q_time = fastest_q['LapTime'].total_seconds()
                q_delta = q_time - pole_time

                feature_rows.append({
                    'Year': year,
                    'GrandPrix': gp_name,
                    'Driver': driver_code,
                    'GridPosition': grid_pos,
                    'QualifyingDelta': q_delta,
                    'FinishPosition': finish_pos
                })
        except Exception as e:
            continue

    return pd.DataFrame(feature_rows)


def train_finishing_model(df: pd.DataFrame):
    """Trains a Random Forest Regressor model on lightweight F1 feature data."""
    X = df[['GridPosition', 'QualifyingDelta']]
    y = df['FinishPosition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"\nModel Training Complete!")
    print(f"Mean Absolute Error (MAE): {mae:.2f} positions")

    return model

if __name__ == '__main__':
    df_data = fetch_season_features(2024)
    if not df_data.empty:
        model = train_finishing_model(df_data)
