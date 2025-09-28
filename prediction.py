import os
import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -----------------------------
# Setup cache
# -----------------------------
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# -----------------------------
# Past 2025 GPs (up to Dutch GP)
# -----------------------------
past_gps = [
   (2025, 'Australian Grand Prix'),
   (2025, 'Chinese Grand Prix'),
   (2025, 'Japanese Grand Prix'),
   (2025, 'Bahrain Grand Prix'),
   (2025, 'Saudi Arabian Grand Prix'),
   (2025, 'Miami Grand Prix'),
   (2025, 'Emilia Romagna Grand Prix'),
   (2025, 'Monaco Grand Prix'),
   (2025, 'Spanish Grand Prix'),
   (2025, 'Canadian Grand Prix'),
   (2025, 'Austrian Grand Prix'),
   (2025, 'British Grand Prix'),
   (2025, 'Belgian Grand Prix'),
   (2025, 'Hungarian Grand Prix'),
]

# -----------------------------
# Next GP to predict
# -----------------------------
next_gp = (2025, 'Dutch Grand Prix')
try:
    race_session = fastf1.get_session(next_gp[0], next_gp[1], 'R')
    race_session.load()
    qualifying_session = fastf1.get_session(next_gp[0], next_gp[1], 'Q')
    qualifying_session.load()
    driver_codes = race_session.results['Abbreviation'].tolist()
except Exception:
    print(f"⚠️ {next_gp[1]} hasn't happened yet. Using drivers from last GP instead.")
    last_year, last_gp_name = past_gps[-1]
    last_session = fastf1.get_session(last_year, last_gp_name, 'R')
    last_session.load()
    driver_codes = last_session.results['Abbreviation'].tolist()
    qualifying_session = fastf1.get_session(last_year, last_gp_name, 'Q')
    qualifying_session.load()

print("Drivers considered:", driver_codes)

# -----------------------------
# Helper: average finish across all past GPs
# -----------------------------
def get_avg_finish(driver_code, gps):
    finishes = []
    for year, gp_name in gps:
        try:
            session = fastf1.get_session(year, gp_name, 'R')
            session.load()
            laps = session.laps
            pos = laps[laps['Driver'] == driver_code]['Position'].iloc[-1]
            finishes.append(pos)
        except Exception:
            continue
    return np.mean(finishes) if finishes else None

# -----------------------------
# Helper: get current standings
# -----------------------------
def get_driver_standings():
    standings = {}
    try:
        last_year, last_gp_name = past_gps[-1]
        session = fastf1.get_session(last_year, last_gp_name, 'R')
        session.load()
        results = session.results
        for i, row in results.iterrows():
            standings[row['Abbreviation']] = {
                'Points': row.get('Points', 0),
                'StandingPos': row.get('Position', 0)
            }
    except Exception:
        pass
    return standings

standings = get_driver_standings()

# -----------------------------
# Build driver features
# -----------------------------
data = []
for code in driver_codes:
    avg_finish = get_avg_finish(code, past_gps)

    # Qualifying delta vs pole
    try:
        driver_q = qualifying_session.laps.pick_drivers(code)
        pole_time = qualifying_session.laps['LapTime'].min()
        q_delta = (driver_q['LapTime'].min() - pole_time).total_seconds() if not driver_q.empty else None
    except Exception:
        q_delta = None

    # Avg speed in last race
    try:
        driver_laps = qualifying_session.laps.pick_drivers(code)
        avg_speed = driver_laps['SpeedST'].mean() if 'SpeedST' in driver_laps.columns else None
    except Exception:
        avg_speed = None

    # Standing info
    points = standings.get(code, {}).get('Points', 0)
    standing_pos = standings.get(code, {}).get('StandingPos', 20)

    data.append({
        'DriverCode': code,
        'Avg_finish_pos': avg_finish,
        'Qualifying_delta_pole': q_delta,
        'Avg_speed': avg_speed,
        'Points': points,
        'StandingPos': standing_pos
    })

df = pd.DataFrame(data)

# Drop drivers with missing data
df = df.dropna(subset=['Avg_finish_pos', 'Qualifying_delta_pole'])

# Encode driver codes
le = LabelEncoder()
df['Driver_enc'] = le.fit_transform(df['DriverCode'])

# -----------------------------
# Placeholder target
# -----------------------------
df['FinishingPos'] = np.random.randint(1, 21, len(df))

X = df[['Avg_finish_pos', 'Qualifying_delta_pole', 'Avg_speed', 'Points', 'StandingPos', 'Driver_enc']]
y = df['FinishingPos']

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Predict finishing positions
df['PredictedPos'] = model.predict(X)

# -----------------------------
# Top 3 prediction
# -----------------------------
medals = ["🥇", "🥈", "🥉"]
top3 = df.sort_values('PredictedPos').head(3).reset_index(drop=True)

print(f"\nTop 3 predicted finishers for {next_gp[1]}:")
for i, row in top3.iterrows():
    last_name = row['DriverCode']
    print(f"{i+1}) {last_name} {medals[i]}")
